//! Queued prompts: keep accepting user input while an agent turn runs.
//!
//! Typing during a turn used to be impossible — the REPL only read input
//! between turns, and Ctrl-C was the only way to say something mid-flight,
//! killing the agent's flow. While a turn is active, a small collector
//! thread reads stdin in raw mode and queues completed lines. When the turn
//! finishes, queued lines are delivered as the next user message with a
//! note saying they were typed while the agent was working.
//!
//! Ctrl-C keeps its meaning: with text on the line it clears the line (like
//! the idle prompt); on an empty line it interrupts the turn. Queued lines
//! are dropped when the turn is interrupted.

use goose::config::Config;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Whether queued prompts are enabled. Opt out with GOOSE_QUEUED_PROMPTS=false.
fn enabled() -> bool {
    Config::global()
        .get_param::<bool>("GOOSE_QUEUED_PROMPTS")
        .unwrap_or(true)
}

/// Byte-level state machine for one queued-input line.
///
/// Pure logic, separated from the terminal I/O so it can be unit tested:
/// feed raw bytes, get back what should happen (echo, submit, cancel...).
#[derive(Default)]
struct LineInput {
    partial: Vec<u8>,
    escape: EscapeState,
    line_started: bool,
}

#[derive(Default, PartialEq, Eq)]
enum EscapeState {
    #[default]
    None,
    SawEscape,
    SawIntro,
}

enum InputAction {
    /// Nothing visible happened (byte swallowed or empty enter).
    None,
    /// Print a marker and start echoing a fresh line.
    StartLine,
    /// Echo this raw byte as part of the line.
    EchoByte,
    /// Backspace: drop the last byte from the partial line.
    Backspace,
    /// Enter: the partial line is complete and should be queued.
    SubmitLine,
    /// Ctrl-C on a non-empty line: clear the line, keep queueing.
    ClearLine,
    /// Ctrl-C on an empty line: interrupt the running turn.
    Interrupt,
}

impl LineInput {
    fn feed(&mut self, byte: u8) -> InputAction {
        match self.escape {
            EscapeState::SawIntro => {
                // Swallow the rest of a CSI/SS3 sequence: parameter bytes
                // 0x30-0x3F, intermediates 0x20-0x2F, final 0x40-0x7E.
                if (0x40..=0x7e).contains(&byte) {
                    self.escape = EscapeState::None;
                }
                return InputAction::None;
            }
            EscapeState::SawEscape => {
                if byte == b'[' || byte == b'O' {
                    self.escape = EscapeState::SawIntro;
                    return InputAction::None;
                }
                // Alt-modified key: drop the modifier and treat the byte
                // as a plain keystroke.
                self.escape = EscapeState::None;
            }
            EscapeState::None => {}
        }

        match byte {
            b'\r' | b'\n' => {
                if self.partial.is_empty() {
                    InputAction::None
                } else {
                    InputAction::SubmitLine
                }
            }
            0x7f | 0x08 => {
                if self.partial.is_empty() {
                    InputAction::None
                } else {
                    self.partial.pop();
                    InputAction::Backspace
                }
            }
            0x03 => {
                if self.partial.is_empty() {
                    InputAction::Interrupt
                } else {
                    InputAction::ClearLine
                }
            }
            0x1b => {
                self.escape = EscapeState::SawEscape;
                InputAction::None
            }
            0x04 => InputAction::None,
            0x00..=0x1f => InputAction::None,
            _ => {
                if self.partial.is_empty() && !self.line_started {
                    self.line_started = true;
                    self.partial.push(byte);
                    InputAction::StartLine
                } else {
                    self.partial.push(byte);
                    InputAction::EchoByte
                }
            }
        }
    }

    fn take_line(&mut self) -> String {
        let line = String::from_utf8_lossy(&self.partial).trim().to_string();
        self.partial.clear();
        self.line_started = false;
        line
    }

    fn clear(&mut self) {
        self.partial.clear();
        self.line_started = false;
    }
}

struct Shared {
    state: Mutex<LineInput>,
    lines: Mutex<Vec<String>>,
    paused: AtomicBool,
    stop: AtomicBool,
}

/// Handle returned by [`start`] when a collector is running.
pub struct Collector {
    shared: Arc<Shared>,
    cancel_token: CancellationToken,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// RAII pause: while held, stdin is back in canonical mode for whatever the
/// main thread wants to ask (permission prompts, elicitation) and the
/// collector does not read.
pub struct PauseGuard {
    shared: Arc<Shared>,
    #[cfg(unix)]
    raw: libc::termios,
}

impl Drop for PauseGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.raw);
            }
            set_stdin_nonblocking(true);
        }
        self.shared.paused.store(false, Ordering::Relaxed);
    }
}

impl Collector {
    /// Suspend collecting and hand the terminal back to the main thread.
    pub fn pause_guard(&self) -> PauseGuard {
        self.shared.paused.store(true, Ordering::Relaxed);
        #[cfg(unix)]
        {
            let mut original = std::mem::MaybeUninit::uninit();
            unsafe {
                libc::tcgetattr(libc::STDIN_FILENO, original.as_mut_ptr());
                let original = original.assume_init();
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &original);
            }
            set_stdin_nonblocking(false);
            PauseGuard {
                shared: Arc::clone(&self.shared),
                raw: raw_termios(),
            }
        }
        #[cfg(not(unix))]
        {
            PauseGuard {
                shared: Arc::clone(&self.shared),
            }
        }
    }

    /// Stop the collector and return (completed lines, unfinished partial,
    /// whether the user interrupted the turn with Ctrl-C).
    pub fn stop(mut self) -> (Vec<String>, String, bool) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let lines = std::mem::take(&mut *self.shared.lines.lock().unwrap());
        let partial = self
            .shared
            .state
            .lock()
            .map(|state| String::from_utf8_lossy(&state.partial).trim().to_string())
            .unwrap_or_default();
        let interrupted = self.cancel_token.is_cancelled();
        (lines, partial, interrupted)
    }
}

/// Start a queued-input collector for the duration of one agent turn.
/// Returns `None` when queueing does not apply: non-interactive runs,
/// piped stdin, a disabled config, or non-unix platforms.
pub fn start(interactive: bool, cancel_token: CancellationToken) -> Option<Collector> {
    if !interactive || !enabled() || !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return None;
    }
    #[cfg(not(unix))]
    {
        return None;
    }
    #[cfg(unix)]
    {
        let shared = Arc::new(Shared {
            state: Mutex::new(LineInput::default()),
            lines: Mutex::new(Vec::new()),
            paused: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let thread_shared = Arc::clone(&shared);
        let thread_token = cancel_token.clone();
        let handle = std::thread::Builder::new()
            .name("queued-input".into())
            .spawn(move || run(thread_shared, thread_token))
            .ok()?;
        Some(Collector {
            shared,
            cancel_token,
            handle: Some(handle),
        })
    }
}

#[cfg(unix)]
fn raw_termios() -> libc::termios {
    let mut termios = std::mem::MaybeUninit::uninit();
    unsafe {
        libc::tcgetattr(libc::STDIN_FILENO, termios.as_mut_ptr());
        let mut raw = termios.assume_init();
        libc::cfmakeraw(&mut raw);
        raw
    }
}

#[cfg(unix)]
fn set_stdin_nonblocking(nonblocking: bool) {
    unsafe {
        let flags = libc::fcntl(libc::STDIN_FILENO, libc::F_GETFL);
        if flags < 0 {
            return;
        }
        let flags = if nonblocking {
            flags | libc::O_NONBLOCK
        } else {
            flags & !libc::O_NONBLOCK
        };
        libc::fcntl(libc::STDIN_FILENO, libc::F_SETFL, flags);
    }
}

/// Terminal configuration restored when the collector thread exits.
#[cfg(unix)]
struct TerminalRestore {
    original: libc::termios,
}

#[cfg(unix)]
impl Drop for TerminalRestore {
    fn drop(&mut self) {
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.original);
            set_stdin_nonblocking(false);
        }
    }
}

#[cfg(unix)]
fn run(shared: Arc<Shared>, cancel_token: CancellationToken) {
    let mut original = std::mem::MaybeUninit::uninit();
    unsafe {
        if libc::tcgetattr(libc::STDIN_FILENO, original.as_mut_ptr()) != 0 {
            return;
        }
    }
    let original = unsafe { original.assume_init() };
    let _restore = TerminalRestore { original };

    let raw = raw_termios();
    unsafe {
        libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw);
    }
    set_stdin_nonblocking(true);

    let mut pollfd = libc::pollfd {
        fd: libc::STDIN_FILENO,
        events: libc::POLLIN,
        revents: 0,
    };
    let mut buffer = [0u8; 128];

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        if shared.paused.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(25));
            continue;
        }
        let ready = unsafe { libc::poll(&mut pollfd, 1, 100) };
        if ready <= 0 {
            continue;
        }
        if shared.stop.load(Ordering::Relaxed) || shared.paused.load(Ordering::Relaxed) {
            continue;
        }
        let read = unsafe {
            libc::read(
                libc::STDIN_FILENO,
                buffer.as_mut_ptr() as *mut libc::c_void,
                buffer.len(),
            )
        };
        if read <= 0 {
            continue;
        }
        for byte in &buffer[..read as usize] {
            handle_byte(&shared, *byte, &cancel_token);
        }
        let _ = std::io::stdout().flush();
    }
}

#[cfg(unix)]
fn handle_byte(shared: &Shared, byte: u8, cancel_token: &CancellationToken) {
    let Ok(mut state) = shared.state.lock() else {
        return;
    };
    let action = state.feed(byte);
    let mut stdout = std::io::stdout();
    match action {
        InputAction::None => {}
        InputAction::StartLine => {
            let _ = stdout.write_all("\r\n  \u{00bb} ".as_bytes());
            let _ = stdout.write_all(&[byte]);
        }
        InputAction::EchoByte => {
            let _ = stdout.write_all(&[byte]);
        }
        InputAction::Backspace => {
            let _ = stdout.write_all("\u{8} \u{8}".as_bytes());
        }
        InputAction::SubmitLine => {
            let line = state.take_line();
            drop(state);
            if !line.is_empty() {
                if let Ok(mut lines) = shared.lines.lock() {
                    lines.push(line.clone());
                }
            }
            let note = format!("\r\n  \u{2713} queued: {line}\r\n");
            let _ = stdout.write_all(console::style(note).dim().to_string().as_bytes());
        }
        InputAction::ClearLine => {
            state.clear();
            let _ = stdout.write_all(
                "\r\n  \u{00bb} (line cleared, keep typing or Ctrl-C to interrupt)\r\n".as_bytes(),
            );
        }
        InputAction::Interrupt => {
            cancel_token.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_from(bytes: &[u8]) -> (String, Vec<InputAction>) {
        let mut input = LineInput::default();
        let mut actions = Vec::new();
        for byte in bytes {
            actions.push(input.feed(*byte));
        }
        (input.take_line(), actions)
    }

    #[test]
    fn plain_text_then_enter_submits_the_line() {
        let (line, actions) = line_from(b"hello\r");
        assert_eq!(line, "hello");
        assert!(matches!(actions.last(), Some(InputAction::SubmitLine)));
    }

    #[test]
    fn first_printable_byte_marks_line_start() {
        let mut input = LineInput::default();
        assert!(matches!(input.feed(b'h'), InputAction::StartLine));
        assert!(matches!(input.feed(b'i'), InputAction::EchoByte));
    }

    #[test]
    fn backspace_removes_the_last_byte() {
        let mut input = LineInput::default();
        for byte in b"abc" {
            let _ = input.feed(*byte);
        }
        assert!(matches!(input.feed(0x7f), InputAction::Backspace));
        let (line, _) = line_from(b"\r");
        let mut fresh = LineInput::default();
        for byte in b"ab" {
            let _ = fresh.feed(*byte);
        }
        let _ = fresh.feed(b'\r');
        assert_eq!(input.take_line(), "ab");
        assert_eq!(fresh.take_line(), "ab");
        assert_eq!(line, "");
    }

    #[test]
    fn escape_sequences_are_swallowed() {
        let mut input = LineInput::default();
        for byte in b"\x1b[D" {
            let _ = input.feed(*byte); // left arrow
        }
        assert!(matches!(input.feed(b'x'), InputAction::StartLine));
        let _ = input.feed(b'\r');
        assert_eq!(input.take_line(), "x");
    }

    #[test]
    fn alt_modifier_does_not_eat_the_character() {
        let mut input = LineInput::default();
        let _ = input.feed(0x1b); // Alt
        assert!(matches!(input.feed(b'x'), InputAction::StartLine));
    }

    #[test]
    fn ctrl_c_interrupts_only_on_an_empty_line() {
        let mut input = LineInput::default();
        assert!(matches!(input.feed(0x03), InputAction::Interrupt));
        let _ = input.feed(b'x');
        assert!(matches!(input.feed(0x03), InputAction::ClearLine));
    }

    #[test]
    fn enter_on_empty_line_does_nothing() {
        let mut input = LineInput::default();
        assert!(matches!(input.feed(b'\r'), InputAction::None));
        assert!(matches!(input.feed(b'\n'), InputAction::None));
    }

    #[test]
    fn utf8_multibyte_survives_the_byte_buffer() {
        let mut input = LineInput::default();
        for byte in "héllo".bytes() {
            let _ = input.feed(byte);
        }
        let _ = input.feed(b'\r');
        assert_eq!(input.take_line(), "héllo");
    }
}
