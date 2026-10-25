#![cfg(unix)]

//! Regression test for resizing the terminal while the REPL prompt is up.
//!
//! The prompt is read on a runtime worker thread, but the kernel hands a
//! process-directed SIGWINCH to whichever thread is willing to take it, so the
//! signal has to be steered to the thread parked in the line editor's `read()`.
//! Otherwise the editor keeps painting against a column count the terminal no
//! longer has and the prompt ends up scrambled.

use super::*;
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const CHILD_MODE: &str = "GOOSE_TERMINAL_RESIZE_TEST";
const CHILD_TEST: &str = "session::input::resize_tests::resize_child";
const ROWS: u16 = 24;
const WIDE_COLS: u16 = 80;
const NARROW_COLS: u16 = 40;
/// Long enough to occupy two rows at the wide size, which is what a resize has
/// to re-layout.
const WRAPPING_INPUT: &str = "ABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJ\
ABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJABCDEFGHIJ";
/// Rustyline turns bracketed paste on as soon as it owns the terminal.
const RAW_MODE: &[u8] = b"\x1b[?2004h";
/// A line erase, emitted while repainting rows the editor had not counted before.
const REPAINT: &[u8] = b"\x1b[K";

struct Prompt {
    child: Child,
    tty: File,
    // The CLI writes its config under here, and the child outlives the test body.
    _home: tempfile::TempDir,
}

impl Prompt {
    fn spawn() -> Self {
        let home = tempfile::tempdir().unwrap();
        let mut master = -1;
        let mut slave = -1;
        // openpty initializes both descriptors; File takes ownership after success.
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            0
        );
        let tty = own_descriptor(master);
        set_winsize(&tty, ROWS, WIDE_COLS);
        let slave = own_descriptor(slave);

        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", CHILD_TEST, "--nocapture"])
            .env(CHILD_MODE, "1")
            .env("GOOSE_HOME", home.path())
            .env("GOOSE_DISABLE_KEYRING", "1")
            .env("TERM", "xterm-256color")
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave))
            .stderr(Stdio::null());
        unsafe {
            // Only async-signal-safe syscalls run between fork and exec.
            command.pre_exec(|| {
                if libc::setsid() < 0 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(io::Error::last_os_error());
                }
                // Closing the test's PTY master must not kill the child during shutdown.
                libc::signal(libc::SIGHUP, libc::SIG_IGN);
                // Mask SIGWINCH before the child grows a second thread, exactly as
                // main() does: only the thread reading the prompt may claim it.
                crate::signal::mask_window_change();
                Ok(())
            });
        }
        let child = command.spawn().unwrap();
        Self {
            child,
            tty,
            _home: home,
        }
    }

    fn child_gone(&mut self) -> bool {
        self.child.try_wait().unwrap().is_some()
    }

    fn write(&mut self, bytes: &[u8]) {
        if self.child_gone() {
            panic!("prompt child exited before the terminal was resized");
        }
        self.tty.write_all(bytes).unwrap();
        self.tty.flush().unwrap();
    }

    fn type_slowly(&mut self, text: &str) {
        for ch in text.chars() {
            self.write(ch.encode_utf8(&mut [0; 4]).as_bytes());
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Read until `needle` has been painted, so actions only follow a settled prompt.
    fn wait_for_paint(&mut self, needle: &[u8], timeout: Duration) {
        let mut seen: Vec<u8> = Vec::new();
        let deadline = Instant::now() + timeout;
        while !contains(&seen, needle) {
            if self.child_gone() {
                panic!("prompt child exited while waiting for {needle:?}, saw {seen:?}");
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "timed out waiting for {needle:?}, saw {seen:?}"
            );
            if readable(self.tty.as_raw_fd(), remaining) {
                seen.extend_from_slice(&read_chunk(&mut self.tty));
            }
        }
    }

    /// Everything painted until the terminal goes quiet, or `overall` elapses.
    fn paint(&mut self, quiet: Duration, overall: Duration) -> Vec<u8> {
        let mut bytes = Vec::new();
        let deadline = Instant::now() + overall;
        while Instant::now() < deadline && !self.child_gone() {
            if !readable(self.tty.as_raw_fd(), quiet) {
                break;
            }
            bytes.extend_from_slice(&read_chunk(&mut self.tty));
        }
        bytes
    }

    fn resize(&mut self, cols: u16) {
        set_winsize(&self.tty, ROWS, cols);
    }

    /// Close the prompt with its own exit gestures, draining output as we go so the
    /// child is never left waiting on a full terminal buffer.
    fn shutdown(&mut self) {
        for keystroke in [&b"\r"[..], &b"\x03"[..], &b"\x03"[..], &b"\x04"[..]] {
            self.tty.write_all(keystroke).ok();
            self.tty.flush().ok();
            std::thread::sleep(Duration::from_millis(300));
            drain(&mut self.tty);
            if self.child_gone() {
                return;
            }
        }
    }
}

impl Drop for Prompt {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn own_descriptor(fd: libc::c_int) -> File {
    // Only descriptors returned by a successful openpty/dup reach this helper.
    let file = unsafe { File::from_raw_fd(fd) };
    assert_eq!(
        unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) },
        0
    );
    file
}

fn set_winsize(tty: &File, rows: u16, cols: u16) {
    let size = libc::winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    assert_eq!(
        unsafe { libc::ioctl(tty.as_raw_fd(), libc::TIOCSWINSZ as _, &size) },
        0
    );
}

fn readable(fd: libc::c_int, within: Duration) -> bool {
    let mut descriptor = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let millis = within.as_millis().clamp(1, i32::MAX as u128) as i32;
    match unsafe { libc::poll(&mut descriptor, 1, millis) } {
        -1 => panic!("{}", io::Error::last_os_error()),
        ready => ready > 0,
    }
}

fn read_chunk(reader: &mut File) -> Vec<u8> {
    let mut chunk = [0u8; 512];
    match reader.read(&mut chunk) {
        Ok(0) => panic!("the prompt child closed the terminal"),
        Ok(n) => chunk[..n].to_vec(),
        Err(e) if e.kind() == io::ErrorKind::Interrupted => Vec::new(),
        Err(e) => panic!("{e}"),
    }
}

/// Read whatever is pending without caring about the child's state.
fn drain(reader: &mut File) {
    let mut chunk = [0u8; 512];
    while readable(reader.as_raw_fd(), Duration::from_millis(100)) {
        if reader.read(&mut chunk).unwrap_or(0) == 0 {
            break;
        }
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn resize_child() {
    if std::env::var(CHILD_MODE).is_err() {
        return;
    }
    // Mirror the shipped process shape: the prompt is read on a worker thread of a
    // multi-threaded runtime, while SIGWINCH was masked before threads existed.
    let handle = std::thread::Builder::new()
        .name("goose-cli-main".to_string())
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                let mut editor = create_editor(
                    None,
                    Arc::new(std::sync::RwLock::new(CompletionCache::new())),
                )
                .unwrap();
                println!("READY");
                io::stdout().flush().unwrap();
                while let Ok(input) = get_input(&mut editor, None) {
                    match input {
                        InputResult::Message(text) => println!("LINE:{text}"),
                        InputResult::Exit => break,
                        _ => {}
                    }
                }
                io::stdout().flush().unwrap();
            })
        })
        .unwrap();
    let _ = handle.join();
}

#[test]
fn prompt_relayouts_when_the_terminal_is_resized() {
    let mut prompt = Prompt::spawn();
    prompt.wait_for_paint(b"READY", Duration::from_secs(60));
    prompt.wait_for_paint(RAW_MODE, Duration::from_secs(30));

    // A prompt that occupies two rows at 80 columns.
    prompt.type_slowly(WRAPPING_INPUT);
    // Drain the painting of the typed text so nothing below is credited to it.
    prompt.paint(Duration::from_millis(300), Duration::from_secs(10));

    // Narrowing the terminal signals SIGWINCH to the child's foreground group.
    prompt.resize(NARROW_COLS);
    let repaint = prompt.paint(Duration::from_millis(300), Duration::from_secs(10));
    assert!(
        contains(&repaint, REPAINT),
        "resizing a wrapped prompt without a keystroke painted nothing: {repaint:?}"
    );

    // The re-layout has to be a redraw, not a reset: the text is still there.
    prompt.write(b"\r");
    prompt.wait_for_paint(
        format!("LINE:{WRAPPING_INPUT}").as_bytes(),
        Duration::from_secs(30),
    );

    prompt.shutdown();
}
