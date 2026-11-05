#![cfg(unix)]

//! Byte-level verification of the queued-input trailing line in a real pty:
//! it prints once, stays silent while unchanged, replaces in a single write,
//! never draws while the cursor is mid-line, and erases on demand.

use super::*;
use std::fs::File;
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const CHILD_MODE: &str = "GOOSE_QUEUED_LINE_TEST";

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn own_descriptor(fd: libc::c_int) -> File {
    let file = unsafe { File::from_raw_fd(fd) };
    assert_eq!(
        unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) },
        0
    );
    file
}

fn set_winsize(fd: libc::c_int, rows: u16, cols: u16) {
    let size = libc::winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    assert_eq!(unsafe { libc::ioctl(fd, libc::TIOCSWINSZ as _, &size) }, 0);
}

/// Read from the pty master until `needle` appears, returning everything.
fn read_until(output: &mut File, needle: &str, deadline: Duration) -> Vec<u8> {
    let end = Instant::now() + deadline;
    let mut bytes = Vec::new();
    while !bytes.windows(needle.len()).any(|w| w == needle.as_bytes()) {
        assert!(
            Instant::now() < end,
            "waiting for {needle:?}, got {bytes:?}"
        );
        let mut pollfd = libc::pollfd {
            fd: output.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut pollfd, 1, 100) };
        assert!(ready >= 0);
        if ready > 0 {
            let mut chunk = [0u8; 512];
            let read = output.read(&mut chunk).unwrap();
            assert!(read > 0, "child closed the pty: {bytes:?}");
            bytes.extend_from_slice(&chunk[..read]);
        }
    }
    bytes
}

fn occurrences(bytes: &[u8], needle: &str) -> usize {
    bytes
        .windows(needle.len())
        .filter(|w| *w == needle.as_bytes())
        .count()
}

#[test]
fn trailing_line_is_stable_through_a_stream() {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "session::output::queued_line_signals::child",
            "--nocapture",
        ])
        .env(CHILD_MODE, "1");

    let mut master = -1;
    let mut slave = -1;
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
    let stdout_fd = unsafe { libc::dup(slave) };
    assert!(stdout_fd >= 0);
    let mut master = own_descriptor(master);
    let stdin = own_descriptor(slave);
    let stdout = own_descriptor(stdout_fd);
    command.stdin(Stdio::from(stdin));
    command.stdout(Stdio::from(stdout));
    set_winsize(master.as_raw_fd(), 24, 80);

    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let child = ChildGuard(command.spawn().unwrap());
    let all = read_until(&mut master, "DONE", Duration::from_secs(10));

    // The first sync prints the indicator once...
    assert_eq!(occurrences(&all, "» first"), 1);
    // ...and the unchanged sync that follows emitted nothing (the READY and
    // UNCHANGED_TICK markers surround only one indicator line).
    // A new text replaces the visible line in a single erase+print write.
    assert_eq!(occurrences(&all, "» second"), 1);
    assert!(occurrences(&all, "\x1b[1A\x1b[2K") >= 1);
    // Mid-stream (cursor not at a line start) drawing is suppressed entirely.
    assert_eq!(occurrences(&all, "» hidden"), 0);
    assert_eq!(occurrences(&all, "MID_STREAM"), 1);
    // A line start is not enough: until the stream goes quiet the
    // indicator stays hidden.
    assert_eq!(occurrences(&all, "» hidden-too"), 0);
    assert_eq!(occurrences(&all, "STILL_STREAMING"), 1);
    // Back at a line start, drawing resumes, and clearing erases the line.
    assert_eq!(occurrences(&all, "» third"), 1);
    drop(child);
}

#[test]
fn child() {
    let Ok(mode) = std::env::var(CHILD_MODE) else {
        return;
    };
    assert_eq!(mode, "1");

    ::std::println!("READY");
    queued_line_sync(Some("» first".into()));
    // Unchanged content must not emit any bytes: this is the anti-flicker
    // guarantee while the user's partial line settles between ticks.
    queued_line_sync(Some("» first".into()));
    ::std::println!("UNCHANGED_TICK");

    // The discipline around a streamed chunk: erase first, print, redraw.
    queued_line_erase();
    ::std::println!("chunk");
    queued_line_sync(Some("» second".into()));

    // Reasoning deltas leave the cursor mid-line: drawing is suppressed.
    queued_line_mid_stream();
    queued_line_sync(Some("» hidden".into()));
    ::std::println!("MID_STREAM");

    // A line start alone is not enough: the stream must also be quiet.
    queued_line_at_line_start();
    queued_line_sync(Some("» hidden-too".into()));
    ::std::println!("STILL_STREAMING");
    std::thread::sleep(std::time::Duration::from_millis(700));
    queued_line_sync(Some("» third".into()));
    queued_line_sync(None);
    ::std::println!("DONE");
}
