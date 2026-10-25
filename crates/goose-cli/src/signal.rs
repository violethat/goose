use std::future::Future;
use std::pin::Pin;
use tokio::signal;

#[cfg(unix)]
pub fn shutdown_signal() -> Pin<Box<dyn Future<Output = ()> + Send>> {
    Box::pin(async move {
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("failed to install signal handler")
                .recv()
                .await;
        };

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
    })
}

#[cfg(not(unix))]
pub fn shutdown_signal() -> Pin<Box<dyn Future<Output = ()> + Send>> {
    Box::pin(async move {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    })
}

/// Keep `SIGWINCH` away from every thread until the prompt asks for it.
///
/// The line editor only learns that the terminal changed size when the signal
/// interrupts the `read()` it is parked in, and the kernel hands a
/// process-directed signal to any thread that will take it. On a multi-threaded
/// runtime that is rarely the thread owning the prompt, which then keeps painting
/// against a column count the terminal no longer has. Masking the signal before
/// the process has a second thread leaves the prompt thread as the only one
/// willing to receive it.
#[cfg(unix)]
pub fn mask_window_change() {
    let mut window_change = zeroed_set();
    unsafe {
        libc::sigaddset(&mut window_change, libc::SIGWINCH);
        libc::pthread_sigmask(libc::SIG_BLOCK, &window_change, std::ptr::null_mut());
    }
}

#[cfg(not(unix))]
pub fn mask_window_change() {}

/// Receive `SIGWINCH` on the calling thread until the returned guard is dropped.
#[cfg(unix)]
#[must_use = "the prompt thread only sees resizes while the guard is alive"]
pub struct WindowChangeSignals(libc::sigset_t);

#[cfg(not(unix))]
#[must_use = "the prompt thread only sees resizes while the guard is alive"]
pub struct WindowChangeSignals;

#[cfg(unix)]
pub fn receive_window_change() -> WindowChangeSignals {
    let mut window_change = zeroed_set();
    let mut previous = zeroed_set();
    unsafe {
        libc::sigaddset(&mut window_change, libc::SIGWINCH);
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &window_change, &mut previous);
    }
    WindowChangeSignals(previous)
}

#[cfg(not(unix))]
pub fn receive_window_change() -> WindowChangeSignals {
    WindowChangeSignals
}

#[cfg(unix)]
impl Drop for WindowChangeSignals {
    fn drop(&mut self) {
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &self.0, std::ptr::null_mut());
        }
    }
}

/// Let a child that takes over the terminal see resizes again. The mask survives
/// `execve`, so an editor or pager started by goose would otherwise never be
/// told the window moved.
///
/// # Safety
///
/// `cmd` must not be spawned concurrently with this call.
#[cfg(unix)]
pub unsafe fn forward_window_change_to(cmd: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;
    cmd.pre_exec(|| {
        let mut window_change = zeroed_set();
        unsafe {
            libc::sigaddset(&mut window_change, libc::SIGWINCH);
            if libc::pthread_sigmask(libc::SIG_UNBLOCK, &window_change, std::ptr::null_mut()) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(())
    });
}

#[cfg(not(unix))]
pub unsafe fn forward_window_change_to(_cmd: &mut std::process::Command) {}

#[cfg(unix)]
fn zeroed_set() -> libc::sigset_t {
    let mut set: libc::sigset_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::sigemptyset(&mut set);
    }
    set
}
