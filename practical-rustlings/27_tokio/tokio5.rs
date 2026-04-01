// Apply timeout + retry policy to a flaky async operation.
//
// Goal:
// - `flaky_call` fails twice, then succeeds.
// - Wrap each attempt in `tokio::time::timeout`.
// - Retry up to `max_attempts` and return the first success.

use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::time::{sleep, timeout, Duration};

static ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

async fn flaky_call() -> Result<u32, &'static str> {
    sleep(Duration::from_millis(20)).await;
    let n = ATTEMPTS.fetch_add(1, Ordering::SeqCst);
    if n < 2 {
        Err("transient")
    } else {
        Ok(42)
    }
}

async fn call_with_retry(max_attempts: usize) -> Result<u32, &'static str> {
    // TODO: Add a loop with timeout-wrapped attempts.
    // - timeout budget per attempt: 100ms
    // - retry on timeout or "transient"
    // - stop after `max_attempts`
    todo!()
}

#[tokio::test(flavor = "current_thread")]
async fn tokio5_retry_succeeds() {
    ATTEMPTS.store(0, Ordering::SeqCst);
    let value = call_with_retry(5).await.expect("should eventually succeed");
    assert_eq!(value, 42);
}
