// Race two async operations and return whichever finishes first.
//
// Goal:
// - Implement `first_finished` using `tokio::select!`.
// - Ensure losing branch is cancelled automatically.

use tokio::time::{sleep, Duration};

async fn fast() -> &'static str {
    sleep(Duration::from_millis(30)).await;
    "fast"
}

async fn slow() -> &'static str {
    sleep(Duration::from_millis(120)).await;
    "slow"
}

async fn first_finished() -> &'static str {
    // TODO: Use `tokio::select!` to await both futures and return the faster one.
    // Expected output in this setup is "fast".
    todo!()
}

#[tokio::test(flavor = "current_thread")]
async fn tokio4_select_first() {
    assert_eq!(first_finished().await, "fast");
}
