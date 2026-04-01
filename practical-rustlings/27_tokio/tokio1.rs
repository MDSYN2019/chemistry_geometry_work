// Spawn several async tasks and gather their results.
//
// Goal:
// - Spawn 10 tasks.
// - Each task waits ~100ms and returns its task index.
// - Await all JoinHandles and collect all values into `results`.

use tokio::time::{sleep, Duration};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut handles = Vec::new();

    for i in 0..10_u32 {
        let handle = tokio::spawn(async move {
            sleep(Duration::from_millis(100)).await;
            i
        });
        handles.push(handle);
    }

    let mut results = Vec::new();

    for handle in handles {
        // TODO: Await each handle and push the successful value into `results`.
        // Hint: `tokio::spawn` returns `JoinHandle<T>` and awaiting gives `Result<T, JoinError>`.
        let _ = handle;
    }

    results.sort_unstable();
    assert_eq!(results, (0..10_u32).collect::<Vec<_>>());
}
