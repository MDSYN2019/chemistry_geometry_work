// Update shared state from multiple async tasks.
//
// Goal:
// - Spawn 10 tasks.
// - Each task increments a shared counter once.
// - Await all tasks, then assert the counter is 10.

use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // TODO: `std::sync::Mutex` can block the executor.
    // Replace with an async-aware mutex from `tokio::sync`.
    let counter = Arc::new(tokio::sync::Mutex::new(0_u32));

    let mut handles = Vec::new();

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        handles.push(tokio::spawn(async move {
            // TODO: Lock, increment, and drop the guard before task end.
            let mut guard = counter.lock().await;
            *guard += 1;
        }));
    }

    for handle in handles {
        handle.await.expect("task should complete");
    }

    let final_count = *counter.lock().await;
    assert_eq!(final_count, 10);
}
