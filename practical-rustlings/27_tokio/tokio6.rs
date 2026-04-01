// Coordinate worker shutdown with `tokio::sync::broadcast`.
//
// Goal:
// - Spawn 3 workers that process ticks until receiving a shutdown signal.
// - Broadcast one shutdown message and ensure all workers exit.

use tokio::sync::broadcast;
use tokio::time::{sleep, Duration};

async fn worker(mut shutdown_rx: broadcast::Receiver<()>) -> u32 {
    let mut ticks = 0_u32;
    loop {
        tokio::select! {
            _ = sleep(Duration::from_millis(10)) => {
                ticks += 1;
            }
            _ = shutdown_rx.recv() => {
                break;
            }
        }
    }
    ticks
}

#[tokio::test(flavor = "current_thread")]
async fn tokio6_broadcast_shutdown() {
    let (shutdown_tx, shutdown_rx) = broadcast::channel(8);

    // TODO: spawn 3 workers, cloning `shutdown_rx` as needed.
    // TODO: wait briefly, send shutdown signal, await workers.
    // Assert each worker ticked at least once before shutdown.

    let _ = (shutdown_tx, shutdown_rx);
}
