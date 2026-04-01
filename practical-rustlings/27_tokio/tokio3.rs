// Build a tiny async producer -> consumer pipeline.
//
// Goal:
// - Producer sends numbers 1..=5 over an async channel.
// - Consumer receives all numbers and returns their sum.
// - Ensure sender is dropped so receiver loop can terminate.

use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

async fn produce(tx: mpsc::Sender<u32>) {
    // TODO: Send all numbers from 1 to 5.
    // Add a short sleep between sends so behavior is observable.
    for n in 1..=5_u32 {
        tx.send(n).await.expect("receiver should be alive");
        sleep(Duration::from_millis(20)).await;
    }
}

async fn consume(mut rx: mpsc::Receiver<u32>) -> u32 {
    let mut sum = 0;

    // TODO: Receive until channel closes and accumulate into `sum`.
    while let Some(v) = rx.recv().await {
        sum += v;
    }

    sum
}

#[tokio::test(flavor = "current_thread")]
async fn tokio3_pipeline() {
    let (tx, rx) = mpsc::channel(8);

    let producer = tokio::spawn(produce(tx));
    let consumer = tokio::spawn(consume(rx));

    producer.await.expect("producer should finish");
    let sum = consumer.await.expect("consumer should finish");

    assert_eq!(sum, 15);
}
