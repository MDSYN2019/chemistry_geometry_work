// Combine values from multiple producer threads into one consumer.

use std::sync::mpsc;
use std::thread;

pub fn fan_in_sum(chunks: Vec<Vec<i32>>) -> i32 {
    let (tx, rx) = mpsc::channel();

    let mut handles = Vec::new();
    for chunk in chunks {
        let producer = tx.clone();
        handles.push(thread::spawn(move || {
            for n in chunk {
                producer.send(n).expect("receiver dropped");
            }
        }));
    }
    drop(tx);

    for h in handles {
        h.join().expect("thread panicked");
    }

    rx.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fan_in_works() {
        let total = fan_in_sum(vec![vec![1, 2], vec![3], vec![4, 5]]);
        assert_eq!(total, 15);
    }
}
