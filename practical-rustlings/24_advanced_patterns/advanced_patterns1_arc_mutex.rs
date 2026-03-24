// Build a thread-safe counter shared across threads.
//
// TODOs:
// 1) Implement `parallel_increment` so each thread increments the same counter.
// 2) Return the final count.

use std::sync::{Arc, Mutex};
use std::thread;

pub fn parallel_increment(num_threads: usize, increments_per_thread: usize) -> usize {
    let counter = Arc::new(Mutex::new(0usize));

    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let shared = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..increments_per_thread {
                let mut guard = shared.lock().expect("mutex poisoned");
                *guard += 1;
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    *counter.lock().expect("mutex poisoned")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_all_increments() {
        assert_eq!(parallel_increment(4, 10), 40);
    }
}
