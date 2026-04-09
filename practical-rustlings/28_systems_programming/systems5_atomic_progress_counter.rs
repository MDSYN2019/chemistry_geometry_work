// Exercise 76: atomic-progress-counter
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn bump(counter: &AtomicUsize, n: usize) {
    counter.fetch_add(n, Ordering::Relaxed);
}
