// Exercise 79: bounded-queue
use std::collections::VecDeque;

pub fn enqueue_bounded(queue: &mut VecDeque<i64>, cap: usize, value: i64) -> Result<(), &'static str> {
    if queue.len() >= cap { return Err("full"); }
    queue.push_back(value);
    Ok(())
}
