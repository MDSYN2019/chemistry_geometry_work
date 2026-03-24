//#![forbid(unused_imports)]

/*
a mutex (mutual exclusion)  is a synchronization primitive that ensures that only one thread can access some data ata time

In rust, it is provided by std::sync::Mutex

---

counter += 1

load counter
add 1
store counter

---

If two threads do this simultaneously, you get a race condition

Thread A reads counter = 5
Thread B read counter = 6

Thread A writes 6
Thread B writes 6


*/
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug)]
struct Job {
    id: usize,
    description: String,
}

#[derive(Debug)]
struct JobQueue {
    jobs: Mutex<VecDeque<Job>>,
}

impl JobQueue {
    fn new() -> Self {
        Self {
            jobs: Mutex::new(VecDeque::new()), // gn
        }
    }
}
fn main() {
    let counter = Arc::new(Mutex::new(0)); // only one thread can access this data at a time
    let mut handles = Vec::new();

    for _ in 0..10 {
        let counter = Arc::clone(&counter);

        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap(); // get the pointer to the counter
            *num += 1; // increment the counter
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
