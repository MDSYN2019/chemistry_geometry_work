/*

Multi-threaded programming

 -> running multiple sequences of instructions at the same time
inside one program


for example:

single-threaded:
---------------
task A -> Task B -> Task C

multi-threaded:
--------------
thread 1: Task A
thread 2: Task B
thread 3: Task C

If you have mulitple CPU cores, they can truly run simultaneously


----

Why do we use multi-threading?

 -> If you have 8 cores:

You can split work across 8 threads

Example:

matrix multiplication
Monte Carlo simulations
molecular dynamics force calculations

In servers:

Thread A handles request 1
Thread B handles request 2
Thread C writes logs

No request blocks the whole ecosystem

---

Waiting tasks
-------------





*/
#![forbid(unused_imports)]
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    // What is mutex here?
    /*

    mutex = mutual exclusion - only one thread can access the protected
    data at a time
    */

    let counter = Arc::new(Mutex::new(0)); // multiple threads can own it,
    // but only on thread can modify it at a time

    let mut handles = vec![];

    for _ in 0..5 {
        // clone the counter so that we can have shared ownership
        let counter = Arc::clone(&counter);

        let handle = thread::spawn(move || {
            // tries to acquire the lock
            // if another thread holds it - wait
            // when lock is acquired - returns MutexGuard
            // when mutexguard drops, lock automatically released
            let mut num = counter.lock().unwrap();
            *num += 1;
        });

        handles.push(handle);
    }
    let mut results = vec![];
    //
    for h in handles {
        match h.join() {
            Ok(v) => {
                println!("Thread finished");
                results.push(v);
            }
            Err(_) => println!("Thread panicked"),
        }
        //let result = handles.join().unwrap();
        println!("{:?}", results);
    }
}
