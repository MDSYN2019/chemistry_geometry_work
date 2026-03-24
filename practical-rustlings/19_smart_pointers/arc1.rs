// In this exercise, we are given a `Vec` of `u32` called `numbers` with values
// ranging from 0 to 99. We would like to use this set of numbers within 8
// different threads simultaneously. Each thread is going to get the sum of
// every eighth value with an offset.
//
// The first thread (offset 0), will sum 0, 8, 16, …
// The second thread (offset 1), will sum 1, 9, 17, …
// The third thread (offset 2), will sum 2, 10, 18, …
// …
// The eighth thread (offset 7), will sum 7, 15, 23, …
//
// Each thread should own a reference-counting pointer to the vector of
// numbers. But `Rc` isn't thread-safe. Therefore, we need to use `Arc`.
//
// Don't get distracted by how threads are spawned and joined. We will practice
// that later in the exercises about threads.

// Don't change the lines below.

#![forbid(unused_imports)]
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    /*
    Personal code to understand what is happening with Arc

    When shared ownership between threads is needed, Arc (atomically reference counted) can be used.
    This struct, via the Clone implementation can create a reference pointer for the location
    of a value in memory heap while increasing the reference counter.

    for example:

    let a = String::from("hello");
    let b = a;

    - only one owner at a time

    But sometimes, you want:

             -> Many threads reading the same config
             -> Many handlers accessing the same DB pool
             -> shared graph structure
             -> Shared model state

     */

    let apple = Arc::new("the same apple");

    for _ in 0..10 {
        // Here, there is no value specified
        let apple = Arc::clone(&apple);

        thread::spawn(move || {
            println!("{:?}", apple);
        });
    }

    let numbers: Vec<_> = (0..100u32).collect();

    // TODO: Define `shared_numbers` by using `Arc`.
    // let shared_numbers = ???;

    let mut join_handles = Vec::new();

    for offset in 0..8 {
        // TODO: Define `child_numbers` using `shared_numbers`.
        // let child_numbers = ???;
        let child_numbers: Vec<u32> = (offset..=99).step_by(8 as usize).collect();
        // generate separate threads here
        let handle = thread::spawn(move || {
            let sum: u32 = child_numbers.iter().filter(|&&n| n % 8 == offset).sum();
            println!("Sum of offset {offset} is {sum}");
        });

        join_handles.push(handle);
    }

    for handle in join_handles.into_iter() {
        handle.join().unwrap();
    }
}
