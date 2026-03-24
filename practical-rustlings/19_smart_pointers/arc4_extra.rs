/*
When shared ownership between threads is needed, arc can be used. This struct,
implementation can create a pointer for the location of a value in the memory
while increasing the reference counter.

 -> The mutex is there to make sure the shared string is not mutated concurrently
by multuple threads at once - it is actually the mechanism that is required that allows safe mutable access, NOT immutability.



*/

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    let mut string_vector = vec!["Orange".to_string()];
    // Only one thread can access this data one at a time.
    let apple = Arc::new(Mutex::new(String::from("The new apple "))); // the value inside is a immutable string slice?
    let mut handles = Vec::new();

    for i in 0..100 {
        let counter = Arc::clone(&apple); // pointer to the reference in memory,
                                          // and also increases the counter
        let mut new_string = i.to_string();

        let handle = thread::spawn(move || {
            let mut thread_num = counter.lock().unwrap(); // get the pointer to the counter
                                                          // push_str already mutates thr string and returns ().

            thread_num.push_str(&new_string);
            thread_num.push(' ');
        });

        handles.push(handle);
    }

    /*
    thread::spawn(...) returns a JoinHandle

    That handle lets you:

    - wait for a thread to finish with join()
    - detect if panicked
    - sometimes collect a return value
     */

    // Need to join the spawned threads
    for handle in handles {
        handle.join().unwrap();
    }

    println!("{}", apple.lock().unwrap());
}
