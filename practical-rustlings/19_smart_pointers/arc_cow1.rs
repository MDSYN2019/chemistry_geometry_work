/*
Arc lets multiple threads share the same data cheaply

Cow lets each worker borrow the shared string if no change is needed, or create
an owned modified version only when necessary


*/

use std::borrow::Cow;
use std::sync::Arc;
use std::thread;

fn normalize(input: &str) -> Cow<'_, str> {
    if input.contains(" ") {
        Cow::Owned(input.replace(" ", "_"))
    } else {
        Cow::Borrowed(input)
    }
}

fn main() {
    let shared: Arc<str> = Arc::from("hello world");
    let mut handles = Vec::new();

    for i in 0..3 {
        let shared_clone = Arc::clone(&shared);

        // operate within each spawned thread
        let handle = thread::spawn(move || {
            let normalized = normalize(&shared_clone);

            match &normalized {
                Cow::Borrowed(s) => {
                    println!("thread {i}: borrowed -> {s}");
                }

                Cow::Owned(s) => {
                    println!("thread {i}: owned -> {s}");
                }
            }
            // convert into owned data
            normalized.into_owned();
        });

        handles.push(handle);
    }

    /*
    join() waits for that thread to finish

    then it returns whatever value the thread closure returned
     */
    for handle in handles {
        let result = handle.join().unwrap();
        println!("final result: {:?}", result);
    }
}
