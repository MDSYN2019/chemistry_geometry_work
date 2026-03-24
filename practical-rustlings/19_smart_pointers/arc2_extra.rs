#![forbid(unused_imports)]
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;
/*

Arc<T> gives shared ownership across threads
Arc - atomically reference counted pointer

If you need mutation too, combine it with a lock Mutex<T> or RwLock<T>
 */

#[derive(Debug, Clone)]
struct ProteinScore {
    binding_score: f64,
    pocket_volume: f64,
}

#[derive(Debug)]
struct SharedCache {
    // RwLock allows many readers or one writer
    inner: RwLock<HashMap<String, ProteinScore>>, // this data is shared between threads, and can be read by many threads
}

/*

We are creating a sharedcache struct, which has a hashmap entry. This is an RWlock, type for the HashMap type, which allows generated threads (all of them) to read the contest of the shared pointer, and one thread to be able to write into the Arc

*/

impl SharedCache {
    fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    // insert string and score
    fn insert(&self, protein_id: String, score: ProteinScore) {
        let mut map = self.inner.write().expect("read lock poisoned"); // 
        map.insert(protein_id, score);
    }

    // read the data - clone the dataset
    fn get(&self, protein_id: &str) -> Option<ProteinScore> {
        let map = self.inner.read().expect("read lock poisoned");
        map.get(protein_id).cloned() // get the shared hashmap and get the protein id 
    }

    fn len(&self) -> usize {
        let map = self.inner.read().expect("read lock poisoned");
        map.len() // get the length of the hashmap
    }
}

fn main() {
    let cache = Arc::new(SharedCache::new()); // creates a SharedCache struct 
    // into an arc, which is shared ownership across threads
    cache.insert(
        "P123435".to_string(),
        ProteinScore {
            binding_score: -8.7,
            pocket_volume: 412.3,
        },
    );

    let mut handles = Vec::new();

    // spawning several readers
    for worker_id in 0..3 {
        let cache_clone = Arc::clone(&cache); // increment the reference count
        // all threads shares the same underlying object

        let handle = thread::spawn(move || {
            for _ in 0..3 {
                let result = cache_clone.get("P12345");
                println!("reader {worker_id}: {:?}", result);
            }
        });

        handles.push(handle);
    }

    // spawn one reader
    {
        let cache_clone = Arc::clone(&cache); // we increment the arc 

        let handle = thread::spawn(move || {
            // spawn the thread ( just one )
            // loop over indices and generate the data we wish to write for this
            // one spawned thread
            for i in 0..3 {
                let protein_id = format!("NEW_{i}");
                let score = ProteinScore {
                    binding_score: -6.0 - i as f64,
                    pocket_volume: 300.0 + 10.0 * i as f64,
                };
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    println!("final cache size = {}", cache.len());
}
