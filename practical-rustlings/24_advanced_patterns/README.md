# Advanced patterns (extra set)

These are 10 additional exercises focused on advanced Rust patterns. They are intentionally a bit more open-ended than early rustlings-style drills.

1. `advanced_patterns1_arc_mutex.rs` — shared mutable counter with `Arc<Mutex<_>>`
2. `advanced_patterns2_rwlock_cache.rs` — read-heavy cache with `RwLock`
3. `advanced_patterns3_mpsc_fan_in.rs` — multiple producers into one consumer
4. `advanced_patterns4_scoped_threads.rs` — borrowing into scoped threads
5. `advanced_patterns5_condvar_queue.rs` — blocking queue with `Condvar`
6. `advanced_patterns6_pin_basics.rs` — pinning and stable address checks
7. `advanced_patterns7_retry_backoff.rs` — retries with incremental backoff
8. `advanced_patterns8_cow_normalize.rs` — avoid allocations via `Cow<'_, str>`
9. `advanced_patterns9_ffi_guard.rs` — safe wrapper around nullable C string pointers
10. `advanced_patterns10_state_machine.rs` — typed state transition API

## Further reading

- [The Rust Book — Fearless Concurrency](https://doc.rust-lang.org/book/ch16-00-concurrency.html)
- [The Rustonomicon](https://doc.rust-lang.org/nomicon/)
- [std::pin](https://doc.rust-lang.org/std/pin/index.html)
