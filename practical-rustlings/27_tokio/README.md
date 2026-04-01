# Tokio

Tokio is Rust's most widely used asynchronous runtime. It gives you an event loop,
lightweight tasks, async-aware synchronization primitives, and async channels.

These exercises focus on:

- spawning and awaiting async tasks,
- sharing mutable state across tasks with `tokio::sync::Mutex`,
- building a small async pipeline with `tokio::sync::mpsc`,
- racing futures with `tokio::select!`,
- retries/timeouts and coordinated shutdown patterns.

## Further information

- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [Tokio docs](https://docs.rs/tokio/latest/tokio/)
- [Rust Async Book](https://rust-lang.github.io/async-book/)
