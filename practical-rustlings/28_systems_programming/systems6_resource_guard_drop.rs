// Exercise 77: resource-guard-drop
use std::sync::atomic::{AtomicBool, Ordering};

pub struct ResourceGuard<'a> { closed: &'a AtomicBool }

impl<'a> ResourceGuard<'a> {
    pub fn new(closed: &'a AtomicBool) -> Self { Self { closed } }
}

impl Drop for ResourceGuard<'_> {
    fn drop(&mut self) { self.closed.store(true, Ordering::SeqCst); }
}
