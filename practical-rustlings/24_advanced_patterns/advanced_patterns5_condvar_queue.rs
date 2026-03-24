// Implement a minimal one-item blocking queue using Mutex + Condvar.

use std::sync::{Condvar, Mutex};

#[derive(Default)]
pub struct OneSlot<T> {
    slot: Mutex<Option<T>>,
    cv: Condvar,
}

impl<T> OneSlot<T> {
    pub fn put(&self, value: T) {
        let mut guard = self.slot.lock().expect("mutex poisoned");
        while guard.is_some() {
            guard = self.cv.wait(guard).expect("condvar wait failed");
        }
        *guard = Some(value);
        self.cv.notify_one();
    }

    pub fn take(&self) -> T {
        let mut guard = self.slot.lock().expect("mutex poisoned");
        while guard.is_none() {
            guard = self.cv.wait(guard).expect("condvar wait failed");
        }
        let value = guard.take().expect("slot must contain value");
        self.cv.notify_one();
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_take() {
        let q = OneSlot::default();
        q.put(99);
        assert_eq!(q.take(), 99);
    }
}
