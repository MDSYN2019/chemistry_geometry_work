// Implement a tiny read-mostly cache with `RwLock`.

use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Default)]
pub struct TinyCache {
    inner: RwLock<HashMap<String, i64>>,
}

impl TinyCache {
    pub fn insert(&self, key: &str, value: i64) {
        let mut w = self.inner.write().expect("rwlock poisoned");
        w.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<i64> {
        let r = self.inner.read().expect("rwlock poisoned");
        r.get(key).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_writes() {
        let cache = TinyCache::default();
        cache.insert("alpha", 7);
        assert_eq!(cache.get("alpha"), Some(7));
        assert_eq!(cache.get("beta"), None);
    }
}
