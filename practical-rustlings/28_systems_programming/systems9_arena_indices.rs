// Exercise 80: arena-indices
#[derive(Debug, Default)]
pub struct Arena<T> { items: Vec<T> }

impl<T> Arena<T> {
    pub fn insert(&mut self, value: T) -> usize {
        self.items.push(value);
        self.items.len() - 1
    }
    pub fn get(&self, idx: usize) -> Option<&T> { self.items.get(idx) }
}
