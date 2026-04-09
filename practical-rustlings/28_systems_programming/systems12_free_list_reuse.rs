// Exercise 83: free-list-reuse
#[derive(Default)]
pub struct SlotMap<T> { slots: Vec<Option<T>> }

impl<T> SlotMap<T> {
    pub fn insert(&mut self, value: T) -> usize {
        if let Some((i, slot)) = self.slots.iter_mut().enumerate().find(|(_, s)| s.is_none()) {
            *slot = Some(value);
            i
        } else {
            self.slots.push(Some(value));
            self.slots.len() - 1
        }
    }
    pub fn remove(&mut self, idx: usize) -> Option<T> { self.slots.get_mut(idx)?.take() }
}
