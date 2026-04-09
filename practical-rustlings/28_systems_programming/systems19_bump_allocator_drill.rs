// Exercise 90: bump-allocator-drill
#[derive(Debug)]
pub struct Bump { buf: Vec<u8>, next: usize }

impl Bump {
    pub fn new(capacity: usize) -> Self { Self { buf: vec![0; capacity], next: 0 } }

    pub fn alloc(&mut self, bytes: usize) -> Option<&mut [u8]> {
        let end = self.next.checked_add(bytes)?;
        if end > self.buf.len() { return None; }
        let start = self.next;
        self.next = end;
        Some(&mut self.buf[start..end])
    }
}
