// Exercise 72: ring-buffer-bytes
// TODO: implement a fixed-capacity byte ring buffer with push/pop.

#[derive(Debug)]
pub struct ByteRing {
    buf: Vec<u8>,
    head: usize,
    tail: usize,
    len: usize,
}

impl ByteRing {
    pub fn with_capacity(capacity: usize) -> Self {
        Self { buf: vec![0; capacity.max(1)], head: 0, tail: 0, len: 0 }
    }

    pub fn push(&mut self, byte: u8) -> Result<(), &'static str> {
        if self.len == self.buf.len() { return Err("full"); }
        self.buf[self.tail] = byte;
        self.tail = (self.tail + 1) % self.buf.len();
        self.len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<u8> {
        if self.len == 0 { return None; }
        let out = self.buf[self.head];
        self.head = (self.head + 1) % self.buf.len();
        self.len -= 1;
        Some(out)
    }
}
