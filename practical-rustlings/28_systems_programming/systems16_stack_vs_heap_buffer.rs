// Exercise 87: stack-vs-heap-buffer
pub fn scratch_buffer(size: usize) -> Vec<u8> {
    vec![0; size]
}
