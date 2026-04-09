// Exercise 75: bounded-io-slice
pub fn bounded_window(buf: &[u8], start: usize, len: usize) -> Option<&[u8]> {
    let end = start.checked_add(len)?;
    buf.get(start..end)
}
