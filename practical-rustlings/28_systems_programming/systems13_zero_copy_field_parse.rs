// Exercise 84: zero-copy-field-parse
pub fn parse_frame(frame: &[u8]) -> Option<(&[u8], &[u8])> {
    if frame.len() < 2 { return None; }
    let header_len = frame[0] as usize;
    let payload_len = frame[1] as usize;
    if frame.len() < 2 + header_len + payload_len { return None; }
    let header = &frame[2..2 + header_len];
    let payload = &frame[2 + header_len..2 + header_len + payload_len];
    Some((header, payload))
}
