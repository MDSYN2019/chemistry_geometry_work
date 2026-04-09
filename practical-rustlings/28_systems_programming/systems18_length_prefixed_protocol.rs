// Exercise 89: length-prefixed-protocol
pub fn encode(payload: &[u8]) -> Option<Vec<u8>> {
    let len = u16::try_from(payload.len()).ok()?;
    let mut out = Vec::with_capacity(payload.len() + 2);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    Some(out)
}

pub fn decode(frame: &[u8]) -> Option<&[u8]> {
    if frame.len() < 2 { return None; }
    let len = u16::from_le_bytes([frame[0], frame[1]]) as usize;
    if frame.len() != len + 2 { return None; }
    Some(&frame[2..])
}
