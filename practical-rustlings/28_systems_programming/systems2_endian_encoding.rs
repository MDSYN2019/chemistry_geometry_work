// Exercise 73: endian-encoding
pub fn encode_u32_le(value: u32) -> [u8; 4] { value.to_le_bytes() }
pub fn decode_u32_le(bytes: [u8; 4]) -> u32 { u32::from_le_bytes(bytes) }
