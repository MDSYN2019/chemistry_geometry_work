// Exercise 82: rolling-checksum
pub fn xor_checksum(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0_u8, |acc, b| acc ^ b)
}
