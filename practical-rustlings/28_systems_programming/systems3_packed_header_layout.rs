// Exercise 74: packed-header-layout
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    pub magic: u16,
    pub version: u8,
    pub flags: u8,
    pub payload_len: u32,
}

impl PacketHeader {
    pub fn to_bytes(self) -> [u8; 8] {
        let mut out = [0; 8];
        out[0..2].copy_from_slice(&self.magic.to_le_bytes());
        out[2] = self.version;
        out[3] = self.flags;
        out[4..8].copy_from_slice(&self.payload_len.to_le_bytes());
        out
    }
}
