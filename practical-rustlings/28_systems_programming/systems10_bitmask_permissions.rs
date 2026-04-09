// Exercise 81: bitmask-permissions
pub const READ: u8 = 0b001;
pub const WRITE: u8 = 0b010;
pub const EXEC: u8 = 0b100;

pub fn has(mask: u8, perm: u8) -> bool { mask & perm == perm }
