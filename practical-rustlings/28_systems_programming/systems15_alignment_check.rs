// Exercise 86: alignment-check
pub fn is_aligned_for<T>(ptr: *const u8) -> bool {
    (ptr as usize).is_multiple_of(std::mem::align_of::<T>())
}
