// Exercise 92: ffi-array-sum
// Goal: safely read an FFI-provided array and compute a sum.
//
// STEP 1: Validate null pointers for non-zero lengths.
// STEP 2: Convert raw parts into a borrowed slice.
// STEP 3: Sum values without allocating.
// STEP 4: Cover length 0 and null-pointer error paths in tests.

pub fn sum_from_ptr(ptr: *const f64, len: usize) -> Result<f64, &'static str> {
    if len == 0 {
        return Ok(0.0);
    }
    if ptr.is_null() {
        return Err("null pointer");
    }

    // SAFETY: `ptr` is non-null and caller promises at least `len` readable f64 values.
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(slice.iter().copied().sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_valid_data() {
        let data = [1.5, 2.0, 3.5];
        let sum = sum_from_ptr(data.as_ptr(), data.len());
        assert_eq!(sum, Ok(7.0));
    }

    #[test]
    fn zero_len_is_ok_even_for_null() {
        assert_eq!(sum_from_ptr(std::ptr::null(), 0), Ok(0.0));
    }

    #[test]
    fn non_zero_len_null_is_error() {
        assert_eq!(sum_from_ptr(std::ptr::null(), 3), Err("null pointer"));
    }
}
