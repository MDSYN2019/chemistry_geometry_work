// Exercise 95: ffi-safe-norm
// Goal: compute an L2 norm from FFI input while validating numerical safety.
//
// STEP 1: Check null pointer / zero length cases.
// STEP 2: Iterate with `x*x` accumulation.
// STEP 3: Reject non-finite intermediate values.
// STEP 4: Return `sqrt(sum)` for valid input.

pub fn l2_norm_checked(ptr: *const f64, len: usize) -> Result<f64, &'static str> {
    if len == 0 {
        return Ok(0.0);
    }
    if ptr.is_null() {
        return Err("null pointer");
    }

    // SAFETY: pointer is non-null and caller guarantees `len` readable elements.
    let values = unsafe { std::slice::from_raw_parts(ptr, len) };

    let mut sum = 0.0;
    for &x in values {
        let term = x * x;
        if !term.is_finite() {
            return Err("non-finite value");
        }
        sum += term;
        if !sum.is_finite() {
            return Err("non-finite value");
        }
    }
    Ok(sum.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_norm() {
        let data = [3.0, 4.0];
        assert_eq!(l2_norm_checked(data.as_ptr(), data.len()), Ok(5.0));
    }

    #[test]
    fn rejects_non_finite_input() {
        let data = [f64::INFINITY, 1.0];
        assert_eq!(l2_norm_checked(data.as_ptr(), data.len()), Err("non-finite value"));
    }

    #[test]
    fn null_pointer_error_for_non_zero_len() {
        assert_eq!(l2_norm_checked(std::ptr::null(), 2), Err("null pointer"));
    }
}
