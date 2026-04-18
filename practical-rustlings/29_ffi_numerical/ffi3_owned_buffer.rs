// Exercise 93: ffi-owned-buffer
// Goal: expose and reclaim Rust-owned buffers across an FFI boundary.
//
// STEP 1: Build a `#[repr(C)]` buffer descriptor.
// STEP 2: Allocate a Vec and leak it intentionally with `forget`.
// STEP 3: Reconstruct the Vec in a matching free function.
// STEP 4: Ensure null pointers and zero lengths are harmless.

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiBuffer {
    pub ptr: *mut f64,
    pub len: usize,
}

pub fn make_buffer(mut data: Vec<f64>) -> FfiBuffer {
    let out = FfiBuffer {
        ptr: data.as_mut_ptr(),
        len: data.len(),
    };
    std::mem::forget(data);
    out
}

pub unsafe fn free_buffer(buffer: FfiBuffer) {
    if buffer.ptr.is_null() || buffer.len == 0 {
        return;
    }
    // SAFETY: pointer/len pair must originate from `make_buffer` with identical element type.
    unsafe { drop(Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len)) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_buffer_contents() {
        let data = vec![2.0, 4.0, 6.0];
        let buffer = make_buffer(data);

        // SAFETY: produced by `make_buffer` above and read for exactly `len` elements.
        let view = unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) };
        assert_eq!(view, &[2.0, 4.0, 6.0]);

        // SAFETY: produced by `make_buffer` above; freeing exactly once.
        unsafe { free_buffer(buffer) };
    }

    #[test]
    fn free_buffer_ignores_empty() {
        // SAFETY: empty/null buffer is explicitly handled as a no-op.
        unsafe {
            free_buffer(FfiBuffer {
                ptr: std::ptr::null_mut(),
                len: 0,
            })
        };
    }
}
