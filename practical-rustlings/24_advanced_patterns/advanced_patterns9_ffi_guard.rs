// Wrap a nullable C string pointer into a safe Rust API.

use std::ffi::CStr;
use std::os::raw::c_char;

pub fn c_string_len(ptr: *const c_char) -> Result<usize, &'static str> {
    if ptr.is_null() {
        return Err("null pointer");
    }

    let c_str = unsafe { CStr::from_ptr(ptr) };
    c_str.to_str().map(|s| s.len()).map_err(|_| "invalid utf8")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn handles_valid_and_null() {
        let s = CString::new("chem").unwrap();
        assert_eq!(c_string_len(s.as_ptr()), Ok(4));
        assert_eq!(c_string_len(std::ptr::null()), Err("null pointer"));
    }
}
