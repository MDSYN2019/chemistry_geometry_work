// Exercise 91: ffi-scalar-call
// Goal: call a C ABI function pointer safely for a single scalar transform.
//
// STEP 1: Create a type alias for the callback signature.
// STEP 2: Reject `None` callbacks with a descriptive error.
// STEP 3: Call the function pointer and return the result.
// STEP 4: Add tests for both missing and present callback cases.

pub type ScalarFn = extern "C" fn(f64) -> f64;

pub fn call_scalar(callback: Option<ScalarFn>, x: f64) -> Result<f64, &'static str> {
    let f = callback.ok_or("callback missing")?;
    Ok(f(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn square(x: f64) -> f64 {
        x * x
    }

    #[test]
    fn missing_callback_is_error() {
        assert_eq!(call_scalar(None, 3.0), Err("callback missing"));
    }

    #[test]
    fn callback_executes() {
        assert_eq!(call_scalar(Some(square), 4.0), Ok(16.0));
    }
}
