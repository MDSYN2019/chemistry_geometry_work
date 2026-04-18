// Exercise 94: ffi-opaque-context
// Goal: pass state through an opaque pointer safely.
//
// STEP 1: Define a context struct for numerical state.
// STEP 2: Provide constructor/destructor that transfer ownership via raw pointer.
// STEP 3: Add a mutation/read API with null checking.
// STEP 4: Test full lifecycle (create -> mutate -> read -> free).

#[derive(Debug)]
pub struct IntegratorContext {
    pub dt: f64,
    pub current_time: f64,
}

pub fn context_new(dt: f64) -> *mut IntegratorContext {
    Box::into_raw(Box::new(IntegratorContext {
        dt,
        current_time: 0.0,
    }))
}

pub unsafe fn context_free(ctx: *mut IntegratorContext) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: `ctx` must be allocated by `context_new` and freed exactly once.
    unsafe { drop(Box::from_raw(ctx)) };
}

pub unsafe fn context_tick(ctx: *mut IntegratorContext, steps: usize) -> Result<f64, &'static str> {
    if ctx.is_null() {
        return Err("null context");
    }
    // SAFETY: pointer is non-null and assumed valid for unique mutable access.
    let ctx = unsafe { &mut *ctx };
    ctx.current_time += ctx.dt * steps as f64;
    Ok(ctx.current_time)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_lifecycle() {
        let ptr = context_new(0.25);

        // SAFETY: pointer created by `context_new` above.
        let t1 = unsafe { context_tick(ptr, 4) }.unwrap();
        assert_eq!(t1, 1.0);

        // SAFETY: pointer created by `context_new` above.
        let t2 = unsafe { context_tick(ptr, 2) }.unwrap();
        assert_eq!(t2, 1.5);

        // SAFETY: free exactly once.
        unsafe { context_free(ptr) };
    }

    #[test]
    fn null_context_errors() {
        // SAFETY: function handles null as explicit error.
        let result = unsafe { context_tick(std::ptr::null_mut(), 1) };
        assert_eq!(result, Err("null context"));
    }
}
