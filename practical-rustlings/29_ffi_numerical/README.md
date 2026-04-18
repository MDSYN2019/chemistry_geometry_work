# FFI + Numerical Drills (Exercises 91–95)

These drills add a focused path for **Rust FFI fundamentals** and **numerical safety**.
Each exercise includes a step-by-step checklist directly in the file so you can implement it incrementally.

| Exercise | Slug | File |
|---|---|---|
| 91 | ffi-scalar-call | `ffi1_scalar_call.rs` |
| 92 | ffi-array-sum | `ffi2_array_sum.rs` |
| 93 | ffi-owned-buffer | `ffi3_owned_buffer.rs` |
| 94 | ffi-opaque-context | `ffi4_opaque_context.rs` |
| 95 | ffi-safe-norm | `ffi5_safe_norm.rs` |

## Suggested workflow
1. Read the top-of-file problem statement.
2. Complete **STEP 1** only, run tests, then continue.
3. Keep `unsafe` blocks as small as possible.
4. Add null / length checks before dereferencing raw pointers.
5. Document ownership rules for anything crossing the FFI boundary.
