#![feature(core_intrinsics)]
pub mod simd;
pub mod timer;

/// Macro for wrapping a Rust function inside an extern C 'step'-function
#[macro_export]
macro_rules! create_extern_c_wrapper {
    ($extern_func:ident, $wrapped_func:ident) => {
        #[no_mangle]
        pub extern "C" fn $extern_func(r_raw: *mut f32, d_raw: *const f32, n: i32) {
            // Catch any unwinding panics so that they won't propagate over the ABI to the calling program, which would be undefined behaviour
            let result = std::panic::catch_unwind(|| {
                // Wrap raw pointers into 'not unsafe' Rust slices with a well defined size
                let d = unsafe { std::slice::from_raw_parts(d_raw, (n * n) as usize) };
                let mut r = unsafe { std::slice::from_raw_parts_mut(r_raw, (n * n) as usize) };
                // Evaluate the wrapped function
                $wrapped_func(&mut r, d, n as usize);
            });
            // Print an error to stderr if something went horribly wrong
            if result.is_err() {
                eprintln!("error: rust panicked");
            }
        }
    };
}
