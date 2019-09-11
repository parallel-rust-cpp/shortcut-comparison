#![feature(core_intrinsics)]
extern crate core;
// For interleaving bits to construct Z-order curve
use core::arch::x86_64::_pdep_u32;

pub mod simd;
pub mod timer;

/// Extern C-ABI wrapper for moving data by raw pointers to a Rust 'step'-implementation
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

// ANCHOR: min
#[inline(always)]
pub fn min(x: f32, y: f32) -> f32 {
    if x < y { x } else { y }
}
// ANCHOR_END: min

// ANCHOR: z_encode
#[inline]
pub fn z_encode(x: u32, y: u32) -> u32 {
    let odd_bits = 0x55555555;
    unsafe { _pdep_u32(x, odd_bits) | (_pdep_u32(y, odd_bits) << 1) }
}
// ANCHOR_END: z_encode
