// OpenMP does not support Rust, but the Rayon library comes close with its parallel iterators
#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*; // Parallel chunks iterator


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    /// For some row i in d, compute all results for a row in r
    let _step_row = |(i, row): (usize, &mut [f32])| {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = d[n*i + k];
                let y = d[n*k + j];
                let z = x + y;
                v = if z < v { z } else { v };
            }
            row[j] = v;
        }
    };
    // Partition the result slice into n rows, and compute results for each row in separate threads
    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(n).enumerate().for_each(_step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n).enumerate().for_each(_step_row);
}


/// C interface that accepts raw C pointers as arguments
// Do not mangle function name to make library linking easier
#[no_mangle]
// Raw pointers can be dereferenced only inside 'unsafe' sections, hence function is marked as unsafe
pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: i32) {
    // Wrap raw pointers into 'not unsafe' Rust slices
    let d = std::slice::from_raw_parts(d_raw, (n * n) as usize);
    let mut r = std::slice::from_raw_parts_mut(r_raw, (n * n) as usize);
    // Evaluate Rust implementation of the step-function
    _step(&mut r, d, n as usize);
}
