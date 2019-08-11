#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon; // Data-parallelism library with a work-stealing approach
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*; // Parallel chunks iterator


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // Function: for some row i and column j in d, compute all results for a row i in r (r_row)
    let step_row = |(i, r_row): (usize, &mut [f32])| {
        for (j, res) in r_row.iter_mut().enumerate() {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = d[n*i + k];
                let y = d[n*k + j];
                let z = x + y;
                v = if z < v { z } else { v };
            }
            *res = v;
        }
    };
    // Partition r into slices, each containing a single row and apply the function on the rows
    #[cfg(not(feature = "no-multi-thread"))] // Process each row as a separate task in parallel
    r.par_chunks_mut(n)
        .enumerate()
        .for_each(step_row);
    #[cfg(feature = "no-multi-thread")] // Process all rows in the main thread
    r.chunks_mut(n)
        .enumerate()
        .for_each(step_row);
}


/// C interface that accepts raw C pointers as arguments
// Do not mangle function name to make library linking easier
#[no_mangle]
// Raw pointers can be dereferenced only inside 'unsafe' sections, hence function is marked as unsafe
pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: i32) {
    // Catch any unwinding panics so that they won't propagate over the ABI to the calling program, which would be undefined behaviour
    let result = std::panic::catch_unwind(|| {
        // Wrap raw pointers into 'not unsafe' Rust slices with a well defined size
        let d = std::slice::from_raw_parts(d_raw, (n * n) as usize);
        let mut r = std::slice::from_raw_parts_mut(r_raw, (n * n) as usize);
        // Evaluate the Rust implementation of the step-function
        _step(&mut r, d, n as usize);
    });
    // Print an error to stderr if something went horribly wrong
    if result.is_err() {
        eprintln!("error: rust panicked");
    }
}
