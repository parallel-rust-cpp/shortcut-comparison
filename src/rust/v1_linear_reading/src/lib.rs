#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*; // Parallel chunks iterator


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // Transpose of d
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[n*j + i] = d[n*i + j];
        }
    }

    // For some row i in d, compute all results for a row in r
    let _step_row = |(i, row): (usize, &mut [f32])| {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = d[n*i + k];
                let y = t[n*j + k];
                let z = x + y;
                v = if z < v { z } else { v };
            }
            row[j] = v;
        }
    };

    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(n).enumerate().for_each(_step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n).enumerate().for_each(_step_row);
}


#[no_mangle]
pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: i32) {
    let d = std::slice::from_raw_parts(d_raw, (n * n) as usize);
    let mut r = std::slice::from_raw_parts_mut(r_raw, (n * n) as usize);
    _step(&mut r, d, n as usize);
}
