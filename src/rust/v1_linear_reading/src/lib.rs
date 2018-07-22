// OpenMP does not support Rust, but the Rayon library comes close with its parallel iterators
extern crate rayon;
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

    // Partition the result slice into n rows, and compute result for each row in parallel
    r.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = d[n*i + k];
                let y = t[n*j + k];
                let z = x + y;
                v = v.min(z);
            }
            row[j] = v;
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    let d = std::slice::from_raw_parts(d_raw, n * n);
    let mut r = std::slice::from_raw_parts_mut(r_raw, n * n);
    _step(&mut r, d, n);
}
