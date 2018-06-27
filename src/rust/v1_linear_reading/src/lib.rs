use std::vec;

extern crate rayon;
use rayon::prelude::*; // par_iter

#[no_mangle]
pub extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    // All data will be moved to intermediate containers because Rust does not
    // allow references to const raw pointers in parallel sections
    let mut t: vec::Vec<f32> = vec![0.0; n * n];
    let mut d: vec::Vec<f32> = vec![0.0; n * n];
    let mut r: vec::Vec<f32> = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..n {
            let x = unsafe { *d_raw.offset((n*i + j) as isize) };
            d[n*i + j] = x;
            t[n*j + i] = x;
        }
    }

    // Run each iteration of outer for loop in parallel
    (0..n).into_par_iter().for_each(|i| {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = d[n*i + k];
                let y = t[n*j + k];
                let z = x + y;
                v = v.min(z);
            }
            r[n*i + j] = v;
        }
    });

    for (i, x) in r.iter().enumerate() {
        unsafe {
            *r_raw.offset(i as isize) = *x;
        }
    }
}
