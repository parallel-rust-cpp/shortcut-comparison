#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // Transpose of d
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[n*j + i] = d[n*i + j];
        }
    }
    // Function: for some row i in d (d_row) and all rows j in t (t_rows), compute all results for row i in r (r_row)
    let step_row = |(r_row, d_row): (&mut [f32], &[f32])| {
        let t_rows = t.chunks(n);
        for (res, t_row) in r_row.iter_mut().zip(t_rows) {
            let mut v = std::f32::INFINITY;
            for (&x, &y) in d_row.iter().zip(t_row) {
                let z = x + y;
                v = if z < v { z } else { v };
            }
            *res = v;
        }
    };
    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(n).zip(d.par_chunks(n)).for_each(step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n).zip(d.chunks(n)).for_each(step_row);
}


#[no_mangle]
pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: i32) {
    let result = std::panic::catch_unwind(|| {
        let d = std::slice::from_raw_parts(d_raw, (n * n) as usize);
        let mut r = std::slice::from_raw_parts_mut(r_raw, (n * n) as usize);
        _step(&mut r, d, n as usize);
    });
    if result.is_err() {
        eprintln!("error: rust panicked");
    }
}
