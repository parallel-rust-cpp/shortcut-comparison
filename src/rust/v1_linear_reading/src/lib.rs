use tools::create_extern_c_wrapper;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // ANCHOR: transpose
    // Transpose of d
    let mut t = vec![0.0; n * n];
    // Function: for some column j in d, copy all elements of that column into row i in t (t_row)
    let transpose_row = |(j, t_row): (usize, &mut [f32])| {
        for (i, x) in t_row.iter_mut().enumerate() {
            *x = d[n*i + j];
        }
    };
    // Copy all columns of d into rows of t in parallel
    // ANCHOR_END: transpose
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: transpose_apply
    t.par_chunks_mut(n)
        .enumerate()
        .for_each(transpose_row);
    // ANCHOR_END: transpose_apply
    #[cfg(feature = "no-multi-thread")]
    t.chunks_mut(n)
        .enumerate()
        .for_each(transpose_row);
    // ANCHOR: step_row
    // Function: for some row i in d (d_row) and all rows t (t_rows), compute n results into a row in r (r_row)
    let step_row = |(r_row, d_row): (&mut [f32], &[f32])| {
        // t is immutable, so we can share it in concurrent invocations of this function
        let t_rows = t.chunks_exact(n);
        for (res, t_row) in r_row.iter_mut().zip(t_rows) {
            let mut v = std::f32::INFINITY;
            for (&x, &y) in d_row.iter().zip(t_row) {
                let z = x + y;
                v = if z < v { z } else { v };
            }
            *res = v;
        }
    };
    // Partition r and d into slices, each containing a single row of r and d,
    // and apply the function on the row pairs
    // ANCHOR_END: step_row
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: step_row_apply
    r.par_chunks_mut(n)
        .zip(d.par_chunks(n))
        .for_each(step_row);
    // ANCHOR_END: step_row_apply
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n)
        .zip(d.chunks(n))
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
