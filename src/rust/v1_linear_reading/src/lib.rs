use tools::create_extern_c_wrapper;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // Transpose of d
    let mut t = vec![0.0; n * n];
    // Function: assing a column j of d into row i of t
    let transpose_row = |(j, t_row): (usize, &mut [f32])| {
        for (i, x) in t_row.iter_mut().enumerate() {
            *x = d[n*i + j];
        }
    };
    // Assign rows of t in parallel from columns of d
    #[cfg(not(feature = "no-multi-thread"))]
    t.par_chunks_mut(n)
        .enumerate()
        .for_each(transpose_row);
    #[cfg(feature = "no-multi-thread")]
    t.chunks_mut(n)
        .enumerate()
        .for_each(transpose_row);
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
    // Partition r and d into slices, each containing a single row of r and d, and apply the function on the row pairs
    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(n)
        .zip(d.par_chunks(n))
        .for_each(step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n)
        .zip(d.chunks(n))
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
