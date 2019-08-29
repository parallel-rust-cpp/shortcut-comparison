use tools::create_extern_c_wrapper;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon; // Data-parallelism library with a work-stealing approach
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*; // Parallel chunks iterator


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // Function: for some row i and every column j in d, compute n results into r (r_row)
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
    r.chunks_exact_mut(n)
        .enumerate()
        .for_each(step_row);
}

create_extern_c_wrapper!(step, _step);
