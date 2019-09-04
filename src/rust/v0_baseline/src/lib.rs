use tools::create_extern_c_wrapper;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon; // Data-parallelism library with a work-stealing approach
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*; // Parallel chunks iterator


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // ANCHOR: step_row
    // Function: for some row i and every column j in d,
    // compute n results into r (r_row)
    let step_row = |(i, r_row): (usize, &mut [f32])| {
        for (j, res) in r_row.iter_mut().enumerate() {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = d[n*i + k];
                let y = d[n*k + j];
                let z = x + y;
                v = v.min(z);
            }
            *res = v;
        }
    };
    // ANCHOR_END: step_row
    // Partition r into slices, each containing a single row and apply the function on the rows
    // ANCHOR: chunks
    #[cfg(not(feature = "no-multi-thread"))] // Process each row as a separate task in parallel
    //// ANCHOR: par_chunks
    r.par_chunks_mut(n)
        .enumerate()
        .for_each(step_row);
    //// ANCHOR_END: par_chunks
    #[cfg(feature = "no-multi-thread")] // Process all rows in the main thread
    //// ANCHOR: seq_chunks
    //// ANCHOR: seq_chunks_mut
    r.chunks_mut(n)
    //// ANCHOR_END: seq_chunks_mut
        .enumerate()
        .for_each(step_row);
    //// ANCHOR_END: seq_chunks
    // ANCHOR_END: chunks
}

// ANCHOR: extern_macro_call
create_extern_c_wrapper!(step, _step);
// ANCHOR_END: extern_macro_call
