use tools::create_extern_c_wrapper;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // ANCHOR: preprocess
    const BLOCK_SIZE: usize = 4;
    let blocks_per_row = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let n_padded = blocks_per_row * BLOCK_SIZE;
    // d and transpose of d with extra room at the end of each row, both initially filled with f32::INFINITY
    let mut vd = vec![std::f32::INFINITY; n_padded * n];
    let mut vt = vec![std::f32::INFINITY; n_padded * n];
    // Function: for one row of vd and vt, copy a row at 'row' of d into vd and column at 'row' of d into vt
    let preprocess_row = |(row, (vd_row, vt_row)): (usize, (&mut [f32], &mut [f32]))| {
        for (col, (x, y)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            if row < n && col < n {
                *x = d[n*row + col];
                *y = d[n*col + row];
            }
        }
    };
    // Partition vd and vt into rows, apply preprocessing in parallel for each row pair
    // ANCHOR_END: preprocess
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: preprocess_apply
    vd.par_chunks_mut(n_padded)
        .zip(vt.par_chunks_mut(n_padded))
        .enumerate()
        .for_each(preprocess_row);
    // ANCHOR_END: preprocess_apply
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_mut(n_padded)
        .zip(vt.chunks_mut(n_padded))
        .enumerate()
        .for_each(preprocess_row);
    // ANCHOR: step_row
    // Function: for some row in vd (vd_row) and all rows in vt (vt_rows),
    // compute all results for a row in r (r_row), corresponding to the row index of vd_row.
    let step_row = |(r_row, vd_row): (&mut [f32], &[f32])| {
        let vt_rows = vt.chunks_exact(n_padded);
        // Length of a zipped iterator is the length of the shorter iterator in the zip pair so this never exceeds n
        for (res, vt_row) in r_row.iter_mut().zip(vt_rows) {
            // Partition both rows into chunks of size 4
            // (x0, x1, x2, x3), (x4, x5, x6, x7), ...
            let vd_blocks = vd_row.chunks_exact(BLOCK_SIZE);
            // (y0, y1, y2, y3), (y4, y5, y6, y7), ...
            let vt_blocks = vt_row.chunks_exact(BLOCK_SIZE);
            // Using an array here is bit more convenient than 4 different variables, e.g. v0, v1, v2, v3
            let mut block = [std::f32::INFINITY; BLOCK_SIZE];
            // Accumulate all results as in v1, but 4 elements at a time
            for (vd_block, vt_block) in vd_blocks.zip(vt_blocks) {
                assert_eq!(vd_block.len(), BLOCK_SIZE);
                assert_eq!(vt_block.len(), BLOCK_SIZE);
                for i in 0..BLOCK_SIZE {
                    let z = vd_block[i] + vt_block[i];
                    let b = block[i];
                    block[i] = if z < b { z } else { b };
                }
            }
            // Fold 4 intermediate values into a single minimum and assign to final result
            *res = block.iter().fold(std::f32::INFINITY, |acc, &x| if x < acc { x } else { acc });
        }
    };
    // ANCHOR_END: step_row
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: step_row_apply
    r.par_chunks_mut(n)
        .zip(vd.par_chunks(n_padded))
        .for_each(step_row);
    // ANCHOR_END: step_row_apply
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n)
        .zip(vd.chunks(n_padded))
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
