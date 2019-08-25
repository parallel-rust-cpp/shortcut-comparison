use tools::create_extern_c_wrapper;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    const BLOCK_SIZE: usize = 4;
    let blocks_per_row = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let n_padded = blocks_per_row * BLOCK_SIZE;
    // Copy d and create its transpose, both padded with f32::INFINITY values to make the amount of columns divisible by blocks_per_row
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
    #[cfg(not(feature = "no-multi-thread"))]
    vd.par_chunks_mut(n_padded)
        .zip(vt.par_chunks_mut(n_padded))
        .enumerate()
        .for_each(preprocess_row);
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_mut(n_padded)
        .zip(vt.chunks_mut(n_padded))
        .enumerate()
        .for_each(preprocess_row);
    // Function: for some row i in vd (vd_row) and all rows j in vt (vt_rows),
    // compute all results for row i in r (r_row)
    let step_row = |(r_row, vd_row): (&mut [f32], &[f32])| {
        let vt_rows = vt.chunks(n_padded);
        // Length of a zipped iterator is the length of the shorter iterator in the zip pair so this never exceeds n
        for (res, vt_row) in r_row.iter_mut().zip(vt_rows) {
            // Accumulate results in blocks
            let vd_blocks = vd_row.chunks(BLOCK_SIZE);
            let vt_blocks = vt_row.chunks(BLOCK_SIZE);
            let mut block = [std::f32::INFINITY; BLOCK_SIZE];
            for (vd_block, vt_block) in vd_blocks.zip(vt_blocks) {
                assert_eq!(vd_block.len(), BLOCK_SIZE);
                assert_eq!(vt_block.len(), BLOCK_SIZE);
                for i in 0..BLOCK_SIZE {
                    let z = vd_block[i] + vt_block[i];
                    let b = block[i];
                    block[i] = if z < b { z } else { b };
                }
            }
            // Fold block values into a single minimum and assign to final result
            *res = block.iter().fold(std::f32::INFINITY, |acc, &x| if x < acc { x } else { acc });
        }
    };
    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(n)
        .zip(vd.par_chunks(n_padded))
        .for_each(step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n)
        .zip(vd.chunks(n_padded))
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
