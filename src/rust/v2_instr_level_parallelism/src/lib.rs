#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*; // Parallel chunks iterator


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    const BLOCK_SIZE: usize = 2;
    let blocks_per_row = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let n_padded = blocks_per_row * BLOCK_SIZE;

    // Create d with right padding, and transpose of d
    let mut vt = vec![std::f32::INFINITY; n_padded * n];
    let mut vd = vec![std::f32::INFINITY; n_padded * n];
    for i in 0..n {
        for j in 0..n {
            vd[n_padded*i + j] = d[n*i + j];
            vt[n_padded*i + j] = d[n*j + i];
        }
    }

    // For some row i in d, compute all results for a row in r
    let _step_row = |(i, row): (usize, &mut [f32])| {
        for j in 0..n {
            let mut block = [std::f32::INFINITY; BLOCK_SIZE];
            for b in 0..blocks_per_row {
                for k in 0..BLOCK_SIZE {
                    let x = vd[n_padded * i + b * BLOCK_SIZE + k];
                    let y = vt[n_padded * j + b * BLOCK_SIZE + k];
                    let z = x + y;
                    block[k] = if z < block[k] { z } else { block[k] };
                }
            }
            // Fold block values into a single minimum and assign to final result
            row[j] = block.iter().fold(block[0], |acc, &x| if x < acc { x } else { acc });
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
