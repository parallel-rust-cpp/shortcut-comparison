use std::vec;

extern crate rayon;
use rayon::prelude::*; // par_chunks_mut

#[no_mangle]
pub extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    #[allow(non_upper_case_globals)]
    const block_size: usize = 4;
    let block_count = (n + block_size - 1) / block_size;
    let n_padded = block_count * block_size;

    let mut d = vec![std::f32::INFINITY; n * n_padded];
    let mut t = vec![std::f32::INFINITY; n * n_padded];

    for i in 0..n {
        for j in 0..n {
            d[n_padded*i + j] = unsafe { *d_raw.offset((n*i + j) as isize) };
            t[n_padded*i + j] = unsafe { *d_raw.offset((n*j + i) as isize) };
        }
    }

    // Parallelize by creating an empty, mutable result vector, which is partitioned into rows and iterated in parallel.

    let mut r: vec::Vec<f32> = vec![0.0; n * n];

    r.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut block = vec![std::f32::INFINITY; block_size];
            for b in 0..block_count {
                for k in 0..block_size {
                    let x = d[n_padded*i + b*block_size + k];
                    let y = t[n_padded*j + b*block_size + k];
                    let z = x + y;
                    block[k] = block[k].min(z);
                }
            }
            let mut res = std::f32::INFINITY;
            for x in block.iter() {
                res = res.min(*x);
            }
            row[j] = res;
        }
    });

    for (i, x) in r.iter().enumerate() {
        unsafe { *r_raw.offset(i as isize) = *x; }
    }
}
