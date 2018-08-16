#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*; // Parallel chunks iterator

extern crate tools;
use tools::simd; // Custom SIMD helpers


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    #[allow(non_upper_case_globals)]
    const m256_length: usize = simd::M256_LENGTH;
    let vecs_per_row = (n + m256_length - 1) / m256_length;

    // Specify sizes of 3x3 blocks used to load 6 vectors into registers and computing 9 results
    #[allow(non_upper_case_globals)]
    const blocksize: usize = 3;
    let blocks_per_col = (n + blocksize - 1) / blocksize;
    let padded_height = blocksize * blocks_per_col;

    // Pack d and its transpose into m256 vectors
    let mut vt = std::vec::Vec::with_capacity(padded_height * vecs_per_row);
    let mut vd = std::vec::Vec::with_capacity(padded_height * vecs_per_row);

    for row in 0..n {
        for col in 0..vecs_per_row {
            // Build 8 element arrays for vd and vt, with infinity padding
            let mut d_slice = [std::f32::INFINITY; m256_length];
            let mut t_slice = [std::f32::INFINITY; m256_length];
            for vec_i in 0..m256_length {
                let i = col * m256_length + vec_i;
                if i < n {
                    d_slice[vec_i] = d[n * row + i];
                    t_slice[vec_i] = d[n * i + row];
                }
            }
            // Convert arrays to 256-bit vectors and assign to vector containers
            vd.push(simd::from_slice(&d_slice));
            vt.push(simd::from_slice(&t_slice));
        }
    }
    // Fill out of bounds rows, if any exist, with infinity
    let inf_slice = [std::f32::INFINITY; m256_length];
    for _ in n..padded_height {
        for _ in 0..vecs_per_row {
            vd.push(simd::from_slice(&inf_slice));
            vt.push(simd::from_slice(&inf_slice));
        }
    }

    #[cfg(not(feature = "no-multi-thread"))]
    // Partition the result slice into row blocks, each containing blocksize of rows
    // Then, compute the result of each row block in parallel
    r.par_chunks_mut(blocksize * n).enumerate().for_each(|(i, row_block)| {
        for j in 0..blocks_per_col {
            // m256 vector block containing blocksize ** 2 vectors
            let mut tmp = [[simd::m256_infty(); blocksize]; blocksize];
            // Horizontally compute minimums into one block on this row block containing blocksize of rows
            for col in 0..vecs_per_row {
                for block_i in 0..blocksize {
                    for block_j in 0..blocksize {
                        let x = vd[vecs_per_row * (i * blocksize + block_i) + col];
                        let y = vt[vecs_per_row * (j * blocksize + block_j) + col];
                        let z = simd::add(x, y);
                        tmp[block_i][block_j] = simd::min(tmp[block_i][block_j], z);
                    }
                }
            }
            // Reduce all vectors in block to minimums and assign final results
            for block_i in 0..blocksize {
                for block_j in 0..blocksize {
                    let res_i = i * blocksize + block_i;
                    let res_j = j * blocksize + block_j;
                    if res_i < n && res_j < n {
                        // Reduce one vector to final result
                        let res = simd::horizontal_min(tmp[block_i][block_j]);
                        // The row-index is from the perspective of the current row_block of r, not rows of r,
                        // hence block_i and not res_i
                        row_block[block_i * n + res_j] = res;
                    }
                }
            }
        }
    });
    #[cfg(feature = "no-multi-thread")]
    for i in 0..blocks_per_col {
        for j in 0..blocks_per_col {
            let mut tmp = [[simd::m256_infty(); blocksize]; blocksize];
            for col in 0..vecs_per_row {
                for block_i in 0..blocksize {
                    for block_j in 0..blocksize {
                        let x = vd[vecs_per_row * (i * blocksize + block_i) + col];
                        let y = vt[vecs_per_row * (j * blocksize + block_j) + col];
                        let z = simd::add(x, y);
                        tmp[block_i][block_j] = simd::min(tmp[block_i][block_j], z);
                    }
                }
            }
            for block_i in 0..blocksize {
                for block_j in 0..blocksize {
                    let res_i = i * blocksize + block_i;
                    let res_j = j * blocksize + block_j;
                    if res_i < n && res_j < n {
                        let res = simd::horizontal_min(tmp[block_i][block_j]);
                        r[res_i * n + res_j] = res;
                    }
                }
            }
        }
    }
}


#[no_mangle]
pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    let d = std::slice::from_raw_parts(d_raw, n * n);
    let mut r = std::slice::from_raw_parts_mut(r_raw, n * n);
    _step(&mut r, d, n);
}
