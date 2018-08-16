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
    let vecs_per_col = (n + m256_length - 1) / m256_length;

    // Pack d and its transpose into containers of vertical m256 vectors
    let mut vt = std::vec::Vec::with_capacity(n * vecs_per_col);
    let mut vd = std::vec::Vec::with_capacity(n * vecs_per_col);

    for row in 0..vecs_per_col {
        for col in 0..n {
            // Build 8 element arrays for vd and vt, with infinity padding
            let mut d_slice = [std::f32::INFINITY; m256_length];
            let mut t_slice = [std::f32::INFINITY; m256_length];
            for vec_j in 0..m256_length {
                let j = row * m256_length + vec_j;
                if j < n {
                    d_slice[vec_j] = d[n * j + col];
                    t_slice[vec_j] = d[n * col + j];
                }
            }
            // Convert arrays to 256-bit vectors and assign to vector containers
            vd.push(simd::from_slice(&d_slice));
            vt.push(simd::from_slice(&t_slice));
        }
    }

    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(m256_length * n).enumerate().for_each(|(i, row_block)| {
        for j in 0..vecs_per_col {
            // Intermediate results
            let mut tmp = [simd::m256_infty(); m256_length];

            // Horizontally compute 8 minimums from each pair of vertical vectors for this row block
            for col in 0..n {
                let vd_i = n * i + col;
                let vt_i = n * j + col;

                const PF: isize = 20;
                simd::prefetch(vd[vd_i..].as_ptr(), PF);
                simd::prefetch(vt[vt_i..].as_ptr(), PF);

                // Load vector pair
                let a0 = vd[vd_i];
                let b0 = vt[vt_i];
                // Compute permutations
                let a2 = simd::swap(a0, 2);
                let a4 = simd::swap(a0, 4);
                let a6 = simd::swap(a4, 2);
                let b1 = simd::swap(b0, 1);
                // Compute 8 independent, intermediate results by combining each permutation
                tmp[0] = simd::min(tmp[0], simd::add(a0, b0));
                tmp[1] = simd::min(tmp[1], simd::add(a0, b1));
                tmp[2] = simd::min(tmp[2], simd::add(a2, b0));
                tmp[3] = simd::min(tmp[3], simd::add(a2, b1));
                tmp[4] = simd::min(tmp[4], simd::add(a4, b0));
                tmp[5] = simd::min(tmp[5], simd::add(a4, b1));
                tmp[6] = simd::min(tmp[6], simd::add(a6, b0));
                tmp[7] = simd::min(tmp[7], simd::add(a6, b1));
            }
            // Swap all comparisons of b1 back for easier vector element extraction
            tmp[1] = simd::swap(tmp[1], 1);
            tmp[3] = simd::swap(tmp[3], 1);
            tmp[5] = simd::swap(tmp[5], 1);
            tmp[7] = simd::swap(tmp[7], 1);

            // Extract each element of each vector and assign to final result
            for jb in 0..m256_length {
                for ib in 0..m256_length {
                    let res_i = ib + i * m256_length;
                    let res_j = jb + j * m256_length;
                    if res_i < n && res_j < n {
                        let v = tmp[ib ^ jb];
                        let vi = jb as u8;
                        row_block[ib * n + res_j] = simd::extract(v, vi);
                    }
                }
            }
        }
    });
    #[cfg(feature = "no-multi-thread")]
    for i in 0..vecs_per_col {
        for j in 0..vecs_per_col {
            let mut tmp = [simd::m256_infty(); m256_length];
            for col in 0..n {
                let vd_i = n * i + col;
                let vt_i = n * j + col;

                const PF: isize = 20;
                simd::prefetch(vd[vd_i..].as_ptr(), PF);
                simd::prefetch(vt[vt_i..].as_ptr(), PF);

                let a0 = vd[vd_i];
                let b0 = vt[vt_i];
                let a2 = simd::swap(a0, 2);
                let a4 = simd::swap(a0, 4);
                let a6 = simd::swap(a4, 2);
                let b1 = simd::swap(b0, 1);
                tmp[0] = simd::min(tmp[0], simd::add(a0, b0));
                tmp[1] = simd::min(tmp[1], simd::add(a0, b1));
                tmp[2] = simd::min(tmp[2], simd::add(a2, b0));
                tmp[3] = simd::min(tmp[3], simd::add(a2, b1));
                tmp[4] = simd::min(tmp[4], simd::add(a4, b0));
                tmp[5] = simd::min(tmp[5], simd::add(a4, b1));
                tmp[6] = simd::min(tmp[6], simd::add(a6, b0));
                tmp[7] = simd::min(tmp[7], simd::add(a6, b1));
            }
            tmp[1] = simd::swap(tmp[1], 1);
            tmp[3] = simd::swap(tmp[3], 1);
            tmp[5] = simd::swap(tmp[5], 1);
            tmp[7] = simd::swap(tmp[7], 1);

            for jb in 0..m256_length {
                for ib in 0..m256_length {
                    let res_i = ib + i * m256_length;
                    let res_j = jb + j * m256_length;
                    if res_i < n && res_j < n {
                        let v = tmp[ib ^ jb];
                        let vi = jb as u8;
                        r[res_i * n + res_j] = simd::extract(v, vi);
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
