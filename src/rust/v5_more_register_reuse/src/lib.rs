use tools::create_extern_c_wrapper;
use tools::simd;
use std::arch::x86_64::__m256;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    #[allow(non_upper_case_globals)]
    const vec_width: usize = simd::M256_LENGTH;
    let vecs_per_col = (n + vec_width - 1) / vec_width;

    // Like v4, but this time pack all elements of d into simd-vectors vertically,
    // i.e. the amount of rows will be divisible by vec_width
    let mut vd = std::vec![simd::m256_infty(); n * vecs_per_col];
    let mut vt = std::vec![simd::m256_infty(); n * vecs_per_col];
    let preprocess_row = |(row, (vd_row, vt_row)): (usize, (&mut [__m256], &mut [__m256]))| {
        for (col, (vd_elem, vt_elem)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            let mut d_slice = [std::f32::INFINITY; vec_width];
            let mut t_slice = [std::f32::INFINITY; vec_width];
            for vec_j in 0..vec_width {
                let j = row * vec_width + vec_j;
                if j < n {
                    d_slice[vec_j] = d[n * j + col];
                    t_slice[vec_j] = d[n * col + j];
                }
            }
            *vd_elem = simd::from_slice(&d_slice);
            *vt_elem = simd::from_slice(&t_slice);
        }
    };
    #[cfg(not(feature = "no-multi-thread"))]
    vd.par_chunks_mut(n)
        .zip(vt.par_chunks_mut(n))
        .enumerate()
        .for_each(preprocess_row);
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_mut(n)
        .zip(vt.chunks_mut(n))
        .enumerate()
        .for_each(preprocess_row);

    // Function: for some row i in d, compute all results for a row block in r,
    // where the block contains rows equal to the length of a 256-bit f32 simd-vector
    let step_row = |(i, (r_row_block, vd_row)): (usize, (&mut [f32], &[__m256]))| {
        assert_eq!(vd_row.len(), n);
        // Compute results for all combinations of simd-vector rows of vt and vd
        for (j, vt_row) in vt.chunks(n).enumerate() {
            assert_eq!(vt_row.len(), n);
            // Intermediate results for vec_width of rows
            let mut tmp = [simd::m256_infty(); vec_width];
            // Horizontally compute 8 minimums from each pair of vertical vectors for this row block
            for (&d0, &t0) in vd_row.iter().zip(vt_row) {
                // Compute permutations of simd-vector elements
                let d2 = simd::swap(d0, 2);
                let d4 = simd::swap(d0, 4);
                let d6 = simd::swap(d4, 2);
                let t1 = simd::swap(t0, 1);
                // Compute 8 independent, intermediate results for 8 rows by combining each permutation
                tmp[0] = simd::min(tmp[0], simd::add(d0, t0));
                tmp[1] = simd::min(tmp[1], simd::add(d0, t1));
                tmp[2] = simd::min(tmp[2], simd::add(d2, t0));
                tmp[3] = simd::min(tmp[3], simd::add(d2, t1));
                tmp[4] = simd::min(tmp[4], simd::add(d4, t0));
                tmp[5] = simd::min(tmp[5], simd::add(d4, t1));
                tmp[6] = simd::min(tmp[6], simd::add(d6, t0));
                tmp[7] = simd::min(tmp[7], simd::add(d6, t1));
            }
            // Swap all comparisons with t1 back so that we can extract the results with a straightforward XOR pattern
            tmp[1] = simd::swap(tmp[1], 1);
            tmp[3] = simd::swap(tmp[3], 1);
            tmp[5] = simd::swap(tmp[5], 1);
            tmp[7] = simd::swap(tmp[7], 1);
            // Set 8 final results
            for block_i in 0..vec_width {
                for (block_j, r_row) in r_row_block.chunks_mut(n).enumerate() {
                    assert_eq!(r_row.len(), n);
                    let res_i = block_j + i * vec_width;
                    let res_j = block_i + j * vec_width;
                    if res_i < n && res_j < n {
                        let v = tmp[block_j ^ block_i];
                        let vi = block_i as u8;
                        r_row[res_j] = simd::extract(v, vi);
                    }
                }
            }
        }
    };
    // Chunk up r into row blocks containing vec_width of rows and vd into rows each containing n simd-vectors
    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(vec_width * n)
        .zip(vd.par_chunks(n))
        .enumerate()
        .for_each(step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(vec_width * n)
        .zip(vd.chunks(n))
        .enumerate()
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
