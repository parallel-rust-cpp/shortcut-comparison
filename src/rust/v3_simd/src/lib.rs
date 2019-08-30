use tools::create_extern_c_wrapper;
// Custom SIMD helpers
use tools::simd;
// f32 SIMD vector containing 8 elements
use std::arch::x86_64::__m256;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // ANCHOR: preprocess
    // How many __m256 vectors we need for all elements from a row or column of d
    let vecs_per_row = (n + simd::M256_LENGTH - 1) / simd::M256_LENGTH;
    // All rows and columns d packed into __m256 vectors, each initially filled with 8 f32::INFINITYs
    let mut vd = std::vec![simd::m256_infty(); n * vecs_per_row];
    let mut vt = std::vec![simd::m256_infty(); n * vecs_per_row];
    // Function: for one row of __m256 vectors in vd and vt,
    // copy all elements from row i in d into vd and all elements from column i in d into vt
    let preprocess_row = |(i, (vd_row, vt_row)): (usize, (&mut [__m256], &mut [__m256]))| {
        // For every vector (indexed by j) in vt_row and vd_row for given row in d (indexed by i and d_j)
        for (j, (vx, vy)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            // Temporary buffers for elements of two __m256s
            let mut d_tmp = [std::f32::INFINITY; simd::M256_LENGTH];
            let mut t_tmp = [std::f32::INFINITY; simd::M256_LENGTH];
            // Iterate over 8 elements to fill the buffers
            for (b, (x, y)) in d_tmp.iter_mut().zip(t_tmp.iter_mut()).enumerate() {
                // Offset by 8 elements to get correct index mapping of j to d
                let d_j = j * simd::M256_LENGTH + b;
                if d_j < n {
                    *x = d[n * i + d_j];
                    *y = d[n * d_j + i];
                }
            }
            // Initialize __m256 vectors from buffer contents and assign them into the std::vec::Vec containers
            *vx = simd::from_slice(&d_tmp);
            *vy = simd::from_slice(&t_tmp);
        }
    };
    // Fill rows of vd and vt in parallel one pair of rows at a time
    // ANCHOR_END: preprocess
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: preprocess_apply
    vd.par_chunks_mut(vecs_per_row)
        .zip(vt.par_chunks_mut(vecs_per_row))
        .enumerate()
        .for_each(preprocess_row);
    // ANCHOR_END: preprocess_apply
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_exact_mut(vecs_per_row)
        .zip(vt.chunks_exact_mut(vecs_per_row))
        .enumerate()
        .for_each(preprocess_row);
    // ANCHOR: step_row
    // Function: for a row of __m256 elements from vd, compute a row of f32 results into r
    let step_row = |(r_row, vd_row): (&mut [f32], &[__m256])| {
        let vt_rows = vt.chunks_exact(vecs_per_row);
        for (res, vt_row) in r_row.iter_mut().zip(vt_rows) {
            let mut tmp = simd::m256_infty();
            for (&x, &y) in vd_row.iter().zip(vt_row) {
                tmp = simd::min(tmp, simd::add(x, y));
            }
            *res = simd::horizontal_min(tmp);
        }
    };
    // ANCHOR_END: step_row
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: step_row_apply
    r.par_chunks_mut(n)
        .zip(vd.par_chunks(vecs_per_row))
        .for_each(step_row);
    // ANCHOR_END: step_row_apply
    #[cfg(feature = "no-multi-thread")]
    r.chunks_exact_mut(n)
        .zip(vd.chunks_exact(vecs_per_row))
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
