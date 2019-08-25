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
    // How many m256 vectors (with 8 f32s each) we need to pack all elements from a row of d
    let vecs_per_row = (n + simd::M256_LENGTH - 1) / simd::M256_LENGTH;
    // Pack d and its transpose into m256 vectors row-wise, each vector containing 8 f32::INFINITYs
    let mut vd = std::vec![simd::m256_infty(); n * vecs_per_row];
    let mut vt = std::vec![simd::m256_infty(); n * vecs_per_row];
    // Function: for one row of SIMD-vectors in vd and vt, copy values from from d,
    // such that vd contains values from row 'row' in d and vt contains values from column 'row' in d
    let preprocess_row = |(row, (vd_row, vt_row)): (usize, (&mut [__m256], &mut [__m256]))| {
        // For every vector (indexed by col) in vt_row and vd_row for given row in d (indexed by row and d_col)
        for (col, (vx, vy)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            // Buffers containing 8 elements, which are converted to SIMD vectors
            let mut d_tmp = [std::f32::INFINITY; simd::M256_LENGTH];
            let mut t_tmp = [std::f32::INFINITY; simd::M256_LENGTH];
            // Iterate over 8 elements to fill the buffers
            for (vec_i, (x, y)) in d_tmp.iter_mut().zip(t_tmp.iter_mut()).enumerate() {
                // Offset by SIMD vector length to get correct index mapping to d
                let d_col = col * simd::M256_LENGTH + vec_i;
                if d_col < n {
                    *x = d[n * row + d_col];
                    *y = d[n * d_col + row];
                }
            }
            // Copy tmp slice contents to 256-bit simd-vectors and assign the simd-vectors into the std::vec containers
            *vx = simd::from_slice(&d_tmp);
            *vy = simd::from_slice(&t_tmp);
        }
    };
    // Perform preprocessing in parallel for each row pair
    #[cfg(not(feature = "no-multi-thread"))]
    vd.par_chunks_mut(vecs_per_row)
        .zip(vt.par_chunks_mut(vecs_per_row))
        .enumerate()
        .for_each(preprocess_row);
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_mut(vecs_per_row)
        .zip(vt.chunks_mut(vecs_per_row))
        .enumerate()
        .for_each(preprocess_row);
    // Function: same as in v1 but we are using SIMD types to process 8 columns simultaneously
    let step_row = |(r_row, vd_row): (&mut [f32], &[__m256])| {
        let vt_rows = vt.chunks(vecs_per_row);
        for (res, vt_row) in r_row.iter_mut().zip(vt_rows) {
            let mut tmp = simd::m256_infty();
            for (&x, &y) in vd_row.iter().zip(vt_row) {
                tmp = simd::min(tmp, simd::add(x, y));
            }
            *res = simd::horizontal_min(tmp);
        }
    };
    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(n)
        .zip(vd.par_chunks(vecs_per_row))
        .for_each(step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n)
        .zip(vd.chunks(vecs_per_row))
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
