#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;

// f32 SIMD vector containing 8 elements
use std::arch::x86_64::__m256;

extern crate tools;
use tools::simd; // Custom SIMD helpers


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    #[allow(non_upper_case_globals)]
    // We are using 256-bit single precision float vectors so this will be equal to 8
    const vec_width: usize = simd::M256_LENGTH;
    let vecs_per_row = (n + vec_width - 1) / vec_width;
    // Pack d and its transpose into m256 vectors, each containing 8 f32::INFINITYs
    let mut vd = std::vec![simd::m256_infty(); n * vecs_per_row];
    let mut vt = std::vec![simd::m256_infty(); n * vecs_per_row];
    {
        let vd_rows = vd.chunks_mut(vecs_per_row);
        let vt_rows = vt.chunks_mut(vecs_per_row);
        for (row, (vd_row, vt_row)) in vd_rows.zip(vt_rows).enumerate() {
            // For every vector (indexed by col) in vt and vd on this row
            for (col, (vx, vy)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
                // Build 8 element arrays for vd and vt, right-padded with f32::INFINITY
                // These will be packed with 8 f32s, which are then converted to SIMD types
                let mut d_tmp = [std::f32::INFINITY; vec_width];
                let mut t_tmp = [std::f32::INFINITY; vec_width];
                // Iterate over 8 elements to fill the SIMD types
                for (vec_i, (x, y)) in d_tmp.iter_mut().zip(t_tmp.iter_mut()).enumerate() {
                    let d_col = col * vec_width + vec_i;
                    if d_col < n {
                        *x = d[n * row + d_col];
                        *y = d[n * d_col + row];
                    }
                }
                // Copy tmp slice contents to 256-bit simd-vectors and assign the simd-vectors into the std::vec containers
                *vx = simd::from_slice(&d_tmp);
                *vy = simd::from_slice(&t_tmp);
            }
        }
    }
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
    r.par_chunks_mut(n).zip(vd.par_chunks(vecs_per_row)).for_each(step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n).zip(vd.chunks(vecs_per_row)).for_each(step_row);
}


#[no_mangle]
pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: i32) {
    let result = std::panic::catch_unwind(|| {
        let d = std::slice::from_raw_parts(d_raw, (n * n) as usize);
        let mut r = std::slice::from_raw_parts_mut(r_raw, (n * n) as usize);
        _step(&mut r, d, n as usize);
    });
    if result.is_err() {
        eprintln!("error: rust panicked");
    }
}
