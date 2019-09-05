use tools::{create_extern_c_wrapper, simd, simd::f32x8};

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // ANCHOR: init
    // How many f32x8 vectors we need for all elements from a row or column of d
    let vecs_per_row = (n + simd::f32x8_LENGTH - 1) / simd::f32x8_LENGTH;
    // All rows and columns d packed into f32x8 vectors,
    // each initially filled with 8 f32::INFINITYs
    let mut vd = std::vec![simd::f32x8_infty(); n * vecs_per_row];
    let mut vt = std::vec![simd::f32x8_infty(); n * vecs_per_row];
    // Assert that all addresses of vd and vt are properly aligned to the size of f32x8
    debug_assert!(vd.iter().all(simd::is_aligned));
    debug_assert!(vt.iter().all(simd::is_aligned));
    // ANCHOR_END: init
    // ANCHOR: preprocess
    // Function: for one row of f32x8 vectors in vd and one row of f32x8 vectors in vt,
    // - copy all elements from row 'i' in d,
    // - pack them into f32x8 vectors,
    // - insert all into row 'i' of vd (vd_row)
    // and
    // - copy all elements from column 'i' in d,
    // - pack them into f32x8 vectors,
    // - insert all into row 'i' of vt (vt_row)
    let pack_simd_row = |(i, (vd_row, vt_row)): (usize, (&mut [f32x8], &mut [f32x8]))| {
        // For every SIMD vector at row 'i', column 'jv' in vt and vd
        for (jv, (vx, vy)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            // Temporary buffers for f32 elements of two f32x8s
            let mut vx_tmp = [std::f32::INFINITY; simd::f32x8_LENGTH];
            let mut vy_tmp = [std::f32::INFINITY; simd::f32x8_LENGTH];
            // Iterate over 8 elements to fill the buffers
            for (b, (x, y)) in vx_tmp.iter_mut().zip(vy_tmp.iter_mut()).enumerate() {
                // Offset by 8 elements to get correct index mapping of j to d
                let j = jv * simd::f32x8_LENGTH + b;
                if i < n && j < n {
                    *x = d[n * i + j];
                    *y = d[n * j + i];
                }
            }
            // Initialize f32x8 vectors from buffer contents
            // and assign them into the std::vec::Vec containers
            *vx = simd::from_slice(&vx_tmp);
            *vy = simd::from_slice(&vy_tmp);
        }
    };
    // Fill rows of vd and vt in parallel one pair of rows at a time
    // ANCHOR_END: preprocess
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: preprocess_apply
    vd.par_chunks_mut(vecs_per_row)
        .zip(vt.par_chunks_mut(vecs_per_row))
        .enumerate()
        .for_each(pack_simd_row);
    // ANCHOR_END: preprocess_apply
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_mut(vecs_per_row)
        .zip(vt.chunks_mut(vecs_per_row))
        .enumerate()
        .for_each(pack_simd_row);
    // ANCHOR: step_row
    // Function: for a row of f32x8 elements from vd, compute a row of f32 results into r
    let step_row = |(r_row, vd_row): (&mut [f32], &[f32x8])| {
        let vt_rows = vt.chunks_exact(vecs_per_row);
        // ANCHOR: step_row_inner
        for (res, vt_row) in r_row.iter_mut().zip(vt_rows) {
            let tmp = vd_row.iter()
                            .zip(vt_row)
                            .fold(simd::f32x8_infty(),
                                  |v, (&x, &y)| simd::min(v, simd::add(x, y)));
            *res = simd::horizontal_min(tmp);
        }
        // ANCHOR_END: step_row_inner
    };
    // ANCHOR_END: step_row
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: step_row_apply
    r.par_chunks_mut(n)
        .zip(vd.par_chunks(vecs_per_row))
        .for_each(step_row);
    // ANCHOR_END: step_row_apply
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(n)
        .zip(vd.chunks(vecs_per_row))
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
