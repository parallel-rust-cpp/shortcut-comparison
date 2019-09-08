use tools::{create_extern_c_wrapper, simd, simd::f32x8};

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // ANCHOR: init
    let vecs_per_col = (n + simd::f32x8_LENGTH - 1) / simd::f32x8_LENGTH;
    // Like v4, but this time pack all elements of d into f32x8s vertically
    let mut vd = std::vec![simd::f32x8_infty(); n * vecs_per_col];
    let mut vt = std::vec![simd::f32x8_infty(); n * vecs_per_col];
    // ANCHOR_END: init
    debug_assert!(vd.iter().all(simd::is_aligned));
    debug_assert!(vt.iter().all(simd::is_aligned));
    // ANCHOR: pack_simd
    // Function: for row i of vd and row i of vt,
    // copy 8 rows of d into vd and 8 columns of d into vt
    let pack_simd_row_block = |(i, (vd_row, vt_row)): (usize, (&mut [f32x8], &mut [f32x8]))| {
        for (jv, (vx, vy)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            let mut vx_tmp = [std::f32::INFINITY; simd::f32x8_LENGTH];
            let mut vy_tmp = [std::f32::INFINITY; simd::f32x8_LENGTH];
            for (b, (x, y)) in vx_tmp.iter_mut().zip(vy_tmp.iter_mut()).enumerate() {
                let j = i * simd::f32x8_LENGTH + b;
                if i < n && j < n {
                    *x = d[n * j + jv];
                    *y = d[n * jv + j];
                }
            }
            *vx = simd::from_slice(&vx_tmp);
            *vy = simd::from_slice(&vy_tmp);
        }
    };
    // ANCHOR_END: pack_simd
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: pack_simd_apply
    vd.par_chunks_mut(n)
        .zip(vt.par_chunks_mut(n))
        .enumerate()
        .for_each(pack_simd_row_block);
    // ANCHOR_END: pack_simd_apply
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_mut(n)
        .zip(vt.chunks_mut(n))
        .enumerate()
        .for_each(pack_simd_row_block);

    // ANCHOR: step_row_block
    //// ANCHOR: step_row_block_init
    ////// ANCHOR: step_row_block_header
    // Function: for 8 rows in d, compute all results for 8 rows into r
    let step_row_block = |(r_row_block, vd_row): (&mut [f32], &[f32x8])| {
        ////// ANCHOR_END: step_row_block_header
        // Chunk up vt into rows, each containing n f32x8 vectors,
        // exactly as vd_row
        for (j, vt_row) in vt.chunks_exact(n).enumerate() {
            // Intermediate results for 8 rows
            let mut tmp = [simd::f32x8_infty(); simd::f32x8_LENGTH];
            //// ANCHOR_END: step_row_block_init
            //// ANCHOR: step_row_block_inner
            // Iterate horizontally over both rows,
            // permute elements of each `f32x8` to create 8 unique combinations,
            // and compute 8 minimums from all combinations
            for (&d0, &t0) in vd_row.iter().zip(vt_row) {
                // Compute permutations of f32x8 elements
                // 2 3 0 1 6 7 4 5
                let d2 = simd::swap(d0, 2);
                // 4 5 6 7 0 1 2 3
                let d4 = simd::swap(d0, 4);
                // 6 7 4 5 2 3 0 1
                let d6 = simd::swap(d4, 2);
                // 1 0 3 2 5 4 7 6
                let t1 = simd::swap(t0, 1);
                // Compute 8 independent, intermediate results for 8 rows
                tmp[0] = simd::min(tmp[0], simd::add(d0, t0));
                tmp[1] = simd::min(tmp[1], simd::add(d0, t1));
                tmp[2] = simd::min(tmp[2], simd::add(d2, t0));
                tmp[3] = simd::min(tmp[3], simd::add(d2, t1));
                tmp[4] = simd::min(tmp[4], simd::add(d4, t0));
                tmp[5] = simd::min(tmp[5], simd::add(d4, t1));
                tmp[6] = simd::min(tmp[6], simd::add(d6, t0));
                tmp[7] = simd::min(tmp[7], simd::add(d6, t1));
            }
            //// ANCHOR_END: step_row_block_inner
            //// ANCHOR: step_row_block_results
            // Swap elements of f32x8s at odd indexes to enable a linear iteration
            // pattern for index tmp_j when extracting elements
            for i in (1..simd::f32x8_LENGTH).step_by(2) {
                tmp[i] = simd::swap(tmp[i], 1);
            }
            // Set 8 final results (i.e. 64 f32 results in total)
            for (tmp_i, r_row) in r_row_block.chunks_exact_mut(n).enumerate() {
                for tmp_j in 0..simd::f32x8_LENGTH {
                    let res_j = j * simd::f32x8_LENGTH + tmp_j;
                    if res_j < n {
                        let v = tmp[tmp_i ^ tmp_j];
                        let vi = tmp_j as u8;
                        r_row[res_j] = simd::extract(v, vi);
                    }
                }
            }
            //// ANCHOR_END: step_row_block_results
        }
    };
    // ANCHOR_END: step_row_block
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: step_row_block_apply
    // Chunk up r into row blocks containing 8 rows, each containing n f32s,
    // and chunk up vd into rows, each containing n f32x8s
    r.par_chunks_mut(simd::f32x8_LENGTH * n)
        .zip(vd.par_chunks(n))
        .for_each(step_row_block);
    // ANCHOR_END: step_row_block_apply
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(simd::f32x8_LENGTH * n)
        .zip(vd.chunks(n))
        .for_each(step_row_block);
}


create_extern_c_wrapper!(step, _step);
