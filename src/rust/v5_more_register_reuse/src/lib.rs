use tools::{create_extern_c_wrapper, simd, simd::f32x8};

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    let vecs_per_col = (n + simd::f32x8_LENGTH - 1) / simd::f32x8_LENGTH;
    // Like v4, but this time pack all elements of d into f32x8s vertically
    let mut vd = std::vec![simd::f32x8_infty(); n * vecs_per_col];
    let mut vt = std::vec![simd::f32x8_infty(); n * vecs_per_col];
    debug_assert!(vd.iter().all(simd::is_aligned));
    debug_assert!(vt.iter().all(simd::is_aligned));
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
    #[cfg(not(feature = "no-multi-thread"))]
    vd.par_chunks_mut(n)
        .zip(vt.par_chunks_mut(n))
        .enumerate()
        .for_each(pack_simd_row_block);
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_mut(n)
        .zip(vt.chunks_mut(n))
        .enumerate()
        .for_each(pack_simd_row_block);

    // Function: for 8 rows starting at row i*8 in d,
    // compute all results for 8 rows starting at row i*8 into r
    let step_row = |(i, (r_row_block, vd_row)): (usize, (&mut [f32], &[f32x8]))| {
        // Chunk up vt into rows, each containing n f32x8 vectors, exactly as vd_row
        for (j, vt_row) in vt.chunks_exact(n).enumerate() {
            // Intermediate results for 8 rows
            let mut tmp = [simd::f32x8_infty(); simd::f32x8_LENGTH];
            // Iterate horizontally over both row, permute each SIMD vector to create 8 unique combinations,
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
            // Swap all results compared against t1 to match the block_j XOR block_i
            // pattern we are using to extract the results
            for i in (1..8).step_by(2) {
                tmp[i] = simd::swap(tmp[i], 1);
            }
            // Set 8 final results (i.e. 64 f32 results in total)
            for block_i in 0..simd::f32x8_LENGTH {
                for (block_j, r_row) in r_row_block.chunks_exact_mut(n).enumerate() {
                    let res_i = block_j + i * simd::f32x8_LENGTH;
                    let res_j = block_i + j * simd::f32x8_LENGTH;
                    if res_i < n && res_j < n {
                        let v = tmp[block_j ^ block_i];
                        let vi = block_i as u8;
                        r_row[res_j] = simd::extract(v, vi);
                    }
                }
            }
        }
    };
    // Chunk up r into row blocks containing 8 rows and vd into rows each containing n f32x8s
    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(simd::f32x8_LENGTH * n)
        .zip(vd.par_chunks(n))
        .enumerate()
        .for_each(step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(simd::f32x8_LENGTH * n)
        .zip(vd.chunks(n))
        .enumerate()
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
