use tools::{create_extern_c_wrapper, simd, simd::f32x8};
// izip for zipping multiple iterators
#[macro_use]
extern crate itertools;
// Destructure iterator into tuple
use itertools::Itertools;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // ANCHOR: init
    const BLOCK_HEIGHT: usize = 3;
    let blocks_per_col = (n + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    let vecs_per_row = (n + simd::f32x8_LENGTH - 1) / simd::f32x8_LENGTH;
    let padded_height = BLOCK_HEIGHT * blocks_per_col;
    // Preprocess exactly as in v3_simd, but make sure the amount of rows is divisible by BLOCK_HEIGHT
    let mut vd = std::vec![simd::f32x8_infty(); padded_height * vecs_per_row];
    let mut vt = std::vec![simd::f32x8_infty(); padded_height * vecs_per_row];
    // ANCHOR_END: init
    debug_assert!(vd.iter().all(simd::is_aligned));
    debug_assert!(vt.iter().all(simd::is_aligned));
    let pack_simd_row = |(i, (vd_row, vt_row)): (usize, (&mut [f32x8], &mut [f32x8]))| {
        for (jv, (vx, vy)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            let mut vx_tmp = [std::f32::INFINITY; simd::f32x8_LENGTH];
            let mut vy_tmp = [std::f32::INFINITY; simd::f32x8_LENGTH];
            for (b, (x, y)) in vx_tmp.iter_mut().zip(vy_tmp.iter_mut()).enumerate() {
                let j = jv * simd::f32x8_LENGTH + b;
                if i < n && j < n {
                    *x = d[n * i + j];
                    *y = d[n * j + i];
                }
            }
            *vx = simd::from_slice(&vx_tmp);
            *vy = simd::from_slice(&vy_tmp);
        }
    };
    #[cfg(not(feature = "no-multi-thread"))]
    vd.par_chunks_mut(vecs_per_row)
        .zip(vt.par_chunks_mut(vecs_per_row))
        .enumerate()
        .for_each(pack_simd_row);
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_mut(vecs_per_row)
        .zip(vt.chunks_mut(vecs_per_row))
        .enumerate()
        .for_each(pack_simd_row);

    // ANCHOR: step_row_block
    //// ANCHOR: step_row_block_head
    // Function: For a row block vd_row_block containing 3 rows of f32x8 vectors,
    // compute results for all row combinations of vd_row_block and row blocks of vt
    let step_row_block = |(i, (r_row_block, vd_row_block)): (usize, (&mut [f32], &[f32x8]))| {
        // Chunk up vt into blocks exactly as vd
        let vt_row_blocks = vt.chunks_exact(BLOCK_HEIGHT * vecs_per_row);
        // Compute results for all combinations of row blocks from vd and vt
        for (j, vt_row_block) in vt_row_blocks.enumerate() {
            //// ANCHOR_END: step_row_block_head
            //// ANCHOR: step_row_block_init
            // Partial results for 9 f32x8 row pairs
            let mut tmp0 = simd::f32x8_infty();
            let mut tmp1 = simd::f32x8_infty();
            let mut tmp2 = simd::f32x8_infty();
            let mut tmp3 = simd::f32x8_infty();
            let mut tmp4 = simd::f32x8_infty();
            let mut tmp5 = simd::f32x8_infty();
            let mut tmp6 = simd::f32x8_infty();
            let mut tmp7 = simd::f32x8_infty();
            let mut tmp8 = simd::f32x8_infty();
            // Extract all rows from the row blocks
            let mut vd_rows = vd_row_block.chunks_exact(vecs_per_row);
            let mut vt_rows = vt_row_block.chunks_exact(vecs_per_row);
            let (vd_row_0, vd_row_1, vd_row_2) = vd_rows.next_tuple().unwrap();
            let (vt_row_0, vt_row_1, vt_row_2) = vt_rows.next_tuple().unwrap();
            //// ANCHOR_END: step_row_block_init
            //// ANCHOR: step_row_block_inner_loop
            // Move horizontally, computing 3 x 3 results for each column
            // At each iteration, load two 'vertical stripes' of 3 f32x8 vectors
            let block_combinations = izip!(vd_row_0, vd_row_1, vd_row_2, vt_row_0, vt_row_1, vt_row_2);
            for (&d0, &d1, &d2, &t0, &t1, &t2) in block_combinations {
                // Combine all 9 pairs of f32x8 vectors from 6 rows at every column
                tmp0 = simd::min(tmp0, simd::add(d0, t0));
                tmp1 = simd::min(tmp1, simd::add(d0, t1));
                tmp2 = simd::min(tmp2, simd::add(d0, t2));
                tmp3 = simd::min(tmp3, simd::add(d1, t0));
                tmp4 = simd::min(tmp4, simd::add(d1, t1));
                tmp5 = simd::min(tmp5, simd::add(d1, t2));
                tmp6 = simd::min(tmp6, simd::add(d2, t0));
                tmp7 = simd::min(tmp7, simd::add(d2, t1));
                tmp8 = simd::min(tmp8, simd::add(d2, t2));
            }
            let tmp = [tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8];
            //// ANCHOR_END: step_row_block_inner_loop
            //// ANCHOR: step_row_block_results
            // Set 9 final results for all combinations of 3 rows starting at i and 3 rows starting at j
            for (block_i, (r_row, tmp_row)) in r_row_block.chunks_exact_mut(n).zip(tmp.chunks_exact(BLOCK_HEIGHT)).enumerate() {
                for (block_j, tmp_res) in tmp_row.iter().enumerate() {
                    let res_i = i * BLOCK_HEIGHT + block_i;
                    let res_j = j * BLOCK_HEIGHT + block_j;
                    if res_i < n && res_j < n {
                        // Reduce one f32x8 to the final result for one pair of rows
                        r_row[res_j] = simd::horizontal_min(*tmp_res);
                    }
                }
            }
            //// ANCHOR_END: step_row_block_results
        }
    };
    // ANCHOR_END: step_row_block
    // Chunk up r and vd into row blocks and compute results of all row combinations between vd and vt
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: step_row_apply
    r.par_chunks_mut(BLOCK_HEIGHT * n)
        .zip(vd.par_chunks(BLOCK_HEIGHT * vecs_per_row))
        .enumerate()
        .for_each(step_row_block);
    // ANCHOR_END: step_row_apply
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(BLOCK_HEIGHT * n)
        .zip(vd.chunks(BLOCK_HEIGHT * vecs_per_row))
        .enumerate()
        .for_each(step_row_block);
}


create_extern_c_wrapper!(step, _step);
