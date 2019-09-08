use tools::{create_extern_c_wrapper, simd, simd::f32x8};

extern crate core;
// For interleaving bits to construct Z-order curve
use core::arch::x86_64::_pdep_u32;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // How many adjacent columns to process during one pass
    // Smaller numbers improve cache locality but adds overhead from having to merge partial results
    const VERTICAL_STRIPE_WIDTH: usize = 500;

    let vecs_per_col = (n + simd::f32x8_LENGTH - 1) / simd::f32x8_LENGTH;

    let mut vd = std::vec![simd::f32x8_infty(); n * vecs_per_col];
    let mut vt = std::vec![simd::f32x8_infty(); n * vecs_per_col];
    debug_assert!(vd.iter().all(simd::is_aligned));
    debug_assert!(vt.iter().all(simd::is_aligned));
    let pack_simd_row = |(i, (vd_row, vt_row)): (usize, (&mut [f32x8], &mut [f32x8]))| {
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
        .for_each(pack_simd_row);
    #[cfg(feature = "no-multi-thread")]
    vd.chunks_mut(n)
        .zip(vt.chunks_mut(n))
        .enumerate()
        .for_each(pack_simd_row);

    // Build a Z-order curve iteration pattern of pairs (i, j) by using interleaved bits of i and j as a sort key
    // Init vector of 3-tuples (ij, i, j)
    let mut row_pairs = std::vec![(0, 0, 0); vecs_per_col * vecs_per_col];
    // Define a function that interleaves one row of indexes
    let interleave_row = |(i, row): (usize, &mut [(usize, usize, usize)])| {
        for (j, x) in row.iter_mut().enumerate() {
            let ij = unsafe { _pdep_u32(i as u32, 0x55555555) | _pdep_u32(j as u32, 0xAAAAAAAA) };
            *x = (ij as usize, i, j);
        }
    };
    // Apply the function independently on all rows and sort by ija
    #[cfg(not(feature = "no-multi-thread"))]
    {
        row_pairs.par_chunks_mut(vecs_per_col).enumerate().for_each(interleave_row);
        row_pairs.par_sort_unstable();
    }
    #[cfg(feature = "no-multi-thread")]
    {
        row_pairs.chunks_mut(vecs_per_col).enumerate().for_each(interleave_row);
        row_pairs.sort_unstable();
    }

    // Non-overlapping working memory for threads to update their results
    // When enumerated in 8 element chunks, indexes the Z-order curve keys
    let mut partial_results = std::vec![simd::f32x8_infty(); vecs_per_col * vecs_per_col * simd::f32x8_LENGTH];

    // Process vd and vt in Z-order one vertical stripe at a time, writing partial results in parallel
    let num_vertical_stripes = (n + VERTICAL_STRIPE_WIDTH - 1) / VERTICAL_STRIPE_WIDTH;
    for stripe in 0..num_vertical_stripes {
        let col_begin = stripe * VERTICAL_STRIPE_WIDTH;
        let next_begin = (stripe + 1) * VERTICAL_STRIPE_WIDTH;
        let col_end = n.min(next_begin);
        // This function computes one block of results into partial_results
        let step_partial_block = |(z, partial_block): (usize, &mut [f32x8])| {
            let (_, i, j) = row_pairs[z];
            // Copy results from previous pass over previous stripe
            let mut tmp = [simd::f32x8_infty(); simd::f32x8_LENGTH];
            tmp.copy_from_slice(&partial_block);
            // Get slices over current stripes of row i and column j
            let vd_stripe = &vd[(n * i + col_begin)..(n * i + col_end)];
            let vt_stripe = &vt[(n * j + col_begin)..(n * j + col_end)];
            for (&d0, &t0) in vd_stripe.iter().zip(vt_stripe) {
                let d2 = simd::swap(d0, 2);
                let d4 = simd::swap(d0, 4);
                let d6 = simd::swap(d4, 2);
                let t1 = simd::swap(t0, 1);
                tmp[0] = simd::min(tmp[0], simd::add(d0, t0));
                tmp[1] = simd::min(tmp[1], simd::add(d0, t1));
                tmp[2] = simd::min(tmp[2], simd::add(d2, t0));
                tmp[3] = simd::min(tmp[3], simd::add(d2, t1));
                tmp[4] = simd::min(tmp[4], simd::add(d4, t0));
                tmp[5] = simd::min(tmp[5], simd::add(d4, t1));
                tmp[6] = simd::min(tmp[6], simd::add(d6, t0));
                tmp[7] = simd::min(tmp[7], simd::add(d6, t1));
            }
            // Store partial results (8 vecs of type f32x8) to global memory
            partial_block.copy_from_slice(&tmp);
        };
        // Process current stripe
        #[cfg(not(feature = "no-multi-thread"))]
        partial_results
            .par_chunks_mut(simd::f32x8_LENGTH)
            .enumerate()
            .for_each(step_partial_block);
        #[cfg(feature = "no-multi-thread")]
        partial_results
            .chunks_mut(simd::f32x8_LENGTH)
            .enumerate()
            .for_each(step_partial_block);
    }

    // TODO remove rz and write results directly into r by reading
    // results from partial_results linearly according to row i and col j of r
    // This requires a mapping from (i, j) into z

    let mut rz = std::vec![0.0; vecs_per_col * vecs_per_col * simd::f32x8_LENGTH * simd::f32x8_LENGTH];
    let set_z_order_result_block = |(z, (rz_block_pair, tmp)): (usize, (&mut [f32], &mut [f32x8]))| {
        let (_, i, j) = row_pairs[z];
        // Result extraction as in v5
        for i in (1..simd::f32x8_LENGTH).step_by(2) {
            tmp[i] = simd::swap(tmp[i], 1);
        }
        for (block_i, rz_block) in rz_block_pair.chunks_exact_mut(simd::f32x8_LENGTH).enumerate() {
            for (block_j, res_z) in rz_block.iter_mut().enumerate() {
                let res_i = block_j + i * simd::f32x8_LENGTH;
                let res_j = block_i + j * simd::f32x8_LENGTH;
                if res_i < n && res_j < n {
                    let v = tmp[block_j ^ block_i];
                    let vi = block_i as u8;
                    *res_z = simd::extract(v, vi);
                }
            }
        }
    };
    // Extract final results in Z-order into rz
    #[cfg(not(feature = "no-multi-thread"))]
    rz.par_chunks_mut(simd::f32x8_LENGTH * simd::f32x8_LENGTH)
        .zip(partial_results.par_chunks_mut(simd::f32x8_LENGTH))
        .enumerate()
        .for_each(set_z_order_result_block);
    #[cfg(feature = "no-multi-thread")]
    rz.chunks_mut(simd::f32x8_LENGTH * simd::f32x8_LENGTH)
        .zip(partial_results.chunks_mut(simd::f32x8_LENGTH))
        .enumerate()
        .for_each(set_z_order_result_block);

    // Finally, copy Z-order results from rz into r in proper order
    for (rz_block_pair, (_, i, j)) in rz.chunks_exact(simd::f32x8_LENGTH * simd::f32x8_LENGTH).zip(row_pairs) {
        for (block_i, rz_block) in rz_block_pair.chunks_exact(simd::f32x8_LENGTH).enumerate() {
            for (block_j, &res_z) in rz_block.iter().enumerate() {
                let res_i = block_j + i * simd::f32x8_LENGTH;
                let res_j = block_i + j * simd::f32x8_LENGTH;
                if res_i < n && res_j < n {
                    r[res_i * n + res_j] = res_z;
                }
            }
        }
    }
}


create_extern_c_wrapper!(step, _step);
