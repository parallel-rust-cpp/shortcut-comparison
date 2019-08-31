use tools::create_extern_c_wrapper;
use tools::simd;
use std::arch::x86_64::__m256;

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

    let vecs_per_col = (n + simd::M256_LENGTH - 1) / simd::M256_LENGTH;

    let mut vd = std::vec![simd::m256_infty(); n * vecs_per_col];
    let mut vt = std::vec![simd::m256_infty(); n * vecs_per_col];
    let preprocess_row = |(row, (vd_row, vt_row)): (usize, (&mut [__m256], &mut [__m256]))| {
        for (col, (vd_elem, vt_elem)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            let mut d_slice = [std::f32::INFINITY; simd::M256_LENGTH];
            let mut t_slice = [std::f32::INFINITY; simd::M256_LENGTH];
            for vec_j in 0..simd::M256_LENGTH {
                let j = row * simd::M256_LENGTH + vec_j;
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

    // Working memory for each thread to update their results
    // When enumerated, also conveniently indexes the Z-order curve keys in 8 element chunks
    let mut partial_results = std::vec![simd::m256_infty(); vecs_per_col * vecs_per_col * simd::M256_LENGTH];

    // Process vd and vt in Z-order one vertical stripe at a time, writing partial results in parallel
    let num_vertical_stripes = (n + VERTICAL_STRIPE_WIDTH - 1) / VERTICAL_STRIPE_WIDTH;
    for stripe in 0..num_vertical_stripes {
        let col_begin = stripe * VERTICAL_STRIPE_WIDTH;
        let next_begin = (stripe + 1) * VERTICAL_STRIPE_WIDTH;
        let col_end = n.min(next_begin);
        // This function computes one block of results into partial_results
        let step_partial_block = |(z, partial_block): (usize, &mut [__m256])| {
            let (_, i, j) = row_pairs[z];
            // Copy results from previous pass over previous stripe
            assert_eq!(partial_block.len(), simd::M256_LENGTH);
            let mut tmp = [simd::m256_infty(); simd::M256_LENGTH];
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
            // Store partial results (8 vecs of type __m256) to global memory
            partial_block.copy_from_slice(&tmp);
        };
        // Process current stripe
        #[cfg(not(feature = "no-multi-thread"))]
        partial_results
            .par_chunks_mut(simd::M256_LENGTH)
            .enumerate()
            .for_each(step_partial_block);
        #[cfg(feature = "no-multi-thread")]
        partial_results
            .chunks_mut(simd::M256_LENGTH)
            .enumerate()
            .for_each(step_partial_block);
    }

    let mut rz = std::vec![0.0; vecs_per_col * vecs_per_col * simd::M256_LENGTH * simd::M256_LENGTH];
    let set_z_order_result_block = |(z, (rz_block_pair, tmp)): (usize, (&mut [f32], &mut [__m256]))| {
        let (_, i, j) = row_pairs[z];
        tmp[1] = simd::swap(tmp[1], 1);
        tmp[3] = simd::swap(tmp[3], 1);
        tmp[5] = simd::swap(tmp[5], 1);
        tmp[7] = simd::swap(tmp[7], 1);
        for (block_i, rz_block) in rz_block_pair.chunks_exact_mut(simd::M256_LENGTH).enumerate() {
            for (block_j, res_z) in rz_block.iter_mut().enumerate() {
                let res_i = block_j + i * simd::M256_LENGTH;
                let res_j = block_i + j * simd::M256_LENGTH;
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
    rz.par_chunks_mut(simd::M256_LENGTH * simd::M256_LENGTH)
        .zip(partial_results.par_chunks_mut(simd::M256_LENGTH))
        .enumerate()
        .for_each(set_z_order_result_block);
    #[cfg(feature = "no-multi-thread")]
    rz.chunks_mut(simd::M256_LENGTH * simd::M256_LENGTH)
        .zip(partial_results.chunks_mut(simd::M256_LENGTH))
        .enumerate()
        .for_each(set_z_order_result_block);

    // TODO is it possible to write results into r in Z-order in parallel?
    // then the sequential step below would become redundant

    // Finally, copy Z-order results from rz into r in proper order
    for (rz_block_pair, (_, i, j)) in rz.chunks_exact(simd::M256_LENGTH * simd::M256_LENGTH).zip(row_pairs) {
        for (block_i, rz_block) in rz_block_pair.chunks_exact(simd::M256_LENGTH).enumerate() {
            for (block_j, &res_z) in rz_block.iter().enumerate() {
                let res_i = block_j + i * simd::M256_LENGTH;
                let res_j = block_i + j * simd::M256_LENGTH;
                if res_i < n && res_j < n {
                    r[res_i * n + res_j] = res_z;
                }
            }
        }
    }
}


create_extern_c_wrapper!(step, _step);
