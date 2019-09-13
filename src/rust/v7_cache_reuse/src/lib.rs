use tools::{create_extern_c_wrapper, simd, simd::f32x8, z_encode};

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    // ANCHOR: init
    // How many adjacent columns to process during one pass
    // Smaller numbers improve cache locality but add overhead
    // from having to merge partial results
    const COLS_PER_STRIPE: usize = 500;
    let vecs_per_col = (n + simd::f32x8_LENGTH - 1) / simd::f32x8_LENGTH;
    // ANCHOR_END: init

    // ANCHOR: interleave
    // Build a Z-order curve iteration pattern of pairs (i, j)
    // by using interleaved bits of i and j as a sort key
    let mut row_pairs = std::vec![(0, 0, 0); vecs_per_col * vecs_per_col];
    // Define a function that interleaves one row of indexes
    let interleave_row = |(i, row): (usize, &mut [(usize, usize, usize)])| {
        for (j, x) in row.iter_mut().enumerate() {
            let z = z_encode(i as u32, j as u32);
            *x = (z as usize, i, j);
        }
    };
    // ANCHOR_END: interleave
    #[cfg(not(feature = "no-multi-thread"))]
    {
    // ANCHOR: interleave_apply
    // Apply the function independently on all rows and sort by ija
    row_pairs
        .par_chunks_mut(vecs_per_col)
        .enumerate()
        .for_each(interleave_row);
    // We don't need stable sort since there are no duplicate keys
    row_pairs.par_sort_unstable();
    // ANCHOR_END: interleave_apply
    }
    #[cfg(feature = "no-multi-thread")]
    {
        row_pairs
            .chunks_mut(vecs_per_col)
            .enumerate()
            .for_each(interleave_row);
        row_pairs.sort_unstable();
    }

    // ANCHOR: init_stripe_data
    // We'll be processing the input one stripe at a time
    let mut vd = std::vec![simd::f32x8_infty(); COLS_PER_STRIPE * vecs_per_col];
    let mut vt = std::vec![simd::f32x8_infty(); COLS_PER_STRIPE * vecs_per_col];
    // Non-overlapping working memory for threads to update their results
    // When enumerated in 8 element chunks, indexes the Z-order curve keys
    let mut partial_results = std::vec![simd::f32x8_infty(); vecs_per_col * vecs_per_col * simd::f32x8_LENGTH];
    // ANCHOR_END: init_stripe_data

    // ANCHOR: stripe_loop_head
    // Process vd and vt in Z-order one vertical stripe at a time, writing partial results in parallel
    let num_vertical_stripes = (n + COLS_PER_STRIPE - 1) / COLS_PER_STRIPE;
    for stripe in 0..num_vertical_stripes {
        let col_begin = stripe * COLS_PER_STRIPE;
        let col_end = n.min((stripe + 1) * COLS_PER_STRIPE);
        // ANCHOR_END: stripe_loop_head
        // Preprocessing as in v5, but one vertical stripe at a time
        let pack_simd_row = |(i, (vd_stripe, vt_stripe)): (usize, (&mut [f32x8], &mut [f32x8]))| {
            for (jv, (vx, vy)) in vd_stripe.iter_mut().zip(vt_stripe.iter_mut()).enumerate() {
                let mut vx_tmp = [std::f32::INFINITY; simd::f32x8_LENGTH];
                let mut vy_tmp = [std::f32::INFINITY; simd::f32x8_LENGTH];
                for (b, (x, y)) in vx_tmp.iter_mut().zip(vy_tmp.iter_mut()).enumerate() {
                    let d_row = i * simd::f32x8_LENGTH + b;
                    let d_col = col_begin + jv;
                    if d_row < n && d_col < col_end {
                        *x = d[n * d_row + d_col];
                        *y = d[n * d_col + d_row];
                    }
                }
                *vx = simd::from_slice(&vx_tmp);
                *vy = simd::from_slice(&vy_tmp);
            }
        };
        #[cfg(not(feature = "no-multi-thread"))]
        vd.par_chunks_mut(COLS_PER_STRIPE)
            .zip(vt.par_chunks_mut(COLS_PER_STRIPE))
            .enumerate()
            .for_each(pack_simd_row);
        #[cfg(feature = "no-multi-thread")]
        vd.chunks_mut(COLS_PER_STRIPE)
            .zip(vt.chunks_mut(COLS_PER_STRIPE))
            .enumerate()
            .for_each(pack_simd_row);
        // ANCHOR_END: stripe_loop_head
        // ANCHOR: stripe_loop_step_partial_block
        // Function: for a f32x8 block of partial results and indexes row i col j,
        // 1. Load tmp from partial results
        // 2. Accumulate results for row i and column j into tmp
        // 3. Write tmp into the original partial results block
        let step_partial_block = |(prev_tmp, &(_, i, j)): (&mut [f32x8], &(usize, usize, usize))| {
            // Copy results from previous pass over previous stripe
            let mut tmp = [simd::f32x8_infty(); simd::f32x8_LENGTH];
            tmp.copy_from_slice(&prev_tmp);
            // Get slices over current stripes of row i and column j
            let vd_row = &vd[(COLS_PER_STRIPE * i)..(COLS_PER_STRIPE * (i + 1))];
            let vt_row = &vt[(COLS_PER_STRIPE * j)..(COLS_PER_STRIPE * (j + 1))];
            for (&d0, &t0) in vd_row.iter().zip(vt_row) {
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
            // for processing next stripe
            prev_tmp.copy_from_slice(&tmp);
        };
        // ANCHOR_END: stripe_loop_step_partial_block
        #[cfg(not(feature = "no-multi-thread"))]
        // ANCHOR: stripe_loop_step_partial_block_apply
        // Process current stripe in parallel, each thread filling one `tmp` block
        partial_results
            .par_chunks_mut(simd::f32x8_LENGTH)
            .zip(row_pairs.par_iter())
            .for_each(step_partial_block);
        // ANCHOR_END: stripe_loop_step_partial_block_apply
        #[cfg(feature = "no-multi-thread")]
        partial_results
            .chunks_mut(simd::f32x8_LENGTH)
            .zip(row_pairs.iter())
            .for_each(step_partial_block);
    }

    // ANCHOR: replace_sort_key
    // Replace ij sorting key by linear index to get a mapping to partial_results,
    // then sort row_pairs by (i, j)
    let replace_z_index_row = |(z_row, index_row): (usize, &mut [(usize, usize, usize)])| {
        for (z, idx) in index_row.iter_mut().enumerate() {
            let (_, i, j) = *idx;
            *idx = (z_row * vecs_per_col + z, i, j);
        }
    };
    let key_ij = |&idx: &(usize, usize, usize)| { (idx.1, idx.2) };
    // ANCHOR_END: replace_sort_key
    #[cfg(not(feature = "no-multi-thread"))]
    {
    // ANCHOR: replace_sort_key_apply
    row_pairs
        .par_chunks_mut(vecs_per_col)
        .enumerate()
        .for_each(replace_z_index_row);
    row_pairs.par_sort_unstable_by_key(key_ij);
    // ANCHOR_END: replace_sort_key_apply
    }
    #[cfg(feature = "no-multi-thread")]
    {
        row_pairs
            .chunks_mut(vecs_per_col)
            .enumerate()
            .for_each(replace_z_index_row);
        row_pairs.sort_unstable_by_key(key_ij);
    }

    // ANCHOR: set_z_order_result_block
    // Function: for 8 rows in r starting at row i*8,
    // read partial results at z-index corresponding to each row i and column j
    // and write them to r
    let set_z_order_result_block = |(i, r_row_block): (usize, &mut [f32])| {
        for j in 0..vecs_per_col {
            // Get z-order index for row i and column j
            let z = row_pairs[i * vecs_per_col + j].0 * simd::f32x8_LENGTH;
            // Load tmp from z-order partial results for this i, j pair
            let mut tmp = [simd::f32x8_infty(); simd::f32x8_LENGTH];
            tmp.copy_from_slice(&partial_results[z..z + simd::f32x8_LENGTH]);
            // Continue exactly as in v5
            for k in (1..simd::f32x8_LENGTH).step_by(2) {
                tmp[k] = simd::swap(tmp[k], 1);
            }
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
        }
    };
    // ANCHOR_END: set_z_order_result_block
    #[cfg(not(feature = "no-multi-thread"))]
    // ANCHOR: set_z_order_result_block_apply
    r.par_chunks_mut(simd::f32x8_LENGTH * n)
        .enumerate()
        .for_each(set_z_order_result_block);
    // ANCHOR_END: set_z_order_result_block_apply
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(simd::f32x8_LENGTH * n)
        .enumerate()
        .for_each(set_z_order_result_block);
}


create_extern_c_wrapper!(step, _step);
