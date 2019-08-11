#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*; // Parallel chunks iterator, parallel sort

extern crate tools;
use tools::simd; // Custom SIMD helpers
use std::arch::x86_64::__m256;

extern crate core;
use core::arch::x86_64::_pdep_u32; // For interleaving bits to construct Z-order curve


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    #[allow(non_upper_case_globals)]
    const m256_length: usize = simd::M256_LENGTH;
    // TODO row slices for cache locality
    // const row_chunk_width: usize = 400;
    let vecs_per_col = (n + m256_length - 1) / m256_length;

    // Initialize memory for packing d and its transpose into containers of vertical m256 vectors
    let mut vd = std::vec![simd::m256_infty(); n * vecs_per_col];
    let mut vt = std::vec![simd::m256_infty(); n * vecs_per_col];
    // Define a function to be applied on each row of d and its transpose
    let preprocess_row = |(row, (vd_row, vt_row)): (usize, (&mut [__m256], &mut [__m256]))| {
        for (col, (vd_elem, vt_elem)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            // Build 8 element arrays for vd and vt, with infinity padding
            let mut d_slice = [std::f32::INFINITY; m256_length];
            let mut t_slice = [std::f32::INFINITY; m256_length];
            for vec_j in 0..m256_length {
                let j = row * m256_length + vec_j;
                if j < n {
                    d_slice[vec_j] = d[n * j + col];
                    t_slice[vec_j] = d[n * col + j];
                }
            }
            // Convert arrays to 256-bit vectors and assign to vector containers
            *vd_elem = simd::from_slice(&d_slice);
            *vt_elem = simd::from_slice(&t_slice);
        }
    };
    // Normalize each row in parallel simultaneously into both vt and vd
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
    // Init vector of 3-tuples
    let mut row_pairs = std::vec![(0, 0, 0); vecs_per_col * vecs_per_col];
    // Define a function that interleaves one row of indexes
    let interleave_row = |(i, row): (usize, &mut [(usize, usize, usize)])| {
        for (j, x) in row.iter_mut().enumerate() {
            let ija = unsafe { _pdep_u32(i as u32, 0x55555555) | _pdep_u32(j as u32, 0xAAAAAAAA) };
            *x = (ija as usize, i, j);
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
    // When enumerated, also conveniently indexes the Z-order curve keys in m256_length chunks
    let mut partial_results = std::vec![simd::m256_infty(); vecs_per_col * vecs_per_col * m256_length];

    // This function computes one block of results into partial_results
    let _step_partial_block = |(z, partial_block): (usize, &mut [__m256])| {
        let (_, i, j) = row_pairs[z];
        // Intermediate results
        let mut tmp = [simd::m256_infty(); m256_length];
        // Horizontally compute 8 minimums from each pair of vertical vectors for this row chunk
        for col in 0..n {
            // Load vector pair
            let a0 = vd[n * i + col];
            let b0 = vt[n * j + col];
            // Compute permutations
            let a2 = simd::swap(a0, 2);
            let a4 = simd::swap(a0, 4);
            let a6 = simd::swap(a4, 2);
            let b1 = simd::swap(b0, 1);
            // Compute 8 independent, intermediate results by combining each permutation
            tmp[0] = simd::min(tmp[0], simd::add(a0, b0));
            tmp[1] = simd::min(tmp[1], simd::add(a0, b1));
            tmp[2] = simd::min(tmp[2], simd::add(a2, b0));
            tmp[3] = simd::min(tmp[3], simd::add(a2, b1));
            tmp[4] = simd::min(tmp[4], simd::add(a4, b0));
            tmp[5] = simd::min(tmp[5], simd::add(a4, b1));
            tmp[6] = simd::min(tmp[6], simd::add(a6, b0));
            tmp[7] = simd::min(tmp[7], simd::add(a6, b1));
        }
        // Store partial results (8 vecs of type __m256) to global memory
        partial_block.copy_from_slice(&tmp);
    };

    // Do the heavy lifting in chunks of size m256_length
    // If run in parallel, this creates (vecs_per_col * vecs_per_col) independent jobs for threads to steal

    #[cfg(not(feature = "no-multi-thread"))]
    partial_results
        .par_chunks_mut(m256_length)
        .enumerate()
        .for_each(_step_partial_block);
    #[cfg(feature = "no-multi-thread")]
    partial_results
        .chunks_mut(m256_length)
        .enumerate()
        .for_each(_step_partial_block);

    // Extract final results
    // TODO in parallel
    for (z, (_, i, j)) in row_pairs.iter().enumerate() {
        let begin = m256_length * z;
        let mut tmp = partial_results[begin..begin + m256_length].to_owned();
        // Swap all comparisons of b1 back for easier vector element extraction
        tmp[1] = simd::swap(tmp[1], 1);
        tmp[3] = simd::swap(tmp[3], 1);
        tmp[5] = simd::swap(tmp[5], 1);
        tmp[7] = simd::swap(tmp[7], 1);
        // Extract each element of each vector and assign to result chunk
        for jb in 0..m256_length {
            for ib in 0..m256_length {
                let res_i = ib + i * m256_length;
                let res_j = jb + j * m256_length;
                if res_i < n && res_j < n {
                    let v = tmp[ib ^ jb];
                    let vi = jb as u8;
                    r[res_i * n + res_j] = simd::extract(v, vi);
                }
            }
        }
    }
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
