use tools::create_extern_c_wrapper;
use tools::simd;
use std::arch::x86_64::__m256;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    #[allow(non_upper_case_globals)]
    const vec_width: usize = simd::M256_LENGTH;
    let vecs_per_col = (n + vec_width - 1) / vec_width;

    let mut vd = std::vec![simd::m256_infty(); n * vecs_per_col];
    let mut vt = std::vec![simd::m256_infty(); n * vecs_per_col];
    let preprocess_row = |(row, (vd_row, vt_row)): (usize, (&mut [__m256], &mut [__m256]))| {
        for (col, (vd_elem, vt_elem)) in vd_row.iter_mut().zip(vt_row.iter_mut()).enumerate() {
            let mut d_slice = [std::f32::INFINITY; vec_width];
            let mut t_slice = [std::f32::INFINITY; vec_width];
            for vec_j in 0..vec_width {
                let j = row * vec_width + vec_j;
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
    vd.chunks_exact_mut(n)
        .zip(vt.chunks_exact_mut(n))
        .enumerate()
        .for_each(preprocess_row);

    // Everything is exactly as in v5, but with prefetch hints in the innermost loop
    const PREFETCH_LENGTH: usize = 20;
    let step_row = |(i, (r_row_block, vd_row)): (usize, (&mut [f32], &[__m256]))| {
        assert_eq!(vd_row.len(), n);
        // Create raw pointers for prefetching
        let vd_ptr = vd.as_ptr();
        let vt_ptr = vt.as_ptr();
        for (j, vt_row) in vt.chunks_exact(n).enumerate() {
            assert_eq!(vt_row.len(), n);
            let mut tmp = [simd::m256_infty(); vec_width];
            for (col, (&d0, &t0)) in vd_row.iter().zip(vt_row).enumerate() {
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
                simd::prefetch(vd_ptr, (n * i + col + PREFETCH_LENGTH) as isize);
                simd::prefetch(vt_ptr, (n * j + col + PREFETCH_LENGTH) as isize);
            }
            tmp[1] = simd::swap(tmp[1], 1);
            tmp[3] = simd::swap(tmp[3], 1);
            tmp[5] = simd::swap(tmp[5], 1);
            tmp[7] = simd::swap(tmp[7], 1);
            for block_i in 0..vec_width {
                for (block_j, r_row) in r_row_block.chunks_exact_mut(n).enumerate() {
                    assert_eq!(r_row.len(), n);
                    let res_i = block_j + i * vec_width;
                    let res_j = block_i + j * vec_width;
                    if res_i < n && res_j < n {
                        let v = tmp[block_j ^ block_i];
                        let vi = block_i as u8;
                        r_row[res_j] = simd::extract(v, vi);
                    }
                }
            }
        }
    };
    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(vec_width * n)
        .zip(vd.par_chunks(n))
        .enumerate()
        .for_each(step_row);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_exact_mut(vec_width * n)
        .zip(vd.chunks_exact(n))
        .enumerate()
        .for_each(step_row);
}


create_extern_c_wrapper!(step, _step);
