use tools::{create_extern_c_wrapper, simd, simd::f32x8};

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
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

    // Everything is exactly as in v5, but we add some prefetch instructions in the innermost loop
    let step_row_block = |(r_row_block, vd_row): (&mut [f32], &[f32x8])| {
        // Create const raw pointers for specifying addresses to prefetch
        let vd_row_ptr = vd_row.as_ptr();
        const PREFETCH_LENGTH: usize = 20;
        for (j, vt_row) in vt.chunks_exact(n).enumerate() {
            let vt_row_ptr = vt_row.as_ptr();
            let mut tmp = [simd::f32x8_infty(); simd::f32x8_LENGTH];
            for (col, (&d0, &t0)) in vd_row.iter().zip(vt_row).enumerate() {
                // Insert prefetch hints for fetching the cache line containing
                // the memory address 20 addresses ahead of the current column
                simd::prefetch(vd_row_ptr, (col + PREFETCH_LENGTH) as isize);
                simd::prefetch(vt_row_ptr, (col + PREFETCH_LENGTH) as isize);
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
            for i in (1..simd::f32x8_LENGTH).step_by(2) {
                tmp[i] = simd::swap(tmp[i], 1);
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
    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(simd::f32x8_LENGTH * n)
        .zip(vd.par_chunks(n))
        .for_each(step_row_block);
    #[cfg(feature = "no-multi-thread")]
    r.chunks_mut(simd::f32x8_LENGTH * n)
        .zip(vd.chunks(n))
        .for_each(step_row_block);
}


create_extern_c_wrapper!(step, _step);
