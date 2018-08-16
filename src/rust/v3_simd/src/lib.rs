#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*; // Parallel chunks iterator

extern crate tools;
use tools::simd; // Custom SIMD helpers


#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    #[allow(non_upper_case_globals)]
    const m256_length: usize = simd::M256_LENGTH;
    let vecs_per_row = (n + m256_length - 1) / m256_length;

    // Pack d and its transpose into m256 vectors, each containing 8 f32::INFINITYs
    let mut vt = std::vec::Vec::with_capacity(n * vecs_per_row);
    let mut vd = std::vec::Vec::with_capacity(n * vecs_per_row);
    for row in 0..n {
        for col in 0..vecs_per_row {
            // Build 8 element arrays for vd and vt, with infinity padding
            let mut d_slice = [std::f32::INFINITY; m256_length];
            let mut t_slice = [std::f32::INFINITY; m256_length];
            for vec_i in 0..m256_length {
                let i = col * m256_length + vec_i;
                if i < n {
                    d_slice[vec_i] = d[n * row + i];
                    t_slice[vec_i] = d[n * i + row];
                }
            }
            // Convert arrays to 256-bit vectors and assign to vector containers
            vd.push(simd::from_slice(&d_slice));
            vt.push(simd::from_slice(&t_slice));
        }
    }

    #[cfg(not(feature = "no-multi-thread"))]
    r.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut tmp = simd::m256_infty();
            for col in 0..vecs_per_row {
                let x = vd[vecs_per_row * i + col];
                let y = vt[vecs_per_row * j + col];
                let z = simd::add(x, y);
                tmp = simd::min(tmp, z);
            }
            row[j] = simd::horizontal_min(tmp);
        }
    });
    #[cfg(feature = "no-multi-thread")]
    for i in 0..n {
        for j in 0..n {
            let mut tmp = simd::m256_infty();
            for col in 0..vecs_per_row {
                let x = vd[vecs_per_row * i + col];
                let y = vt[vecs_per_row * j + col];
                let z = simd::add(x, y);
                tmp = simd::min(tmp, z);
            }
            r[i*n + j] = simd::horizontal_min(tmp);
        }
    }
}


#[no_mangle]
pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    let d = std::slice::from_raw_parts(d_raw, n * n);
    let mut r = std::slice::from_raw_parts_mut(r_raw, n * n);
    _step(&mut r, d, n);
}
