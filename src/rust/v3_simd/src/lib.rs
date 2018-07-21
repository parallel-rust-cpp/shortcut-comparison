use std::arch::x86_64::*; // Intel SIMD intrinsic mappings

extern crate rayon;
use rayon::prelude::*; // par_chunks_mut

#[allow(non_upper_case_globals)]
const m256_length: usize = 8;

/// Return a 256-bit vector containing 8 infinity values
fn f8infty() -> __m256 {
    unsafe { _mm256_set1_ps(std::f32::INFINITY) }
}


/// Create a 256-bit vector from a f32 slice of length 8
fn f32_slice_to_m256(row: &[f32]) -> __m256 {
    unsafe { _mm256_set_ps(row[0], row[1], row[2], row[3],
                           row[4], row[5], row[6], row[7]) }
}


/// Permute 1, 2, or 4 element ranges with their neighbors.
/// E.g.
/// swap([0, 1, 2, 3, 4, 5, 6, 7 ], 1) == [1, 0, 3, 2, 5, 4, 7, 6]
/// swap([0, 1, 2, 3, 4, 5, 6, 7 ], 2) == [2, 3, 0, 1, 6, 7, 4, 5]
/// swap([0, 1, 2, 3, 4, 5, 6, 7 ], 4) == [4, 5, 6, 7, 0, 1, 2, 3]
fn swap(v: __m256, control: i8) -> __m256 {
    // Read the shuffle control from right to left
    // e.g. for control = 1: (1, 0, 3, 2) (and (5, 4, 7, 6) for the 2nd 128-bit lane)
    match control {
        1 => unsafe { _mm256_shuffle_ps(v, v, 0b_10_11_00_01) },
        2 => unsafe { _mm256_shuffle_ps(v, v, 0b_01_00_11_10) },
        4 => unsafe { _mm256_permute2f128_ps(v, v, 1) },
        _ => panic!("Invalid shuffle control for 256-bit vector, must be 1, 2, or 4"),
    }
}


/// Return the smallest element from a 256-bit float vector
/// v              = [0, 1, 2, 3, 4, 5, 6, 7]
/// swap(v, 1)     = [1, 0, 3, 2, 5, 4, 7, 6]
/// min_1          = [0, 0, 2, 2, 4, 4, 6, 6]
/// swap(min_1, 2) = [2, 2, 0, 0, 6, 6, 4, 4]
/// min_2          = [0, 0, 0, 0, 4, 4, 4, 4]
/// swap(min_2, 4) = [4, 4, 4, 4, 0, 0, 0, 0]
/// min_4          = [0, 0, 0, 0, 0, 0, 0, 0]
fn horizontal_min(v: __m256) -> f32 {
    unsafe {
        let min_1 = _mm256_min_ps(swap(v, 1), v);
        let min_2 = _mm256_min_ps(swap(min_1, 2), min_1);
        let min_4 = _mm256_min_ps(swap(min_2, 4), min_2);
        // All elements of min_4 are the minimum of v, extract the lowest 32 bits
        _mm256_cvtss_f32(min_4)
    }
}

fn m256_add(v: __m256, w: __m256) -> __m256 {
    unsafe { _mm256_add_ps(v, w) }
}


fn m256_min(v: __m256, w: __m256) -> __m256 {
    unsafe { _mm256_min_ps(v, w) }
}


fn _step(r: &mut [f32], d: &[f32], n: usize) {
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
            vd.push(f32_slice_to_m256(&d_slice));
            vt.push(f32_slice_to_m256(&t_slice));
        }
    }

    // Partition the result slice into n rows, and compute result for each row in parallel
    r.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut tmp = f8infty();
            for col in 0..vecs_per_row {
                let x = vd[vecs_per_row * i + col];
                let y = vt[vecs_per_row * j + col];
                let z = m256_add(x, y);
                tmp = m256_min(tmp, z);
            }
            row[j] = horizontal_min(tmp);
        }
    });
}


#[no_mangle]
pub extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    let d = unsafe { std::slice::from_raw_parts(d_raw, n * n) };
    let mut r = unsafe { std::slice::from_raw_parts_mut(r_raw, n * n) };
    _step(&mut r, d, n);
}
