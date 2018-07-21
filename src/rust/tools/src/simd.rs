use std::arch::x86_64::*; // Intel SIMD intrinsic mappings
use std::f32;

/// Amount of f32 elements in a 256-bit vector, aka __m256
pub const M256_LENGTH: usize = 8;

/// Return a 256-bit vector containing 8 infinity values
pub fn f8infty() -> __m256 {
    unsafe { _mm256_set1_ps(f32::INFINITY) }
}

/// Create a 256-bit vector from a f32 slice of length 8
pub fn from_slice(row: &[f32]) -> __m256 {
    unsafe { _mm256_set_ps(row[0], row[1], row[2], row[3],
                           row[4], row[5], row[6], row[7]) }
}

/// Permute 1, 2, or 4 element ranges with their neighbors.
/// E.g.
/// swap([0, 1, 2, 3, 4, 5, 6, 7], 1) == [1, 0, 3, 2, 5, 4, 7, 6]
/// swap([0, 1, 2, 3, 4, 5, 6, 7], 2) == [2, 3, 0, 1, 6, 7, 4, 5]
/// swap([0, 1, 2, 3, 4, 5, 6, 7], 4) == [4, 5, 6, 7, 0, 1, 2, 3]
///
/// To make sense of the 8-bit shuffle control, read it in binary from right to left
/// e.g. for i = 1, control is 10110001.
/// Reading from right to left in 2 bit chunks we get (1, 0, 3, 2),
/// and (5, 4, 7, 6) for the 2nd 128-bit lane.
///
pub fn swap(v: __m256, width: i8) -> __m256 {
    match width {
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
///
pub fn horizontal_min(v: __m256) -> f32 {
    unsafe {
        let min_1 = _mm256_min_ps(swap(v, 1), v);
        let min_2 = _mm256_min_ps(swap(min_1, 2), min_1);
        let min_4 = _mm256_min_ps(swap(min_2, 4), min_2);
        // All elements of min_4 are the minimum of v, extract the lowest 32 bits
        _mm256_cvtss_f32(min_4)
    }
}

pub fn add(v: __m256, w: __m256) -> __m256 {
    unsafe { _mm256_add_ps(v, w) }
}

pub fn min(v: __m256, w: __m256) -> __m256 {
    unsafe { _mm256_min_ps(v, w) }
}
