use std::arch::x86_64::*; // Intel SIMD intrinsic mappings

#[allow(non_camel_case_types)]
pub type f32x8 = __m256;
#[allow(non_upper_case_globals)]
pub const f32x8_LENGTH: usize = 8;

/// Return a 256-bit vector containing 8 infinity values of f32
#[inline]
pub fn f32x8_infty() -> f32x8 {
    unsafe { _mm256_set1_ps(std::f32::INFINITY) }
}

#[inline]
pub fn add(v: f32x8, w: f32x8) -> f32x8 {
    unsafe { _mm256_add_ps(v, w) }
}

#[inline]
pub fn min(v: f32x8, w: f32x8) -> f32x8 {
    unsafe { _mm256_min_ps(v, w) }
}

/// Extract the lowest 32 bits of a 256-bit vector as a float
#[inline]
pub fn lowestf32(v: f32x8) -> f32 {
    unsafe { _mm256_cvtss_f32(v) }
}

/// Create a 256-bit vector from a f32 slice of length 8
#[inline]
pub fn from_slice(s: &[f32]) -> f32x8 {
    assert_eq!(s.len(), f32x8_LENGTH);
    unsafe { _mm256_set_ps(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]) }
}

/// Permute 1, 2, or 4 wide chunks with adjacent chunks
/// E.g.
/// swap([0, 1, 2, 3, 4, 5, 6, 7], 1) == [1, 0, 3, 2, 5, 4, 7, 6]
/// swap([0, 1, 2, 3, 4, 5, 6, 7], 2) == [2, 3, 0, 1, 6, 7, 4, 5]
/// swap([0, 1, 2, 3, 4, 5, 6, 7], 4) == [4, 5, 6, 7, 0, 1, 2, 3]
///
/// To make sense of the 8-bit shuffle control, read it in binary from right to left
/// e.g. for width = 1, control is 10_11_00_01.
/// Reading from right to left in 2 bit chunks we get (1, 0, 3, 2) for the 1st 128-bit lane,
/// and (5, 4, 7, 6) for the 2nd 128-bit lane.
///
#[inline]
pub fn swap(v: f32x8, width: i8) -> f32x8 {
    match width {
        1 => unsafe { _mm256_shuffle_ps(v, v, 0b_10_11_00_01) },
        2 => unsafe { _mm256_shuffle_ps(v, v, 0b_01_00_11_10) },
        4 => unsafe { _mm256_permute2f128_ps(v, v, 1) },
        _ => panic!("Invalid shuffle control for 256-bit vector, must be 1, 2, or 4"),
    }
}

#[inline]
pub fn prefetch(p: *const f32x8, length: isize) {
    unsafe { _mm_prefetch((p as *const i8).offset(length), _MM_HINT_T0) }
}

/// Use an index to extract a single f32 from a 256-bit vector of single precision floats
#[inline]
pub fn extract(v: f32x8, i: u8) -> f32 {
    // Create a permutation of v such that the 32 lowest bits correspond to the ith 32-bit chunk of v
    let permuted = match i {
        7 => v,
        6 => swap(v, 1),
        5 => swap(v, 2),
        4 => swap(swap(v, 1), 2),
        3 => swap(v, 4),
        2 => swap(swap(v, 1), 4),
        1 => swap(swap(v, 2), 4),
        0 => swap(swap(swap(v, 1), 2), 4),
        _ => panic!("Invalid index for vector containing 8 elements"),
    };
    // Extract the lowest 32 bits
    lowestf32(permuted)
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
#[inline]
pub fn horizontal_min(v: f32x8) -> f32 {
    let min_1 = min(swap(v, 1), v);
    let min_2 = min(swap(min_1, 2), min_1);
    let min_4 = min(swap(min_2, 4), min_2);
    // All elements of min_4 are the minimum of v, extract the lowest 32 bits
    lowestf32(min_4)
}

/// Print the contents of a 256-bit vector
#[inline]
pub fn print_vec(v: f32x8, padding: usize, precision: usize) {
    for i in 0..f32x8_LENGTH as u8 {
        let x: f32 = extract(v, i);
        print!("{:padding$.precision$} ", x, padding=padding, precision=precision);
    }
}

/// Assert that a f32x8 is properly aligned
#[inline(always)]
pub fn assert_aligned(v: &f32x8) {
    assert_eq!((v as *const f32x8).align_offset(std::mem::align_of::<f32x8>()), 0);
}
