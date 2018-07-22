#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = d[n*i + k];
                let y = d[n*k + j];
                let z = x + y;
                v = v.min(z);
            }
            r[n*i + j] = v;
        }
    }
}

/// C interface that accepts raw C pointers as arguments
// Do not mangle function name to make library linking easier
#[no_mangle]
// Raw pointers can be dereferenced only inside 'unsafe' sections, hence function is marked as unsafe
pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    // Wrap raw pointers into 'not unsafe' Rust slices
    let d = std::slice::from_raw_parts(d_raw, n * n);
    let mut r = std::slice::from_raw_parts_mut(r_raw, n * n);
    // Evaluate Rust implementation of the step-function
    _step(&mut r, d, n);
}
