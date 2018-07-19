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

// Do not mangle function name to enable library linking
#[no_mangle]
// C interface that accepts raw C pointers as arguments
pub extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    // Wrap raw pointers given as parameter into Rust slices
    // Raw pointers can be dereferenced only inside 'unsafe' sections
    let d = unsafe { std::slice::from_raw_parts(d_raw, n * n) };
    let mut r = unsafe { std::slice::from_raw_parts_mut(r_raw, n * n) };
    _step(&mut r, d, n);
}
