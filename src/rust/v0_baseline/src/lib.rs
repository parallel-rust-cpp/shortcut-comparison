fn step_slices(r: &mut [f32], d: &[f32], n: usize) {
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

fn step_raw(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = unsafe { *d_raw.offset((n*i + k) as isize) };
                let y = unsafe { *d_raw.offset((n*k + j) as isize) };
                let z = x + y;
                v = v.min(z);
            }
            unsafe {
                *r_raw.offset((n*i + j) as isize) = v
            }
        }
    }
}

// Do not mangle function name to enable library linking
#[no_mangle]
// C interface that accepts raw C pointers as arguments
pub extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    // Wrap given raw pointers into a Rust slices
    // Dereferencing raw pointers can only be done inside 'unsafe' sections
    //let d = unsafe { std::slice::from_raw_parts(d_raw, n * n) };
    //let mut r = unsafe { std::slice::from_raw_parts_mut(r_raw, n * n) };

    //step_slices(&mut r, d, n);
    step_raw(r_raw, d_raw, n);
}
