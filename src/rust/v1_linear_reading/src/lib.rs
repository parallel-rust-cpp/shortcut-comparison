use std::vec;

#[no_mangle]
pub extern "C" fn step(r: *mut f32, d: *const f32, n: usize) {
    let mut t: vec::Vec<f32> = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[n*j + i] = unsafe { *d.offset((n*i + j) as isize) };
        }
    }

    for i in 0..n {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = unsafe { *d.offset((n*i + k) as isize) };
                let y = t[n*j + k];
                let z = x + y;
                v = v.min(z);
            }
            unsafe {
                *r.offset((n*i + j) as isize) = v;
            }
        }
    }
}
