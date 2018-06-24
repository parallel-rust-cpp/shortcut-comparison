// Do not mangle function name to enable library linking
#[no_mangle]
pub extern "C" fn step(r: *mut f32, d: *const f32, n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                // Raw pointers may be dereferenced only in 'unsafe' sections
                let z = unsafe {
                    let x = *d.offset((n*i + k) as isize);
                    let y = *d.offset((n*k + j) as isize);
                    x + y
                };
                v = v.min(z);
            }
            unsafe {
                *r.offset((n*i + j) as isize) = v;
            }
        }
    }
}
