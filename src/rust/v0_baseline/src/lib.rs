// Do not mangle function name to enable library linking
#[no_mangle]
pub extern "C" fn step(r: *mut f32, d: *const f32, n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                // Raw pointers may be dereferenced only in 'unsafe' sections
                let x = unsafe { *d.offset((n*i + k) as isize) };
                let y = unsafe { *d.offset((n*k + j) as isize) };
                let z = x + y;
                v = v.min(z);
            }
            unsafe {
                *r.offset((n*i + j) as isize) = v;
            }
        }
    }
}
