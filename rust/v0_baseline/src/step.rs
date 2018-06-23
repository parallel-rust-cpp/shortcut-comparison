pub mod step {
    use std::vec;
    use std::f32;

    // Vec<T> is a lightweight pointer wrapper with additional fields for capacity and length
    // TODO try with raw pointers in an 'unsafe' section
    pub fn step(r: &mut vec::Vec<f32>, d: &vec::Vec<f32>, n: usize) {
        for i in 0..n {
            for j in 0..n {
                let mut v = f32::INFINITY;
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
}
