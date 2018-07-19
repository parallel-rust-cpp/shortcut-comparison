use std::vec;

fn _step(r: &mut [f32], d: &[f32], n: usize) {
    const BLOCK_SIZE: usize = 4;
    let block_count = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let n_padded = block_count * BLOCK_SIZE;

    // Transpose of d
    let mut t: vec::Vec<f32> = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[n*j + i] = d[n*i + j];
        }
    }

    // Partition the result slice into n rows, and compute result for each row in parallel
    r.chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut block = vec![std::f32::INFINITY; BLOCK_SIZE];
            for b in 0..block_count {
                for k in 0..BLOCK_SIZE {
                    let x = d[n_padded*i + b*BLOCK_SIZE + k];
                    let y = t[n_padded*j + b*BLOCK_SIZE + k];
                    let z = x + y;
                    block[k] = block[k].min(z);
                }
            }
            let mut res = std::f32::INFINITY;
            for x in block.iter() {
                res = res.min(*x);
            }
            row[j] = res;
        }
    });
}

#[no_mangle]
pub extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    let d = unsafe { std::slice::from_raw_parts(d_raw, n * n) };
    let mut r = unsafe { std::slice::from_raw_parts_mut(r_raw, n * n) };
    _step(&mut r, d, n);
}
