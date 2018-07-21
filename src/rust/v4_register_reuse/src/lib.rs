extern crate rayon;
use rayon::prelude::*; // Parallel chunks iterator

extern crate tools;
use tools::simd; // Custom SIMD helpers


fn _step(r: &mut [f32], d: &[f32], n: usize) {
    #[allow(non_upper_case_globals)]
    const m256_length: usize = simd::M256_LENGTH;
    let vecs_per_row = (n + m256_length - 1) / m256_length;

    // Cache blocking with 3x3 blocks containing 9 m256 vectors
    #[allow(non_upper_case_globals)]
    const blocksize: usize = 3;
    let blocks_per_n = (n + blocksize - 1) / blocksize;
    let padded_n = blocksize * blocks_per_n;

    // Pack d and its transpose into m256 vectors, each containing 8 f32::INFINITYs
    let mut vt = std::vec::Vec::with_capacity(padded_n * vecs_per_row);
    let mut vd = std::vec::Vec::with_capacity(padded_n * vecs_per_row);
    for row in 0..n {
        for col in 0..vecs_per_row {
            // Build 8 element arrays for vd and vt, with infinity padding
            let mut d_slice = [std::f32::INFINITY; m256_length];
            let mut t_slice = [std::f32::INFINITY; m256_length];
            for vec_i in 0..m256_length {
                let i = col * m256_length + vec_i;
                if i < n {
                    d_slice[vec_i] = d[n * row + i];
                    t_slice[vec_i] = d[n * i + row];
                }
            }
            // Convert arrays to 256-bit vectors and assign to vector containers
            vd.push(simd::from_slice(&d_slice));
            vt.push(simd::from_slice(&t_slice));
        }
    }
    // Fill padded rows with infinity
    let inf_slice = [std::f32::INFINITY; m256_length];
    for _ in n..padded_n {
        for _ in 0..vecs_per_row {
            vd.push(simd::from_slice(&inf_slice));
            vt.push(simd::from_slice(&inf_slice));
        }
    }

    // Partition the result slice into rows of blocks, and compute result for each row in parallel
    r.par_chunks_mut(blocks_per_n).enumerate().for_each(|(i, block_row)| {
        for j in 0..blocks_per_n {
            // m256 vector block containing blocksize ** 2 vectors
            let mut tmp = [[simd::f8infty(); blocksize]; blocksize];
            // Compute blocksize ** 2 values in one iteration
            for col in 0..vecs_per_row {
                let x0 = vd[vecs_per_row * i * blocksize + col];
                let x1 = vd[vecs_per_row * (i * blocksize + 1) + col];
                let x2 = vd[vecs_per_row * (i * blocksize + 2) + col];
                let y0 = vt[vecs_per_row * j * blocksize + col];
                let y1 = vt[vecs_per_row * (j * blocksize + 1) + col];
                let y2 = vt[vecs_per_row * (j * blocksize + 2) + col];
                tmp[0][0] = simd::min(tmp[0][0], simd::add(x0, y0));
                tmp[0][1] = simd::min(tmp[0][1], simd::add(x0, y1));
                tmp[0][2] = simd::min(tmp[0][2], simd::add(x0, y2));
                tmp[1][0] = simd::min(tmp[1][0], simd::add(x1, y0));
                tmp[1][1] = simd::min(tmp[1][1], simd::add(x1, y1));
                tmp[1][2] = simd::min(tmp[1][2], simd::add(x1, y2));
                tmp[2][0] = simd::min(tmp[2][0], simd::add(x2, y0));
                tmp[2][1] = simd::min(tmp[2][1], simd::add(x2, y1));
                tmp[2][2] = simd::min(tmp[2][2], simd::add(x2, y2));
            }
            // Reduce all block vectors to assign blocksize ** 2 of final results
            for block_i in 0..blocksize {
                for block_j in 0..blocksize {
                    let res_i = i * blocksize + block_i;
                    let res_j = j * blocksize + block_j;
                    if res_i < n && res_j < n {
                        let v = tmp[block_i][block_j];
                        block_row[res_j] = simd::horizontal_min(v);
                    }
                }
            }
        }
    });
}


#[no_mangle]
pub extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: usize) {
    let d = unsafe { std::slice::from_raw_parts(d_raw, n * n) };
    let mut r = unsafe { std::slice::from_raw_parts_mut(r_raw, n * n) };
    _step(&mut r, d, n);
}
