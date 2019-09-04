#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    let mut t = vec![0.0; n * n];
    // ANCHOR: step_row
    // Function: for some row i in d (d_row) and all rows t (t_rows), compute n results into a row in r (r_row)
    let step_row = |(i, r_row): (usize, &mut [f32])| {
        for (j, res) in r_row.iter_mut().enumerate() {
            let mut v = std::f32::INFINITY;
            // ANCHOR: step_row_inner
            for k in 0..n {
                let x = d[n*i + k];
                let y = t[n*j + k];
                let z = x + y;
                v = v.min(z);
            }
            // ANCHOR_END: step_row_inner
            *res = v;
        }
    };
    // Partition r and d into slices, each containing a single row of r and d,
    // and apply the function on the row pairs
    r.par_chunks_mut(n)
        .enumerate()
        .for_each(step_row);
    // ANCHOR_END: step_row
}

    // ANCHOR: step_row_inner_no_nan
    for k in 0..n {
        let x = d[n*i + k];
        let y = t[n*j + k];
        let z = x + y;
        v = if v < z { v } else { z }
    }
    // ANCHOR_END: step_row_inner_no_nan
