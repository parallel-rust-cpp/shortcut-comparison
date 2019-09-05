    // ANCHOR: step_row
    let step_row = |(i, r_row): (usize, &mut [f32])| {
        // Get a view of row i of d as a subslice
        let d_row = &d[n*i..n*(i+1)];
        for (j, res) in r_row.iter_mut().enumerate() {
            // Same for row j in t
            let t_row = &t[n*j..n*(j+1)];
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = d_row[k];
                let y = t_row[k];
                let z = x + y;
                v = if v < z { v } else { z };
            }
            *res = v;
        }
    };
    // ANCHOR_END: step_row
    // ANCHOR: step_row_inner_iter
    for (&x, &y) in d_row.iter().zip(t_row.iter()) {
        let z = x + y;
        v = if v < z { v } else { z };
    }
    // ANCHOR_END: step_row_inner_iter
