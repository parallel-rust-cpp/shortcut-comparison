// Destructure iterator into tuple
use itertools::Itertools;

#[inline]
fn _step(r: &mut [f32], d: &[f32], n: usize) {
    let step_row = |(r_row, vd_row): (&mut [f32], &[f32])| {
        let vt_rows = vt.chunks_exact(n_padded);
        for (res, vt_row) in r_row.iter_mut().zip(vt_rows) {
            // ANCHOR: inner_loop
            let vd_blocks = vd_row.chunks_exact(BLOCK_SIZE);
            let vt_blocks = vt_row.chunks_exact(BLOCK_SIZE);
            // Encourage the compiler to use different registers for each f32 value
            let mut tmp0 = std::f32::INFINITY;
            let mut tmp1 = std::f32::INFINITY;
            let mut tmp2 = std::f32::INFINITY;
            let mut tmp3 = std::f32::INFINITY;
            for (vd_block, vt_block) in vd_blocks.zip(vt_blocks) {
                let (x0, x1, x2, x3) = vd_block.iter().next_tuple().unwrap();
                let (y0, y1, y2, y3) = vt_block.iter().next_tuple().unwrap();
                tmp0 = min(tmp0, x0 + y0);
                tmp1 = min(tmp1, x1 + y1);
                tmp2 = min(tmp2, x2 + y2);
                tmp3 = min(tmp3, x3 + y3);
            }
            *res = min(tmp0, min(tmp1, min(tmp2, tmp3)));
            // ANCHOR_END: inner_loop
        }
    };
}
