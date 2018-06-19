use std::vec::Vec;
use std::f32; // for infinity constant only

fn step(r: &mut Vec<f32>, d: &Vec<f32>, n: usize) {
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

fn main() {
    const n: usize = 3;
    let d: Vec<f32> = vec![
        0., 8., 2.,
        1., 0., 9.,
        4., 5., 0.
    ];
    let mut r: Vec<f32> = vec![0.; n*n];
    step(&mut r, &d, n);
    for i in 0..n {
        for j in 0..n {
            print!("{} ", r[i*n + j]);
        }
        println!();
    }
}
