use std::env;
use std::process;
use std::vec;
use std::time;
use std::str::FromStr;

extern crate rand;
use rand::Rng;

extern crate v0_baseline;

fn next_float() -> f32 {
    rand::thread_rng().gen_range(0.0, 1.0)
}

fn benchmark(n: usize) {
    let mut result: vec::Vec<f32> = vec![0.0; n * n];
    let mut data: vec::Vec<f32> = vec![0.0; n * n];
    for i in 0..n * n {
        data[i] = next_float();
    }
    let data = data;

    let before = time::Instant::now();
    v0_baseline::step::step(&mut result, &data, n);
    let after = time::Instant::now();

    let duration = after.duration_since(before);
    println!("{}.{}", duration.as_secs(), duration.subsec_micros());
}

fn parse_int(s: &String) -> i32 {
    i32::from_str(s).unwrap()
}

fn main() {
    let args: vec::Vec<String> = env::args().collect();
    let (n, iterations) = match args.len() {
        3 => (parse_int(&args[1]), parse_int(&args[2])),
        2 => (parse_int(&args[1]), 1),
        _ => {
            eprintln!("benchmark usage: {} N [ITERATIONS]", &args[0]);
            process::exit(1);
        }
    };
    println!("benchmarking {0} with input containing {1}*{1} elements", &args[0], n);
    for _ in 0..iterations {
        benchmark(n as usize);
    }
}
