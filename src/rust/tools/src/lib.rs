use std::time;
use std::vec;

pub struct Stopwatch {
    instants: vec::Vec<time::Instant>,
}

impl Stopwatch {
    pub fn new() -> Stopwatch {
        Stopwatch { instants: vec::Vec::new() }
    }
    pub fn click(&mut self) {
        self.instants.push(time::Instant::now());
    }
    pub fn dump(&self) {
        self.instants.as_slice().windows(2).enumerate().for_each(|(i, w)| {
            let (earlier, later) = (w[0], w[1]);
            let elapsed = later.duration_since(earlier);
            println!("{}-{} : {} ms", i, i+1, (1000 * elapsed.as_secs() + elapsed.subsec_millis() as u64));
        });
    }
}
