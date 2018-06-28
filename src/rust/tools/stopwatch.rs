use std::time;

struct Stopwatch {
    instants: vec::Vec<Instant> = vec::Vec::new();
}

impl Stopwatch {
    fn measure(&self) {
        instants.push(Instant::now());
    }
    fn dump(&self) {
        self.instants.as_slice().windows(2).enumerate().for_each(|(i, w)| {
            let (earlier, later) = (w[0], w[1]);
            let elapsed = later.duration_since(earlier).as_millis();
            println!("{}. : {} ms", i, elapsed);
        });
    }
}
