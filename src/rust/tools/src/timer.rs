use std::time;
use std::vec;
extern crate core;

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

    pub fn report(&self) {
        self.instants.as_slice().windows(2).enumerate().for_each(|(i, w)| {
            let (earlier, later) = (w[0], w[1]);
            let elapsed = later.duration_since(earlier);
            let elapsed_ms = 1000 * elapsed.as_secs() + elapsed.subsec_millis() as u64;
            println!("{}-{} : {} ms", i, i+1, elapsed_ms);
        });
    }
}

// TODO generics or associated types or something

pub struct CycleCounter {
    instants: vec::Vec<i64>,
}

impl CycleCounter {
    pub fn new() -> CycleCounter {
        CycleCounter { instants: vec::Vec::new() }
    }

    pub fn click(&mut self) {
        let cpu_timestamp = unsafe { core::arch::x86_64::_rdtsc() };
        self.instants.push(cpu_timestamp);
    }

    pub fn report(&self) {
        self.instants.as_slice().windows(2).enumerate().for_each(|(i, w)| {
            let (earlier, later) = (w[0], w[1]);
            let cycles = later - earlier;
            println!("{}-{} : {} cycles", i, i+1, cycles);
        });
    }
}

/*
 * cargo:
 * [dependencies]
 * tools = { path = "../tools"}
 *
 * main.rs:
 * extern crate tools;
 * use tools::timer;
 *
 * in function:
 * let mut s = timer::CycleCounter::new();
 * s.click();
 * ... do many cycles ...
 * s.click();
 * s.report();
 *
 */
