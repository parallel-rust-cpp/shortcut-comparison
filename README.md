# Performance comparison of parallel Rust and C++

This project compares the behaviour and performance of two solutions to a simple graph problem, called the shortcut problem.
The reference solution, written in C++, and a description of the shortcut problem can be found [here](http://ppc.cs.aalto.fi/ch2/).
The reference solution will be compared to a [Rust](https://github.com/rust-lang/rust) implementation, which is provided by this project.

This repository contains the benchmark program and source code of all `step`-function implementations.
A human-readable explanation of the Rust implementations can be found on [this page](https://parallel-rust-cpp.github.io/).


## Running the benchmarks

Run the whole pipeline with a smaller, debug benchmark size to check everything is working (should not take more than 15 minutes):
```bash
bash benchmark.bash --debug
```
If you want to run the same benchmarks as described [here](https://parallel-rust-cpp.github.io/), run without `--debug` (might take a few hours):
```bash
bash benchmark.bash
```
