This project compares the behaviour and performance of two solutions to a simple graph problem, called the shortcut problem.
The reference solution, written in C++, and a description of the shortcut problem can be found [here](http://ppc.cs.aalto.fi/ch2/).
The reference solution will be compared to a [Rust](https://github.com/rust-lang/rust) implementation, which is provided by this project.

## Requirements

This project provides 3 scripts for building, benchmarking and testing the project.
These scripts assume the following executables are available on your path:

* python3
* gcc
* make
* cmake
* perf
* cargo
* rustc

You can install and configure both the Rust compiler `rustc` and its package management tool `cargo` by using [rustup](https://github.com/rust-lang-nursery/rustup.rs).

If you use the rustup script, change the default toolchain to `nightly` and continue installation.

If you installed the `rustup` binary:
```
rustup install nightly
rustup default nightly
rustup update
```

## Building

Run the provided build script (use `--verbose` because errors are not yet caught properly):
```
./build.py --verbose
```
Assuming all dependencies have been installed, this will create an out of source build into the directory `./build`.

All executables for testing each version of the `step` function are in the `build/bin` directory.

## Testing

Make sure all implementations of the step function yield results equal to the output of the C++ v0 baseline implementation:
```
./test.py build/bin
```

## Running

### Everything at once

Run all benchmarks with `perf stat`, using one thread and 5 smallest sizes for input:
```
./bench.py build/bin -n 5
```

All benchmark sizes, 4 threads and only the linear reading implementations:
```
./bench.py build/bin -t 4 -i v1
```

### Single benchmark

Example: Run a benchmark for the C++ step version that implements linear reading.
Benchmark for 10 iterations, with input of size 1000x1000, consisting of random floating point numbers uniformly distributed in range `[0, 1]`:
```
./build/bin/v1_linear_reading_cpp benchmark 1000 10
```

Run the same benchmark using only one thread:
```
OMP_NUM_THREADS=1 ./build/bin/v1_linear_reading_cpp benchmark 1000 10
```

Example: Test that the baseline Rust implementation is correct:
```
./build/bin/v0_baseline_rust test 500 10
```

Example: Run the Rust step version that implements instruction level parallelism.
Benchmark for 2 iterations, with random input of size 4000x4000, and using 8 threads:
```
RAYON_NUM_THREADS=8 ./build/bin/v2_instr_level_parallelism_rust benchmark 4000 2
```
