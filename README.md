This project compares the behaviour and performance of two solutions to a simple graph problem, called the shortcut problem.
The reference solution, written in C++, and a description of the shortcut problem can be found [here](http://ppc.cs.aalto.fi/ch2/).
The reference solution will be compared to a [Rust](https://github.com/rust-lang/rust) implementation, which is provided by this project.

## Requirements

* python3
* cargo
* rustc
* cmake
* gcc

## Building

Run the provided build script (use `--verbose` because errors are not yet caught properly):
```
./build.py --verbose
```
Assuming all dependencies have been installed, this will create an out of source build into the directory `./build`.

All executables for testing each version of the `step` function are in the `build/bin` directory.

## Running

### Everything at once

Run all benchmarks with `perf stat`, using one thread and 5 smallest sizes for input:
```
./bench.py build/bin -n 5
```

All benchmark sizes and 4 threads:
```
./bench.py build/bin -t 4
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

Example: Run the Rust step version that implements linear reading,
Benchmark for 1 iteration, with random input of size 4000x4000, and using 2 threads:
```
RAYON_NUM_THREADS=2 ./build/bin/v1_linear_reading_rust benchmark 4000
```
