# Rust and C++, a performance comparison

This project compares the behaviour and performance of two solutions to a simple graph problem, called the shortcut problem.
The reference solution, written in C++, and a description of the shortcut problem can be found [here](http://ppc.cs.aalto.fi/ch2/).
The reference solution will be compared to a [Rust](https://github.com/rust-lang/rust) implementation, which is provided by this project.

If you are already familiar with the approach presented by the reference [C++ solution](http://ppc.cs.aalto.fi/ch2/), a more thorough explanation of the provided Rust implementation can be found [here](/docs/intro.md).

## The `step` function

The reference solution provides 8 versions of the `step`-function, each containing an incremental improvement, built on top of the previous implementation.

Version | Description
--- | ---
`v0_baseline` | Straightforward solution with 3 for-loops and no preprocessing
`v1_linear_reading` | Copy input and create its transpose, enabling a linear memory access pattern
`v2_instr_level_parallelism` | Break instruction dependency chains for improved CPU instruction throughput
`v3_simd` | Use vector registers and SIMD instructions explicitly for reducing the amount of required CPU instructions
`v4_register_reuse` | Read vectors in blocks of 6 and do 9+9 arithmetic operations for an improved operations per memory access ratio
`v5_more_register_reuse` | Reorder the vector representation of the input from horizontal to vertical. Read the vertical vector data in pairs and do 8+8 arithmetic operations, improving the ratio of operations per memory access even further
`v6_prefetching` | Add software prefetching hints for improving memory throughput
`v7_cache_reuse` | (multi-core not implemented) Add [Z-order curve](https://en.wikipedia.org/wiki/Z-order_curve) memory access pattern for improving cache reuse


## Requirements

This project provides 3 scripts for building, benchmarking and testing the project.
These scripts assume the following executables are available on your path:

* python3
* g++
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

If you prefer to run compiled things in Docker containers, a pre-built image is available [here](https://hub.docker.com/r/matiaslindgren/shortcut-comparison/).
The image has been built using the same Dockerfile found in this repo.
In order to use the `perf` tool from within the container, you need to run the container with elevated [privileges](https://docs.docker.com/engine/reference/run/#runtime-privilege-and-linux-capabilities).
Also [relevant](https://stackoverflow.com/questions/44745987/use-perf-inside-a-docker-container-without-privileged).

Download the image, create a temporary container, and run it with [CAP_SYS_ADMIN](https://linux.die.net/man/7/capabilities) privileges:
```
docker run --rm -it --cap-add SYS_ADMIN matiaslindgren/shortcut-comparison
```

You should now be running an interactive shell inside the container, which should have all dependencies needed to run the commands shown below.

## Building

Run the provided build script, (use `--verbose` because errors are not yet caught properly):
```
./build.py --verbose
```
Assuming all dependencies have been installed, this will create an out of source build into the directory `./build`.

All executables for testing each version of the `step` function are in the `build/bin` directory.

## Testing

Test all implementations against the C++ v0 baseline implementation:
```
./test.py
```

## Benchmarking

See:
```
./bench.py --help
```

### Everything at once

Examples:

Run all benchmarks with `perf stat`, using one thread and 5 smallest sizes for input:
```
./bench.py -m 5
```

Run all benchmarks, will take considerably more time than the previous command:
```
./bench.py
```

All benchmark sizes, 4 threads and only the linear reading implementations:
```
./bench.py -t 4 -i v1
```

Inputs of size 2500 and 4000, 4 threads and only the SIMD implementations:
```
./bench.py -n 7 -m 9 -t 4 -i v1
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

### Findings

* Linking Rust static libraries into benchmarking tools compiled from C++ incurs significant overhead in the form of excessive amounts of CPU cycles. Maybe the benchmarking code needs to also be written in Rust to make sure there is no weirdness from FFI.
* The Rust compiler seems to be rather lenient what comes to automatically inlining cross-crate function calls. By making the hottest functions in the `tools::simd` module eligible for inlining (by adding the `#[inline]` attribute), the amount of CPU cycles during benchmarking was reduced by a factor of 10.
* Prefetching does hardly help in Rust.
Given the high ratio of instructions per cycles during execution of the Rust implementations, it seems that the Rust compiler is able to generate instructions that saturate all CPU execution ports rather well.
Therefore, no ports are left for executing the prefetch instructions, and using them actually makes the running times even worse.
* [Prefer an `if else` expression over `f32::min`](/docs/f32_min_method.md)
