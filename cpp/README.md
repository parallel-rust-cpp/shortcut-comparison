This C++ implementation is the reference solution to the shortcut problem, and has been copied, as unmodified as possible, from [here](http://ppc.cs.aalto.fi/ch2/).

All subdirectories prefixed with `v?_` contain incrementally improved versions of the initial baseline solution `v0`.

## Building

Create a directory for build output:
```
mkdir build
```

Generate makefiles and make project into the build directory:
```
cd build
cmake -G "Unix Makefiles" ../cpp
make
```

Binaries for benchmarking each version of the `step` function are in the `target` directory.

### Running

Example: Run the `v1` version benchmark for 10 iterations, with input of size 100x100, consisting of random floating point numbers uniformly distributed in range `[0, 1]`:
```
./target/v1_linear_reading 100 10
```

Run the same benchmark using only one thread:
```
OMP_NUM_THREADS=1 ./target/v1_linear_reading 100 10
```
