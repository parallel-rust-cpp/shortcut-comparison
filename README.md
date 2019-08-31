# Performance comparison of parallel Rust and C++

This project compares the behaviour and performance of two solutions to a simple graph problem, called the shortcut problem.
The reference solution, written in C++, and a description of the shortcut problem can be found [here](http://ppc.cs.aalto.fi/ch2/).
The reference solution will be compared to a [Rust](https://github.com/rust-lang/rust) implementation, which is provided by this project.

This repository contains the benchmark program and source code of all `step`-function implementations.
A human-readable explanation of the Rust implementations can be found on [this page](https://parallel-rust-cpp.github.io/).

## Example benchmark on Intel Xeon E3-1230 v5

![Single thread benchmark results comparing the amount of single precision floating point instructions per second for 8 different implementations of the same program written in both Rust and C++.](reports/Xeon-E3-1230-v5/single_core/plot2.png "Single threaded performance")

![Multi thread benchmark results comparing the amount of single precision floating point instructions per second for 8 different implementations of the same program written in both Rust and C++.](reports/Xeon-E3-1230-v5/multi_core/plot2.png "Multi threaded performance")
