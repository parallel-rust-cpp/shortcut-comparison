Rust port of the [reference solution](/cpp).

All subdirectories prefixed with `v?_` contain crates with incrementally improved versions of initial baseline solution `v0`.
The crates should be compiled as static libraries, which are linked to the C++ benchmarking and testing applications.
There is no need to invoke `rustc` or `cargo` from here, use the [`build.py` script](/build.py).
