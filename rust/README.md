Rust port of the [reference solution](/cpp).

All subdirectories prefixed with `v?_` contain incrementally improved versions of initial baseline solution `v0`.

## Building

```
cd v0_baseline
cargo build --release
./target/release/shortcut
```


#### TODO

* compile C++ benchmarking functionality into a static library and link to rust apps
