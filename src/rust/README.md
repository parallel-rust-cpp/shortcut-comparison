Rust port of the [reference solution](/cpp).

All subdirectories prefixed with `v?_` contain incrementally improved versions of initial baseline solution `v0`.

## Building

```
cargo build --release
```

## Running

```
./target/release/shortcut 1000 10
```



#### TODO

* compile C++ benchmarking functionality into a static library and link to rust apps
