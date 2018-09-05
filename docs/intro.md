# Solving the shortcut problem with Rust

This tutorial will walk through a Rust implementation of the [shortcut problem](http://ppc.cs.aalto.fi/ch2/).
We will be implementing 8 Rust libraries, all containing an incrementally improved version of a function called `step`, using a [C++ implementation](/src/cpp/) as a reference point to compare the performance of our Rust libraries.
All implementations will be parallelized with the help of an external Rust library called [Rayon](https://docs.rs/rayon/1.0.2/rayon/).
This tutorial assumes the reader feels comfortable both reading and writing C or C++, but is new to the Rust language.


## Rust

[Rust](https://www.rust-lang.org/en-US/) is a relatively young, systems programming language, which uses LLVM as a compiler backend to produce native code for various platforms.
Compared to C++, Rust enforces rather strict data ownership and reference borrowing semantics through static type checking and compile-time lifetime analysis of variables.
This is a fundamental design choice, built into the language.
According to the Rust documentation, these restrictions are meant to provide memory and thread safe programs by default, without the overhead from running a dynamic garbage collector.

Since this project will focus mostly on investigating the performance aspects of Rust, many significant language features are left uncovered.
The reader is encouraged to explore other tutorials in addition to this one in order to get a more holistic overview of the language.


## Calling Rust functions from C++

In order to avoid writing two separate programs for benchmarking, one for C++ and one for Rust, it would be nice if we could write Rust functions that the C++ linker understands.
We could then compile and link our Rust libraries to the benchmarking program, written in C++.

Consider the following C++ declaration of the `step` function:
```cpp
    extern "C" {
        void step(float*, const float*, int);
    }
```
We would like to implement this declaration in Rust so that when we compile the Rust implementation into a library and link it to our C++ application, we could call the Rust function `step` using regular C++ syntax.
Our function must accept raw pointers, because that's how the C++ application will pass allocated memory to us.
However, since Rust provides safer primitives, built on top of raw pointers, we would prefer to use these primitives and avoid handling raw pointers where possible.
Therefore, we implement the algorithm logic in a private function called `_step` (explained in next chapter) and expose its functionality through a public, thin C wrapper:
```rust
    #[no_mangle]
    pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: i32) {
        let d = std::slice::from_raw_parts(d_raw, (n * n) as usize);
        let mut r = std::slice::from_raw_parts_mut(r_raw, (n * n) as usize);
        _step(&mut r, d, n as usize);
    }
```

Let's break that down.

We instruct the compiler to retain the symbol name of the function (`step`) without name mangling, by using a compile-time [attribute](https://doc.rust-lang.org/reference/attributes.html#miscellaneous-attributes):
```rust
    #[no_mangle]
```

We declare an [`extern`](https://doc.rust-lang.org/book/second-edition/ch19-01-unsafe-rust.html#using-extern-functions-to-call-external-code) Rust function with public visibility, using the C-language application binary interface (ABI):
```rust
    pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: i32) {
```
The function takes as arguments two pointers to single precision floating point numbers, and one 32-bit integer.

The [`unsafe`](https://doc.rust-lang.org/book/second-edition/ch19-01-unsafe-rust.html#unsafe-rust) keyword is a rather powerful feature of Rust, which basically disables most (but not all) of the compile-time memory safety checks within the scope declared unsafe.
In exchange, we are allowed to e.g. dereference raw pointers and call platform specific intrinsics, which would create compile-time errors if done in a regular "not-unsafe" section.

Instead of declaring all our functions unsafe and keep passing around raw pointers like in C, we wrap the raw pointers into [slices](https://doc.rust-lang.org/std/primitive.slice.html).
Slices are Rust primitive types which provide a dynamically-sized view into a block of memory, basically a pointer with a length.

Here, we construct an immutable slice of length `n * n`, starting at the address pointed by `d_raw`:
```rust
        let d = std::slice::from_raw_parts(d_raw, (n * n) as usize);
```

We wrap `r_raw` also into a slice, but declare it mutable to allow writing into its memory block:
```rust
        let mut r = std::slice::from_raw_parts_mut(r_raw, (n * n) as usize);
```

Now we have two "not-unsafe" Rust primitive types that point to the same memory blocks as the pointers passed down by the C++ program calling our `step` function.
We can proceed by calling our Rust function `_step`, which provides the actual Rust implementation of the algorithm:
```rust
        _step(&mut r, d, n as usize);
```
The implementation of `_step` is discussed in more detail in the [next chapter](v0.md).


### Borrowing and ownership

The semantics of reference [borrowing](https://doc.rust-lang.org/book/second-edition/ch04-02-references-and-borrowing.html) is a fundamental part of the approach Rust takes to achieve thread safety.
When we pass `r` into `_step` in the previous code block example, we have to explicitly tell the compiler we are about to transfer a mutable reference `r` into the scope of `_step` from the scope of `step`.
In Rust this is called a mutable borrow.
Mutable borrows cannot be aliased, which means it is not possible to have more than one mutable reference to `r` within one scope at a time.
Immutable borrows, on the other hand, may be aliased.
This means that we can have an arbitrary amount of references to `d` in any scope.

Note also that the parent C++ program, which will be calling our `step` function, has [ownership](https://doc.rust-lang.org/book/second-edition/ch04-01-what-is-ownership.html) of the data.
While the Rust compiler may restrict the amount of concurrent mutable references for `r`, there is obviously no way it can ensure the parent program will not mutate the underlying data `r` refers to.
However, Rust is able to restrict the amount of mutable references we may create from `r`, which will be an important fact to consider when parallelizing our Rust implementation.


## References, additional reading

* [The Rust Book, 2nd ed.](https://doc.rust-lang.org/book/second-edition/index.html)
* [Rustonomicon (advanced Rust programming)](https://doc.rust-lang.org/nomicon/)
* [on the Rust FFI](https://blog.rust-lang.org/2015/04/24/Rust-Once-Run-Everywhere.html)
* [Rust for C++ programmers](https://github.com/nrc/r4cppp)
* [Rayon: data parellelism in Rust](http://smallcultfollowing.com/babysteps/blog/2015/12/18/rayon-data-parallelism-in-rust/)
