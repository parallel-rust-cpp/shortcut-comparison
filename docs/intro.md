# Rust

[Rust](https://www.rust-lang.org/en-US/) is a relatively young, systems programming language supporting compilation using LLVM into native code for various platforms.
The Rust compiler enforces strict ownership rules using compile-time type checking, which provides memory safety without the overhead from running a dynamic garbage collector.
Initially developed by engineers at Mozilla to provide more efficient tools to solve concurrency related problems in their C++ codebase for Firefox, Rust is currently maintained and owned by the Rust community.

The [rust-lang](https://www.rust-lang.org/en-US/) main page promises a "blazingly fast language", with "zero-cost abstractions".
Although the most promising features of Rust are related to strict memory safety guarantees, such as freedom of data races and thread safety, this project specifically explores the performance aspect of Rust.

## Calling Rust functions from C++

In order to avoid writing two separate programs for benchmarking, one for C++ and one for Rust, it would be nice if we could write Rust functions that the C++ linker would understand.
This would allow us to compile our Rust code into static libraries, which we can then link to our benchmarking program written in C++.

Consider the following C++ declaration of the `step` function:
```cpp
    extern "C" {
        void step(float*, const float*, int);
    }
```
We could use the [`unsafe`](https://doc.rust-lang.org/book/second-edition/ch19-01-unsafe-rust.html#unsafe-rust) keyword and implement the declaration as a Rust function using only raw pointers.
However, we would prefer to avoid doing this because then we would miss out on all the compile-time memory safety guarantees provided by the Rust compiler.
Instead, we implement the actual logic in a private function called `_step`, but expose its functionality using a public, thin C wrapper:
```rust
    #[no_mangle]
    pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: i32) {
        let d = std::slice::from_raw_parts(d_raw, (n * n) as usize);
        let mut r = std::slice::from_raw_parts_mut(r_raw, (n * n) as usize);
        _step(&mut r, d, n as usize);
    }
```

Let's break that down.

We use a compile-time [attribute](https://doc.rust-lang.org/reference/attributes.html#miscellaneous-attributes) which instructs the compiler to retain the symbol name of the function (`step`), without name mangling:
```rust
    #[no_mangle]
```

We declare an [`extern`](https://doc.rust-lang.org/book/second-edition/ch19-01-unsafe-rust.html#using-extern-functions-to-call-external-code) Rust function with public visibility, using the C-language application binary interface (ABI):
```rust
    pub unsafe extern "C" fn step(r_raw: *mut f32, d_raw: *const f32, n: i32) {
```
The function takes as arguments two pointers to single precision floating point numbers, and one 32-bit integer.

The [`unsafe`](https://doc.rust-lang.org/book/second-edition/ch19-01-unsafe-rust.html#unsafe-rust) keyword is a rather powerful feature of Rust, which basically disables most (but not all) of the compile-time memory safety checks within the scope declared unsafe.
In exchange, we are allowed to e.g. dereference raw pointers inside the unsafe scope, which would be a compile time error in "not-unsafe" sections.
Note that the unsafe scope does not extend to functions invoked inside the unsafe section, which in this case is only the 3 lines of `step`.

Instead of declaring all our functions unsafe and keep passing around raw pointers like in C, we wrap the raw pointers into [slices](https://doc.rust-lang.org/std/primitive.slice.html).
Slices are Rust primitive types which provide a dynamically-sized view into a block of memory.

Here, we construct an immutable slice of length `n * n`, starting at the address pointed by `d_raw`:
```rust
        let d = std::slice::from_raw_parts(d_raw, (n * n) as usize);
```

We wrap `r_raw` also into a slice, but declare it mutable to allow writing into its memory block:
```rust
        let mut r = std::slice::from_raw_parts_mut(r_raw, (n * n) as usize);
```

Now we have two "not-unsafe" Rust primitive types that point to the same memory blocks as the pointers passed down by the C++ program calling our `step` function.
We can proceed by calling our Rust function [`_step`](/src/rust/v0_baseline/src/lib.rs), which provides the actual Rust implementation of the [step](http://ppc.cs.aalto.fi/ch2/) function:
```rust
        _step(&mut r, d, n as usize);
```

Note that when we pass `r` into `_step`, we have to explicitly tell compiler that we are about to transfer a mutable reference `r` into the scope of `_step`.
While this might feel redundant, the semantics of [references](https://doc.rust-lang.org/book/second-edition/ch04-02-references-and-borrowing.html) (especially mutable references) are a fundamental part of the Rust language, allowing the compiler to do static memory safety checking.

## Parallel Rust

One of the most interesting aspects of Rust is its thread safety [guarantees](https://doc.rust-lang.org/book/second-edition/ch16-00-concurrency.html) regarding concurrent execution.
The Rust approach to safe concurrency is in extensive compile-time ownership analysis, which according to the documentation is supposed to transform runtime concurrency bugs into compile-time errors.
Given the inherent challenges of writing concurrent programs using imperative languages, the goal of Rust to provide fearless concurrency seems well justified.

### OpenMP?

The [reference solution](http://ppc.cs.aalto.fi/ch2/) implements parallel execution with the [OpenMP](http://ppc.cs.aalto.fi/ch2/openmp/) library.
OpenMP does not support Rust, so it would be nice if we had a parallelism library, implemented using a similar, work-stealing approach.
Fortunately, [Rayon](https://docs.rs/rayon/1.0.2/rayon/) provides one reasonably stable and easy to use Rust alternative to OpenMP.

### Parallelizing the step function

When parallelizing C++ code with OpenMP, it is the [responsibility](http://ppc.cs.aalto.fi/ch2/openmp/) of the programmer to verify no [race conditions](https://stackoverflow.com/questions/26998183/how-do-i-deal-with-a-data-race-in-openmp) are introduced from the parallelization.
By contrast, parallelizing Rust code with Rayon parallel iterators, data race freedom will be [guaranteed](http://smallcultfollowing.com/babysteps/blog/2015/12/18/rayon-data-parallelism-in-rust/#data-race-freedom) by the Rayon library.
This is possible due to the strong ownership requirements of the Rust static type system, which are enforced by the compiler.
E.g. concurrent threads are never allowed to share ownership of a mutable reference to a block of memory.

Lets take a look at an example.
Consider the following closure, which computes the minimums in the Rust implementation of the [v0 step function](/src/rust/v0_baseline/src/lib.rs):
```rust
    let _step_row = |(i, row): (usize, &mut [f32])| {
        for j in 0..n {
            let mut v = std::f32::INFINITY;
            for k in 0..n {
                let x = d[n*i + k];
                let y = d[n*k + j];
                let z = x + y;
                v = if z < v { z } else { v };
            }
            row[j] = v;
        }
    };
```
The function takes as parameter a tuple `(i, row)`, where `i` is a slice index and `row` is a mutable slice into some memory block containing single precision floating point numbers.
Note that the closure will capture the input slice `d` by reference from the outer scope (`_step` function).
All references are immutable by default in Rust, and so the only mutable reference is the slice passed in as parameter.

We are now able to parallelize our Rust implementation with an approach similar to the reference solution, which uses one thread per row of input `d` to write results into `r` for `n` pairs of elements from `d`.

We use the [`par_chunks_mut`](https://docs.rs/rayon/1.0.2/rayon/slice/trait.ParallelSliceMut.html#method.par_chunks_mut) function from Rayon to partition the result slice `r` into mutable, non-overlapping slices of length `n`, and apply the closure `_step_row` in parallel on each mutable slice, i.e. rows of `r`:
```rust
    r.par_chunks_mut(n).enumerate().for_each(_step_row);
```

If the above approach seems confusing, consider the last line of the `_step_row` closure:
```rust
    row[j] = v;
```
If we were to iterate over `r` using two nested for loops of length `n` (as in the [C++ reference solution](http://ppc.cs.aalto.fi/ch2/v0/) for v0), then for all `i` and `j`, `r[n*i + j]` would refer to the same memory location as `row[j]`, for every tuple `(i, row)` passed to `_step_row`.

## References, additional reading

* [The Rust Book, 2nd ed.](https://doc.rust-lang.org/book/second-edition/index.html)
* [Rustonomicon (advanced Rust programming)](https://doc.rust-lang.org/nomicon/)
* [on the Rust FFI](https://blog.rust-lang.org/2015/04/24/Rust-Once-Run-Everywhere.html)
* [Rust for C++ programmers](https://github.com/nrc/r4cppp)
* [Rayon: data parellelism in Rust](http://smallcultfollowing.com/babysteps/blog/2015/12/18/rayon-data-parallelism-in-rust/)
