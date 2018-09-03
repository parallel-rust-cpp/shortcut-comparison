# Rust `f32::min`

It seems that an `if else` expression generates more efficient code compared to the `min` function implemented for the `f32` primitive.

## `v.min(z)`

Consider the Rust v0 [implementation](/src/rust/v0_baseline/src/lib.rs) of the step function.
The most performance critical part of the function is where we update the minimum:

```rust
let x = d[n*i + k];
let y = d[n*k + j];
let z = x + y;
v = v.min(z);
```

Running the benchmark for v0, with input `d` containing 1000 rows, gives the following results:
```
N (rows)   time (us)   instructions         cycles     insn/cycle
    1000     1910352    15071090836     6488529647           2.32
    1000     1970740    15062220206     6693203177           2.25
    1000     1932586    15003339575     6561395998           2.29
    1000     1937618    15076803471     6579849012           2.29
    1000     1963813    15111316824     6658706623           2.27
```

Inspecting the output of the compiler (using toolchain `nightly-x86_64-unknown-linux-gnu`) reveals that there seems to be something weird going on around the `vminss` instruction:

```
  6,17 │ 80:   lea    (%r11,%r10,1),%rax
  8,80 │       cmp    %rdx,%rax
       │     ↓ jae    101
  0,01 │       lea    (%r9,%rbp,1),%rax
  0,01 │       cmp    %rdx,%rax
       │     ↓ jae    f0
  6,23 │       vmovss (%r12,%r10,4),%xmm2
  9,19 │       add    $0x1,%r10
  5,98 │       vaddss (%rbx,%rbp,4),%xmm2,%xmm2
  6,87 │       vcmpunordss %xmm2,%xmm2,%xmm3
 11,11 │       vminss %xmm2,%xmm1,%xmm2
 45,55 │       vblendvps %xmm3,%xmm1,%xmm2,%xmm1
  0,01 │       add    %rcx,%rbp
       │       cmp    %rcx,%r10
       │     ↑ jb     80
```

It would seem as the compiler has generated some rather elaborate scheme involving copying bits around in the registers, just to compute the minimum of two floating point numbers.

## `if z < v { z } else { v }`

Let's replace the `min` function with a straightforward `if else` expression:

```rust
let x = d[n*i + k];
let y = d[n*k + j];
let z = x + y;
v = if z < v { z } else { v };
```

This significantly improves the instruction throughput:

```
N (rows)   time (us)   instructions         cycles     insn/cycle
    1000     1282589    13095861876     4363765751           3.00
    1000     1282371    12991188277     4360188681           2.98
    1000     1293437    13059277971     4403982333           2.97
    1000     1278064    13057718097     4346650003           3.00
    1000     1281490    12999692126     4365108443           2.98
```

We can also see that the `vcmpunordss` and `vblendvps` instructions are gone:

```
  0,01 │ 80:   lea    (%r11,%r10,1),%rax
 23,02 │       cmp    %rdx,%rax
       │     ↓ jae    f6
  0,01 │       lea    (%r9,%rbp,1),%rax
  0,05 │       cmp    %rdx,%rax
       │     ↓ jae    e5
  0,03 │       vmovss (%r12,%r10,4),%xmm2
 22,62 │       add    $0x1,%r10
  6,85 │       vaddss (%rbx,%rbp,4),%xmm2,%xmm2
 47,32 │       vminss %xmm1,%xmm2,%xmm1
  0,01 │       add    %rcx,%rbp
       │       cmp    %rcx,%r10
       │     ↑ jb     80
```
