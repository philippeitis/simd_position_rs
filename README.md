# simd_position
A crate providing SIMD acceleration for finding the position of the first value in a collection either equal to, or not equal to a given value.

```rust
use simd_position::Position;
fn main() {
    let a = vec![1, 2, 3, 4];
    let val = 2;
    assert_eq!(a.iter().position(|x| x.eq(&val)), a.find_val_eq(val));
    assert_eq!(a.iter().position(|x| x.ne(&val)), a.find_val_ne(val));
}
```

## Benchmarks
Benchmarks using `RUSTFLAGS=-C target-cpu=znver2`, `cargo +nightly bench`. Code available in `src/benchmarks/bench.rs`
```
test bench::bench_u32_first_nonzero_naive_1000000 ... bench:     480,178 ns/iter (+/- 38,643)
test bench::bench_u32_first_nonzero_simd_1000000  ... bench:      81,858 ns/iter (+/- 4,781)

test bench::bench_u32_position_naive_1000000      ... bench:     120,047 ns/iter (+/- 6,366)
test bench::bench_u32_position_simd_1000000       ... bench:      40,603 ns/iter (+/- 4,124)

test bench::bench_u8_first_nonzero_naive_1000000  ... bench:     241,350 ns/iter (+/- 16,515)
test bench::bench_u8_first_nonzero_simd_1000000   ... bench:      20,291 ns/iter (+/- 1,008)
```