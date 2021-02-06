#![feature(test)]
extern crate test;

extern crate simd_position;

macro_rules! bench_position_eq {
    ($size:literal, $typ:ty, $fn1:ident, $fn2:ident) => {
        #[bench]
        fn $fn1(b: &mut Bencher) {
            let a: Vec<$typ> = (0..$size).collect();
            b.iter(|| a.iter().position(|x| x.eq(&($size / 2))));
        }

        #[bench]
        fn $fn2(b: &mut Bencher) {
            let a: Vec<$typ> = (0..$size).collect();
            b.iter(|| a.find_value_eq($size / 2));
        }
    };
}

macro_rules! bench_first_nonzero {
    ($size:literal, $typ:ty, $fn1:ident, $fn2:ident) => {
        #[bench]
        fn $fn1(b: &mut Bencher) {
            let mut a: Vec<$typ> = vec![0; $size];
            a.push(1);
            a.extend(vec![0; $size]);

            b.iter(|| a.iter().position(|x| x.ne(&0)));
        }

        #[bench]
        fn $fn2(b: &mut Bencher) {
            let mut a: Vec<$typ> = vec![0; $size];
            a.push(1);
            a.extend(vec![0; $size]);

            b.iter(|| a.find_value_ne(0));
        }
    };
}

mod bench {
    use test::Bencher;

    use simd_position::Position;

    bench_position_eq!(
        10,
        u8,
        bench_u8_position_naive_10,
        bench_u8_position_simd_10
    );
    bench_position_eq!(
        100,
        u8,
        bench_u8_position_naive_100,
        bench_u8_position_simd_100
    );
    bench_position_eq!(
        255,
        u8,
        bench_u8_position_naive_255,
        bench_u8_position_simd_255
    );

    bench_first_nonzero!(
        10,
        u8,
        bench_u8_first_nonzero_naive_10,
        bench_u8_first_nonzero_simd_10
    );
    bench_first_nonzero!(
        100,
        u8,
        bench_u8_first_nonzero_naive_100,
        bench_u8_first_nonzero_simd_100
    );
    bench_first_nonzero!(
        1000,
        u8,
        bench_u8_first_nonzero_naive_1000,
        bench_u8_first_nonzero_simd_1000
    );
    bench_first_nonzero!(
        10000,
        u8,
        bench_u8_first_nonzero_naive_10000,
        bench_u8_first_nonzero_simd_10000
    );
    bench_first_nonzero!(
        100000,
        u8,
        bench_u8_first_nonzero_naive_100000,
        bench_u8_first_nonzero_simd_100000
    );
    bench_first_nonzero!(
        1000000,
        u8,
        bench_u8_first_nonzero_naive_1000000,
        bench_u8_first_nonzero_simd_1000000
    );

    bench_position_eq!(
        10,
        u32,
        bench_u32_position_naive_10,
        bench_u32_position_simd_10
    );
    bench_position_eq!(
        100,
        u32,
        bench_u32_position_naive_100,
        bench_u32_position_simd_100
    );
    bench_position_eq!(
        1000,
        u32,
        bench_u32_position_naive_1000,
        bench_u32_position_simd_1000
    );
    bench_position_eq!(
        10000,
        u32,
        bench_u32_position_naive_10000,
        bench_u32_position_simd_10000
    );
    bench_position_eq!(
        100000,
        u32,
        bench_u32_position_naive_100000,
        bench_u32_position_simd_100000
    );
    bench_position_eq!(
        1000000,
        u32,
        bench_u32_position_naive_1000000,
        bench_u32_position_simd_1000000
    );

    bench_first_nonzero!(
        10,
        u32,
        bench_u32_first_nonzero_naive_10,
        bench_u32_first_nonzero_simd_10
    );
    bench_first_nonzero!(
        100,
        u32,
        bench_u32_first_nonzero_naive_100,
        bench_u32_first_nonzero_simd_100
    );
    bench_first_nonzero!(
        1000,
        u32,
        bench_u32_first_nonzero_naive_1000,
        bench_u32_first_nonzero_simd_1000
    );
    bench_first_nonzero!(
        10000,
        u32,
        bench_u32_first_nonzero_naive_10000,
        bench_u32_first_nonzero_simd_10000
    );
    bench_first_nonzero!(
        100000,
        u32,
        bench_u32_first_nonzero_naive_100000,
        bench_u32_first_nonzero_simd_100000
    );
    bench_first_nonzero!(
        1000000,
        u32,
        bench_u32_first_nonzero_naive_1000000,
        bench_u32_first_nonzero_simd_1000000
    );
}
