#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub trait Position<N: Sized> {
    fn find_value_eq(self, value: N) -> Option<usize>;

    fn find_value_ne(self, value: N) -> Option<usize>;
}

pub trait PositionOps {
    type Register: Copy;

    fn fill_register(self) -> Self::Register;

    fn chunk_size() -> usize;

    fn cmp_reg_eq(a: Self::Register, b: Self::Register) -> Self::Register;

    fn get_index_in_mask(mask: i32) -> Option<usize>;
}

impl<N> Position<N> for &[N] where N: PositionOps<Register=__m128i> + PartialEq + Copy {
    fn find_value_eq(self, value: N) -> Option<usize> {
        let cmp = value.fill_register();
        for (index, chunk) in self.chunks_exact(N::chunk_size()).enumerate() {
            let vals = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const _)};
            let eqmask = N::cmp_reg_eq(cmp, vals);
            let mask = unsafe { _mm_movemask_epi8(eqmask) };
            if let Some(sub_index) = N::get_index_in_mask(mask) {
                return Some((index * N::chunk_size()) + sub_index);
            }
        }
        for (index, &val) in self.chunks_exact(N::chunk_size()).remainder().iter().enumerate() {
            if val == value {
                return Some(index + (self.len() & !(N::chunk_size() - 1)));
            }
        }

        None
    }

    fn find_value_ne(self, value: N) -> Option<usize> {
        let cmp = value.fill_register();
        for (index, chunk) in self.chunks_exact(N::chunk_size()).enumerate() {
            let vals = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const _)};
            let eqmask = N::cmp_reg_eq(cmp, vals);
            let mask = unsafe { !_mm_movemask_epi8(eqmask) };
            if let Some(sub_index) = N::get_index_in_mask(mask) {
                return Some((index * N::chunk_size()) + sub_index);
            }
        }
        for (index, &val) in self.chunks_exact(N::chunk_size()).remainder().iter().enumerate() {
            if val != value {
                return Some(index + (self.len() & !(N::chunk_size() - 1)));
            }
        }

        None
    }
}

impl PositionOps for u8 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register(self) -> Self::Register {
        unsafe { _mm_set1_epi8(self as i8) }
    }

    #[inline(always)]
    fn chunk_size() -> usize {
        16
    }

    #[inline(always)]
    fn cmp_reg_eq(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi8(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize;
        if index < 16 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps for i8 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register(self) -> Self::Register {
        unsafe { _mm_set1_epi8(self) }
    }

    #[inline(always)]
    fn chunk_size() -> usize {
        16
    }

    #[inline(always)]
    fn cmp_reg_eq(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi8(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize;
        if index < 16 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps for u16 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register(self) -> Self::Register {
        unsafe { _mm_set1_epi16(self as i16) }
    }

    #[inline(always)]
    fn chunk_size() -> usize {
        8
    }

    #[inline(always)]
    fn cmp_reg_eq(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi16(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize / 2;
        if index < 8 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps for i16 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register(self) -> Self::Register {
        unsafe { _mm_set1_epi16(self) }
    }

    #[inline(always)]
    fn chunk_size() -> usize {
        8
    }

    #[inline(always)]
    fn cmp_reg_eq(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi16(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize / 2;
        if index < 8 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps for u32 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register(self) -> Self::Register {
        unsafe { _mm_set1_epi32(self as i32) }
    }

    #[inline(always)]
    fn chunk_size() -> usize {
        4
    }

    #[inline(always)]
    fn cmp_reg_eq(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask(mask: i32) -> Option<usize> {
        for i in 0..4 {
            let bit_mask = 1 << (i * 4);
            if mask & bit_mask == bit_mask {
                return Some(i);
            }
        }

        None
    }
}

impl PositionOps for i32 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register(self) -> Self::Register {
        unsafe { _mm_set1_epi32(self) }
    }

    #[inline(always)]
    fn chunk_size() -> usize {
        4
    }

    #[inline(always)]
    fn cmp_reg_eq(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask(mask: i32) -> Option<usize> {
        for i in 0..4 {
            let bit_mask = 1 << (i * 4);
            if mask & bit_mask == bit_mask {
                return Some(i);
            }
        }

        None
    }
}

impl PositionOps for u64 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register(self) -> Self::Register {
        unsafe { _mm_set1_epi64x(self as i64) }
    }

    #[inline(always)]
    fn chunk_size() -> usize {
        2
    }

    #[inline(always)]
    fn cmp_reg_eq(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi64(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask(mask: i32) -> Option<usize> {
        for i in 0..2 {
            let bit_mask = 1 << (i * 8);
            if mask & bit_mask == bit_mask {
                return Some(i);
            }
        }

        None
    }
}

impl PositionOps for i64 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register(self) -> Self::Register {
        unsafe { _mm_set1_epi64x(self) }
    }

    #[inline(always)]
    fn chunk_size() -> usize {
        2
    }

    #[inline(always)]
    fn cmp_reg_eq(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi64(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask(mask: i32) -> Option<usize> {
        for i in 0..2 {
            let bit_mask = 1 << (i * 8);
            if mask & bit_mask == bit_mask {
                return Some(i);
            }
        }

        None
    }
}


#[cfg(test)]
mod tests {
    macro_rules! test_position_impl {
        ($typ:ty, $fn1:ident, $fn2:ident, $fn3:ident) => {
            #[test]
            fn $fn1() {
                let a: [$typ; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
                assert_eq!(a.find_value_eq(5), Some(4));
            }

            #[test]
            fn $fn2() {
                let a: Vec<$typ> = (0..=255).collect();
                for (index, &val) in a.clone().iter().enumerate() {
                    assert_eq!(a.find_value_eq(val), Some(index));
                }
            }

            #[test]
            fn $fn3() {
                let mut a: Vec<$typ> = vec![0; 10000];
                a.push(1);
                a.extend(vec![0; 10000]);

                assert_eq!(a.find_value_ne(0), Some(10000));
            }
        };
    }

    macro_rules! test_position_signed_impl {
        ($typ:ty, $fn1:ident, $fn2:ident, $fn3:ident) => {
            #[test]
            fn $fn1() {
                let a: [$typ; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];
                assert_eq!(a.find_value_eq(1), Some(5));
            }

            #[test]
            fn $fn2() {
                let a: Vec<$typ> = (-128..=127).collect();
                for (index, &val) in a.clone().iter().enumerate() {
                    assert_eq!(a.find_value_eq(val), Some(index));
                }
            }

            #[test]
            fn $fn3() {
                let mut a: Vec<$typ> = vec![0; 10000];
                a.push(1);
                a.extend(vec![0; 10000]);

                assert_eq!(a.find_value_ne(0), Some(10000));
            }
        };
    }

    use crate::Position;

    test_position_impl!(u8, test_u8_find_eq_short, test_u8_find_eq_medium, test_u8_find_ne_long);
    test_position_impl!(u16, test_u16_find_eq_short, test_u16_find_eq_medium, test_u16_find_ne_long);
    test_position_impl!(u32, test_u32_find_eq_short, test_u32_find_eq_medium, test_u32_find_ne_long);
    test_position_impl!(u64, test_u64_find_eq_short, test_u64_find_eq_medium, test_u64_find_ne_long);

    test_position_signed_impl!(i8, test_i8_find_eq_short, test_i8_find_eq_medium, test_i8_find_ne_long);
    test_position_signed_impl!(i16, test_i16_find_eq_short, test_i16_find_eq_medium, test_i16_find_ne_long);
    test_position_signed_impl!(i32, test_i32_find_eq_short, test_i32_find_eq_medium, test_i32_find_ne_long);
    test_position_signed_impl!(i64, test_i64_find_eq_short, test_i64_find_eq_medium, test_i64_find_ne_long);

}
