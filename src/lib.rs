#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub trait Position<N: Sized>: Position128<N> + Position256<N> {
    fn find_value_eq(self, value: N) -> Option<usize>;

    fn find_value_ne(self, value: N) -> Option<usize>;
}

#[cfg(target_arch = "x86")]
pub trait Position<N: Sized>: Position128<N> {
    fn find_value_eq(self, value: N) -> Option<usize>;

    fn find_value_ne(self, value: N) -> Option<usize>;
}

pub trait Position128<N: Sized>    {
    fn find_value_eq_128(self, value: N) -> Option<usize>;

    fn find_value_ne_128(self, value: N) -> Option<usize>;
}

pub trait Position256<N: Sized>    {
    fn find_value_eq_256(self, value: N) -> Option<usize>;

    fn find_value_ne_256(self, value: N) -> Option<usize>;
}

pub trait PositionOps128: Sized {
    type Register: Copy;

    fn fill_register_128(self) -> Self::Register;

    #[inline(always)]
    fn chunk_size_128() -> usize {
        16 / std::mem::size_of::<Self>()
    }

    fn cmp_reg_eq_128(a: Self::Register, b: Self::Register) -> Self::Register;

    fn get_index_in_mask_128(mask: i32) -> Option<usize>;
}

pub trait PositionOps256: Sized {
    type Register: Copy;

    fn fill_register_256(self) -> Self::Register;

    #[inline(always)]
    fn chunk_size_256() -> usize {
        32 / std::mem::size_of::<Self>()
    }

    fn cmp_reg_eq_256(a: Self::Register, b: Self::Register) -> Self::Register;

    fn get_index_in_mask_256(mask: i32) -> Option<usize>;
}

impl<N> Position128<N> for &[N] where N: PositionOps128<Register=__m128i> + PartialEq + Copy {
    fn find_value_eq_128(self, value: N) -> Option<usize> {
        let cmp = value.fill_register_128();
        for (index, chunk) in self.chunks_exact(N::chunk_size_128()).enumerate() {
            let vals = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const _)};
            let eqmask = N::cmp_reg_eq_128(cmp, vals);
            let mask = unsafe { _mm_movemask_epi8(eqmask) };
            if let Some(sub_index) = N::get_index_in_mask_128(mask) {
                return Some((index * N::chunk_size_128()) + sub_index);
            }
        }
        for (index, &val) in self.chunks_exact(N::chunk_size_128()).remainder().iter().enumerate() {
            if val == value {
                return Some(index + (self.len() & !(N::chunk_size_128() - 1)));
            }
        }

        None
    }

    fn find_value_ne_128(self, value: N) -> Option<usize> {
        let cmp = value.fill_register_128();
        for (index, chunk) in self.chunks_exact(N::chunk_size_128()).enumerate() {
            let vals = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const _)};
            let eqmask = N::cmp_reg_eq_128(cmp, vals);
            let mask = unsafe { !_mm_movemask_epi8(eqmask) };
            if let Some(sub_index) = N::get_index_in_mask_128(mask) {
                return Some((index * N::chunk_size_128()) + sub_index);
            }
        }
        for (index, &val) in self.chunks_exact(N::chunk_size_128()).remainder().iter().enumerate() {
            if val != value {
                return Some(index + (self.len() & !(N::chunk_size_128() - 1)));
            }
        }

        None
    }
}

impl<N> Position256<N> for &[N] where N: PositionOps256<Register=__m256i> + PartialEq + Copy {
    fn find_value_eq_256(self, value: N) -> Option<usize> {
        let cmp = value.fill_register_256();
        for (index, chunk) in self.chunks_exact(N::chunk_size_256()).enumerate() {
            let vals = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const _)};
            let eqmask = N::cmp_reg_eq_256(cmp, vals);
            let mask = unsafe { _mm256_movemask_epi8(eqmask) };
            if let Some(sub_index) = N::get_index_in_mask_256(mask) {
                return Some((index * N::chunk_size_256()) + sub_index);
            }
        }
        for (index, &val) in self.chunks_exact(N::chunk_size_256()).remainder().iter().enumerate() {
            if val == value {
                return Some(index + (self.len() & !(N::chunk_size_256() - 1)));
            }
        }

        None
    }

    fn find_value_ne_256(self, value: N) -> Option<usize> {
        let cmp = value.fill_register_256();
        for (index, chunk) in self.chunks_exact(N::chunk_size_256()).enumerate() {
            let vals = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const _)};
            let eqmask = N::cmp_reg_eq_256(cmp, vals);
            let mask = unsafe { !_mm256_movemask_epi8(eqmask) };
            if let Some(sub_index) = N::get_index_in_mask_256(mask) {
                return Some((index * N::chunk_size_256()) + sub_index);
            }
        }
        for (index, &val) in self.chunks_exact(N::chunk_size_256()).remainder().iter().enumerate() {
            if val != value {
                return Some(index + (self.len() & !(N::chunk_size_256() - 1)));
            }
        }

        None
    }
}

impl<N> Position<N> for &[N] where N: Sized + Copy + PositionOps128<Register=__m128i> + PositionOps256<Register=__m256i> + PartialEq {
    fn find_value_eq(self, value: N) -> Option<usize> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            return if is_x86_feature_detected!("avx") {
                self.find_value_eq_256(value)
            } else {
                self.find_value_eq_128(value)
            };
        }
        self.iter().position(|x| x.eq(&value))
    }

    fn find_value_ne(self, value: N) -> Option<usize> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            return if is_x86_feature_detected!("avx") {
                self.find_value_ne_256(value)
            } else {
                self.find_value_ne_128(value)
            };
        }
        self.iter().position(|x| x.ne(&value))

    }
}

impl PositionOps128 for u8 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register_128(self) -> Self::Register {
        unsafe { _mm_set1_epi8(self as i8) }
    }

    #[inline(always)]
    fn cmp_reg_eq_128(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi8(a, b) }
    }



    #[inline(always)]
    fn get_index_in_mask_128(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize;
        if index < 16 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps128 for i8 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register_128(self) -> Self::Register {
        unsafe { _mm_set1_epi8(self) }
    }

    #[inline(always)]
    fn cmp_reg_eq_128(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi8(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_128(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize;
        if index < 16 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps128 for u16 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register_128(self) -> Self::Register {
        unsafe { _mm_set1_epi16(self as i16) }
    }

    #[inline(always)]
    fn cmp_reg_eq_128(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi16(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_128(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize / 2;
        if index < 8 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps128 for i16 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register_128(self) -> Self::Register {
        unsafe { _mm_set1_epi16(self) }
    }

    #[inline(always)]
    fn cmp_reg_eq_128(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi16(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_128(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize / 2;
        if index < 8 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps128 for u32 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register_128(self) -> Self::Register {
        unsafe { _mm_set1_epi32(self as i32) }
    }

    #[inline(always)]
    fn cmp_reg_eq_128(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_128(mask: i32) -> Option<usize> {
        for i in 0..4 {
            let bit_mask = 1 << (i * 4);
            if mask & bit_mask == bit_mask {
                return Some(i);
            }
        }

        None
    }
}

impl PositionOps128 for i32 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register_128(self) -> Self::Register {
        unsafe { _mm_set1_epi32(self) }
    }

    #[inline(always)]
    fn cmp_reg_eq_128(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_128(mask: i32) -> Option<usize> {
        for i in 0..4 {
            let bit_mask = 1 << (i * 4);
            if mask & bit_mask == bit_mask {
                return Some(i);
            }
        }

        None
    }
}

impl PositionOps128 for u64 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register_128(self) -> Self::Register {
        unsafe { _mm_set1_epi64x(self as i64) }
    }

    #[inline(always)]
    fn cmp_reg_eq_128(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi64(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_128(mask: i32) -> Option<usize> {
        for i in 0..2 {
            let bit_mask = 1 << (i * 8);
            if mask & bit_mask == bit_mask {
                return Some(i);
            }
        }

        None
    }
}

impl PositionOps128 for i64 {
    type Register = __m128i;

    #[inline(always)]
    fn fill_register_128(self) -> Self::Register {
        unsafe { _mm_set1_epi64x(self) }
    }

    #[inline(always)]
    fn cmp_reg_eq_128(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm_cmpeq_epi64(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_128(mask: i32) -> Option<usize> {
        for i in 0..2 {
            let bit_mask = 1 << (i * 8);
            if mask & bit_mask == bit_mask {
                return Some(i);
            }
        }

        None
    }
}

impl PositionOps256 for u8 {
    type Register = __m256i;

    #[inline(always)]
    fn fill_register_256(self) -> Self::Register {
        unsafe { _mm256_set1_epi8(self as i8) }
    }


    #[inline(always)]
    fn cmp_reg_eq_256(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm256_cmpeq_epi8(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_256(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize;
        if index < 32 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps256 for i8 {
    type Register = __m256i;

    #[inline(always)]
    fn fill_register_256(self) -> Self::Register {
        unsafe { _mm256_set1_epi8(self) }
    }


    #[inline(always)]
    fn cmp_reg_eq_256(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm256_cmpeq_epi8(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_256(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize;
        if index < 32 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps256 for u16 {
    type Register = __m256i;

    #[inline(always)]
    fn fill_register_256(self) -> Self::Register {
        unsafe { _mm256_set1_epi16(self as i16) }
    }

    #[inline(always)]
    fn cmp_reg_eq_256(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm256_cmpeq_epi16(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_256(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize / 2;
        if index < 16 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps256 for i16 {
    type Register = __m256i;

    #[inline(always)]
    fn fill_register_256(self) -> Self::Register {
        unsafe { _mm256_set1_epi16(self) }
    }

    #[inline(always)]
    fn cmp_reg_eq_256(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm256_cmpeq_epi16(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_256(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize / 2;
        if index < 16 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps256 for u32 {
    type Register = __m256i;

    #[inline(always)]
    fn fill_register_256(self) -> Self::Register {
        unsafe { _mm256_set1_epi32(self as i32) }
    }

    #[inline(always)]
    fn cmp_reg_eq_256(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm256_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_256(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize / 4;
        if index < 8 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps256 for i32 {
    type Register = __m256i;

    #[inline(always)]
    fn fill_register_256(self) -> Self::Register {
        unsafe { _mm256_set1_epi32(self) }
    }

    #[inline(always)]
    fn cmp_reg_eq_256(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm256_cmpeq_epi32(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_256(mask: i32) -> Option<usize> {
        let index = mask.trailing_zeros() as usize / 4;
        if index < 8 {
            Some(index)
        } else {
            None
        }
    }
}

impl PositionOps256 for u64 {
    type Register = __m256i;

    #[inline(always)]
    fn fill_register_256(self) -> Self::Register {
        unsafe { _mm256_set1_epi64x(self as i64) }
    }

    #[inline(always)]
    fn cmp_reg_eq_256(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm256_cmpeq_epi64(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_256(mask: i32) -> Option<usize> {
        for i in 0..4 {
            let bit_mask = 1 << (i * 8);
            if mask & bit_mask == bit_mask {
                return Some(i);
            }
        }

        None
    }
}

impl PositionOps256 for i64 {
    type Register = __m256i;

    #[inline(always)]
    fn fill_register_256(self) -> Self::Register {
        unsafe { _mm256_set1_epi64x(self) }
    }

    #[inline(always)]
    fn cmp_reg_eq_256(a: Self::Register, b: Self::Register) -> Self::Register {
        unsafe { _mm256_cmpeq_epi64(a, b) }
    }

    #[inline(always)]
    fn get_index_in_mask_256(mask: i32) -> Option<usize> {
        for i in 0..4 {
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
