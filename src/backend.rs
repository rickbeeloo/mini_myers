use std::convert::TryInto;
use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, Shl, Shr, Sub};

use wide::{u16x16, u32x8, u64x4, u8x32, CmpEq};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// So much boilerplate still, cant we generalize over the wide type or something

pub trait SimdBackend: Copy + 'static + Send + Sync + Default + std::fmt::Debug {
    type Simd: Copy
        + Add<Output = Self::Simd>
        + Sub<Output = Self::Simd>
        + BitAnd<Output = Self::Simd>
        + BitOr<Output = Self::Simd>
        + BitAndAssign<Self::Simd>
        + BitOrAssign<Self::Simd>
        + BitXor<Output = Self::Simd>
        + Shl<i32, Output = Self::Simd>
        + Shr<i32, Output = Self::Simd>
        + Shl<u32, Output = Self::Simd>
        + Shr<u32, Output = Self::Simd>
        + CmpEq<Output = Self::Simd>
        + PartialEq;

    type Scalar: Copy + PartialEq + std::fmt::Debug;
    type LaneArray: AsRef<[Self::Scalar]> + AsMut<[Self::Scalar]> + Copy + Default;
    type QueryBlock: Copy + std::fmt::Debug;

    const LANES: usize;
    const LIMB_BITS: usize;

    /// Convert a u64 mask word to the backend's scalar type
    fn mask_word_to_scalar(word: u64) -> Self::Scalar;

    /// Convert i64 to the backend's scalar type
    fn scalar_from_i64(value: i64) -> Self::Scalar;

    /// Convert array to SIMD vector
    fn from_array(arr: Self::LaneArray) -> Self::Simd;

    /// Convert SIMD vector to array
    fn to_array(vec: Self::Simd) -> Self::LaneArray;

    /// Unsigned greater-than comparison (requires special handling)
    fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd;

    /// Convert byte slice to query block type
    fn to_query_block(slice: &[u8]) -> Self::QueryBlock;
    fn scalar_to_u64(value: Self::Scalar) -> u64;

    // Minimal splat helpers - these are thin wrappers but necessary since we can't call splat generically
    fn splat_all_ones() -> Self::Simd;
    fn splat_zero() -> Self::Simd;
    fn splat_one() -> Self::Simd;
    fn splat_scalar(value: Self::Scalar) -> Self::Simd;

    // Check if any lane is zero (hit detected)
    fn lanes_with_zero(vec: Self::Simd) -> u32;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct I32x8Backend;

impl SimdBackend for I32x8Backend {
    type Simd = u32x8;
    type Scalar = u32;
    type LaneArray = [u32; 8];
    type QueryBlock = u8x32;

    const LANES: usize = 8;
    const LIMB_BITS: usize = 32;

    #[inline(always)]
    fn mask_word_to_scalar(word: u64) -> Self::Scalar {
        word as u32
    }

    #[inline(always)]
    fn scalar_from_i64(value: i64) -> Self::Scalar {
        value.try_into().expect("value does not fit in u32")
    }

    #[inline(always)]
    fn from_array(arr: Self::LaneArray) -> Self::Simd {
        u32x8::new(arr)
    }

    #[inline(always)]
    fn to_array(vec: Self::Simd) -> Self::LaneArray {
        vec.to_array()
    }

    #[inline(always)]
    fn to_query_block(slice: &[u8]) -> Self::QueryBlock {
        u8x32::new(slice.try_into().expect("Slice must be 32 bytes"))
    }

    #[inline(always)]
    fn splat_all_ones() -> Self::Simd {
        u32x8::splat(!0)
    }

    #[inline(always)]
    fn splat_zero() -> Self::Simd {
        u32x8::splat(0)
    }

    #[inline(always)]
    fn splat_one() -> Self::Simd {
        u32x8::splat(1)
    }

    #[inline(always)]
    fn splat_scalar(value: Self::Scalar) -> Self::Simd {
        u32x8::splat(value)
    }

    #[inline(always)]
    fn scalar_to_u64(value: Self::Scalar) -> u64 {
        value as u64
    }

    // Thanks Ragnar
    // https://github.com/RagnarGrootKoerkamp/sassy/blob/0772487a8f08c37f5742aa6217f4744312b38a8e/src/profiles.rs#L50-L67
    #[inline(always)]
    fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        unsafe {
            use std::mem::transmute;
            use wide::{i32x8, CmpGt};
            let a: i32x8 = transmute(lhs);
            let b: i32x8 = transmute(rhs);
            let mask = i32x8::splat((1u32 << 31) as i32);
            transmute(CmpGt::simd_gt(a ^ mask, b ^ mask))
        }
    }

    #[inline(always)]
    fn lanes_with_zero(vec: Self::Simd) -> u32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::mem::transmute;
            let v: __m256i = transmute(vec);
            let zero = _mm256_setzero_si256();
            let eq = _mm256_cmpeq_epi32(v, zero);
            // movemask_ps returns bit i set if float i's sign bit is set
            _mm256_movemask_ps(transmute(eq)) as u32
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let arr = Self::to_array(vec);
            let mut result = 0u32;
            for (i, &v) in arr.as_ref().iter().enumerate() {
                if v == 0 {
                    result |= 1 << i;
                }
            }
            result
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct I64x4Backend;

impl SimdBackend for I64x4Backend {
    type Simd = u64x4;
    type Scalar = u64;
    type LaneArray = [u64; 4];
    type QueryBlock = [u8; 64];

    const LANES: usize = 4;
    const LIMB_BITS: usize = 64;

    #[inline(always)]
    fn mask_word_to_scalar(word: u64) -> Self::Scalar {
        word
    }

    #[inline(always)]
    fn scalar_from_i64(value: i64) -> Self::Scalar {
        value as u64
    }

    #[inline(always)]
    fn from_array(arr: Self::LaneArray) -> Self::Simd {
        u64x4::new(arr)
    }

    #[inline(always)]
    fn to_array(vec: Self::Simd) -> Self::LaneArray {
        vec.to_array()
    }

    #[inline(always)]
    fn to_query_block(slice: &[u8]) -> Self::QueryBlock {
        slice.try_into().expect("Slice must be 64 bytes")
    }

    #[inline(always)]
    fn splat_all_ones() -> Self::Simd {
        u64x4::splat(!0)
    }

    #[inline(always)]
    fn splat_zero() -> Self::Simd {
        u64x4::splat(0)
    }

    #[inline(always)]
    fn splat_one() -> Self::Simd {
        u64x4::splat(1)
    }

    #[inline(always)]
    fn splat_scalar(value: Self::Scalar) -> Self::Simd {
        u64x4::splat(value)
    }

    #[inline(always)]
    fn scalar_to_u64(value: Self::Scalar) -> u64 {
        value
    }

    // Thanks Ragnar
    // https://github.com/RagnarGrootKoerkamp/sassy/blob/0772487a8f08c37f5742aa6217f4744312b38a8e/src/profiles.rs#L50-L67
    #[inline(always)]
    fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        unsafe {
            use std::mem::transmute;
            use wide::{i64x4, CmpGt};
            let a: i64x4 = transmute(lhs);
            let b: i64x4 = transmute(rhs);
            let mask = i64x4::splat((1u64 << 63) as i64);
            transmute(CmpGt::simd_gt(a ^ mask, b ^ mask))
        }
    }

    #[inline(always)]
    fn lanes_with_zero(vec: Self::Simd) -> u32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::mem::transmute;
            let v: __m256i = transmute(vec);
            let zero = _mm256_setzero_si256();
            let eq = _mm256_cmpeq_epi64(v, zero);
            // For 64-bit lanes, use movemask_pd (treats as 4 doubles)
            _mm256_movemask_pd(transmute(eq)) as u32
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let arr = Self::to_array(vec);
            let mut result = 0u32;
            for (i, &v) in arr.as_ref().iter().enumerate() {
                if v == 0 {
                    result |= 1 << i;
                }
            }
            result
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct I16x16Backend;

impl SimdBackend for I16x16Backend {
    type Simd = u16x16;
    type Scalar = u16;
    type LaneArray = [u16; 16];
    type QueryBlock = [u8; 16];

    const LANES: usize = 16;
    const LIMB_BITS: usize = 16;

    #[inline(always)]
    fn mask_word_to_scalar(word: u64) -> Self::Scalar {
        word as u16
    }

    #[inline(always)]
    fn scalar_from_i64(value: i64) -> Self::Scalar {
        value.try_into().expect("value does not fit in u16")
    }

    #[inline(always)]
    fn from_array(arr: Self::LaneArray) -> Self::Simd {
        u16x16::new(arr)
    }

    #[inline(always)]
    fn to_array(vec: Self::Simd) -> Self::LaneArray {
        vec.to_array()
    }

    #[inline(always)]
    fn to_query_block(slice: &[u8]) -> Self::QueryBlock {
        slice.try_into().expect("Slice must be 16 bytes")
    }

    #[inline(always)]
    fn splat_all_ones() -> Self::Simd {
        u16x16::splat(!0)
    }

    #[inline(always)]
    fn splat_zero() -> Self::Simd {
        u16x16::splat(0)
    }

    #[inline(always)]
    fn splat_one() -> Self::Simd {
        u16x16::splat(1)
    }

    #[inline(always)]
    fn splat_scalar(value: Self::Scalar) -> Self::Simd {
        u16x16::splat(value)
    }

    #[inline(always)]
    fn scalar_to_u64(value: Self::Scalar) -> u64 {
        value as u64
    }

    #[inline(always)]
    fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        unsafe {
            use std::mem::transmute;
            use wide::{i16x16, CmpGt};
            let a: i16x16 = transmute(lhs);
            let b: i16x16 = transmute(rhs);
            let mask = i16x16::splat((1u32 << 15) as i16); // 1u32 << 15 is 32768
            transmute(CmpGt::simd_gt(a ^ mask, b ^ mask))
        }
    }

    #[inline(always)]
    fn lanes_with_zero(vec: Self::Simd) -> u32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::mem::transmute;
            let v: __m256i = transmute(vec);
            let zero = _mm256_setzero_si256();
            let eq = _mm256_cmpeq_epi16(v, zero);
            let mask = _mm256_movemask_epi8(eq) as u32;
            // Pack: bits 0,2,4,...,30 -> bits 0,1,2,...,15
            // Use pext if available, otherwise bit manipulation
            #[cfg(target_feature = "bmi2")]
            {
                std::arch::x86_64::_pext_u32(mask, 0x55555555)
            }
            #[cfg(not(target_feature = "bmi2"))]
            {
                // Manual bit extraction: every other bit
                let mut result = 0u32;
                for i in 0..16 {
                    if (mask & (1 << (i * 2))) != 0 {
                        result |= 1 << i;
                    }
                }
                result
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let arr = Self::to_array(vec);
            let mut result = 0u32;
            for (i, &v) in arr.as_ref().iter().enumerate() {
                if v == 0 {
                    result |= 1 << i;
                }
            }
            result
        }
    }
}

// Not sure, maybe U32 vs u32 is a bit confusing?
// maybe, like 32bits? or something
pub type U32 = I32x8Backend;
pub type U64 = I64x4Backend;
pub type U16 = I16x16Backend;
pub type DefaultBackend = U32;
