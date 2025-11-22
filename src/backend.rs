use std::convert::TryInto;
use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, Shl, Shr, Sub};

use wide::{u16x16, u32x8, u64x4, u8x32, CmpEq};

//todo: simplify this, we could just keep type specific loading (like QueryBlock)
// and just call wide functions directly?

pub trait SimdBackend: Copy + 'static + Send + Sync + Default {
    type Simd: Copy
        + Add<Output = Self::Simd>
        + Sub<Output = Self::Simd>
        + BitAnd<Output = Self::Simd>
        + BitOr<Output = Self::Simd>
        + BitAndAssign<Self::Simd>
        + BitOrAssign<Self::Simd>
        + BitXor<Output = Self::Simd>
        // + Shl<Self::Simd, Output = Self::Simd>
        //   + Shr<Self::Simd, Output = Self::Simd>
        + Shl<i32, Output = Self::Simd>
        + Shr<i32, Output = Self::Simd>
        + Shl<u32, Output = Self::Simd>
        + Shr<u32, Output = Self::Simd>
        + CmpEq<Output = Self::Simd>;

    type Scalar: Copy + PartialEq;
    type LaneArray: AsRef<[Self::Scalar]> + AsMut<[Self::Scalar]> + Copy + Default;
    type QueryBlock: Copy + std::fmt::Debug;

    const LANES: usize;
    const LIMB_BITS: usize;
    const CARRY_SHIFT: u32;
    const MAX_POSITIVE: Self::Scalar;

    fn splat_all_ones() -> Self::Simd;
    fn splat_zero() -> Self::Simd;
    fn splat_one() -> Self::Simd;
    fn splat_scalar(value: Self::Scalar) -> Self::Simd;
    fn scalar_from_i64(value: i64) -> Self::Scalar;
    fn scalar_from_usize(value: usize) -> Self::Scalar {
        Self::scalar_from_i64(value as i64)
    }
    fn scalar_to_i64(value: Self::Scalar) -> i64;
    fn splat_from_i64(value: i64) -> Self::Simd {
        Self::splat_scalar(Self::scalar_from_i64(value))
    }
    fn splat_from_usize(value: usize) -> Self::Simd {
        Self::splat_scalar(Self::scalar_from_usize(value))
    }
    fn mask_word_to_scalar(word: u64) -> Self::Scalar;
    fn scalar_to_f32(value: Self::Scalar) -> f32;
    fn to_array(vec: Self::Simd) -> Self::LaneArray;
    fn from_array(arr: Self::LaneArray) -> Self::Simd;
    fn min(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd;
    fn blend(mask: Self::Simd, t: Self::Simd, f: Self::Simd) -> Self::Simd;
    fn simd_gt(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd;
    fn to_query_block(slice: &[u8]) -> Self::QueryBlock;
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
    const CARRY_SHIFT: u32 = 32;
    const MAX_POSITIVE: Self::Scalar = u32::MAX;

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
    fn scalar_from_i64(value: i64) -> Self::Scalar {
        value.try_into().expect("value does not fit in u32")
    }

    #[inline(always)]
    fn scalar_to_i64(value: Self::Scalar) -> i64 {
        value as i64
    }

    #[inline(always)]
    fn mask_word_to_scalar(word: u64) -> Self::Scalar {
        word as u32
    }

    #[inline(always)]
    fn scalar_to_f32(value: Self::Scalar) -> f32 {
        value as f32
    }

    #[inline(always)]
    fn to_array(vec: Self::Simd) -> Self::LaneArray {
        vec.to_array()
    }

    #[inline(always)]
    fn from_array(arr: Self::LaneArray) -> Self::Simd {
        u32x8::new(arr)
    }

    #[inline(always)]
    fn min(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        lhs.min(rhs)
    }

    #[inline(always)]
    fn blend(mask: Self::Simd, t: Self::Simd, f: Self::Simd) -> Self::Simd {
        mask.blend(t, f)
    }

    #[inline(always)]
    fn to_query_block(slice: &[u8]) -> Self::QueryBlock {
        u8x32::new(slice.try_into().expect("Slice must be 32 bytes"))
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
}

/// Backend using 64-bit limbs with four SIMD lanes, useful for very long queries.
#[derive(Clone, Copy, Debug, Default)]
pub struct I64x4Backend;

impl SimdBackend for I64x4Backend {
    type Simd = u64x4;
    type Scalar = u64;
    type LaneArray = [u64; 4];
    type QueryBlock = [u8; 64];

    const LANES: usize = 4;
    const LIMB_BITS: usize = 64;
    const CARRY_SHIFT: u32 = 64;
    const MAX_POSITIVE: Self::Scalar = u64::MAX;

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
    fn scalar_from_i64(value: i64) -> Self::Scalar {
        value as u64
    }

    #[inline(always)]
    fn scalar_to_i64(value: Self::Scalar) -> i64 {
        value as i64
    }

    #[inline(always)]
    fn mask_word_to_scalar(word: u64) -> Self::Scalar {
        word
    }

    #[inline(always)]
    fn scalar_to_f32(value: Self::Scalar) -> f32 {
        value as f64 as f32
    }

    #[inline(always)]
    fn to_array(vec: Self::Simd) -> Self::LaneArray {
        vec.to_array()
    }

    #[inline(always)]
    fn from_array(arr: Self::LaneArray) -> Self::Simd {
        u64x4::new(arr)
    }

    #[inline(always)]
    fn min(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        lhs.min(rhs)
    }

    #[inline(always)]
    fn blend(mask: Self::Simd, t: Self::Simd, f: Self::Simd) -> Self::Simd {
        mask.blend(t, f)
    }

    #[inline(always)]
    fn to_query_block(slice: &[u8]) -> Self::QueryBlock {
        slice.try_into().expect("Slice must be 64 bytes")
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
    const CARRY_SHIFT: u32 = 16;
    const MAX_POSITIVE: Self::Scalar = u16::MAX;

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
    fn scalar_from_i64(value: i64) -> Self::Scalar {
        value.try_into().expect("value does not fit in u16")
    }

    #[inline(always)]
    fn scalar_to_i64(value: Self::Scalar) -> i64 {
        value as i64
    }

    #[inline(always)]
    fn mask_word_to_scalar(word: u64) -> Self::Scalar {
        word as u16
    }

    #[inline(always)]
    fn scalar_to_f32(value: Self::Scalar) -> f32 {
        value as f32
    }

    #[inline(always)]
    fn to_array(vec: Self::Simd) -> Self::LaneArray {
        vec.to_array()
    }

    #[inline(always)]
    fn from_array(arr: Self::LaneArray) -> Self::Simd {
        u16x16::new(arr)
    }

    #[inline(always)]
    fn min(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        lhs.min(rhs)
    }

    #[inline(always)]
    fn blend(mask: Self::Simd, t: Self::Simd, f: Self::Simd) -> Self::Simd {
        mask.blend(t, f)
    }

    #[inline(always)]
    fn to_query_block(slice: &[u8]) -> Self::QueryBlock {
        slice.try_into().expect("Slice must be 16 bytes")
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
}

// Not sure, maybe U32 vs u32 is a bit confusing?
// maybe, like 32bits? or something
pub type U32 = I32x8Backend;
pub type U64 = I64x4Backend;
pub type U16 = I16x16Backend;
pub type DefaultBackend = U32;
