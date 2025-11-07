use std::convert::TryInto;
use std::ops::{Add, BitAnd, BitOr, BitXor, Shl, Shr, Sub};

use wide::{i32x8, i64x4, CmpEq, CmpGt};

pub trait SimdBackend: Copy + 'static {
    type Simd: Copy
        + Add<Output = Self::Simd>
        + Sub<Output = Self::Simd>
        + BitAnd<Output = Self::Simd>
        + BitOr<Output = Self::Simd>
        + BitXor<Output = Self::Simd>
        + Shl<Self::Simd, Output = Self::Simd>
        + Shr<Self::Simd, Output = Self::Simd>
        + Shl<i32, Output = Self::Simd>
        + Shr<i32, Output = Self::Simd>
        + Shl<u32, Output = Self::Simd>
        + Shr<u32, Output = Self::Simd>
        + CmpEq<Output = Self::Simd>
        + CmpGt<Output = Self::Simd>;

    type Scalar: Copy + PartialEq;
    type LaneArray: AsRef<[Self::Scalar]> + AsMut<[Self::Scalar]> + Copy + Default;

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
}

/// Default backend using 32-bit limbs with eight SIMD lanes (AVX2-friendly).
#[derive(Clone, Copy, Debug, Default)]
pub struct I32x8Backend;

impl SimdBackend for I32x8Backend {
    type Simd = i32x8;
    type Scalar = i32;
    type LaneArray = [i32; 8];

    const LANES: usize = 8;
    const LIMB_BITS: usize = 32;
    const CARRY_SHIFT: u32 = 31;
    const MAX_POSITIVE: Self::Scalar = i32::MAX;

    #[inline(always)]
    fn splat_all_ones() -> Self::Simd {
        i32x8::splat(!0)
    }

    #[inline(always)]
    fn splat_zero() -> Self::Simd {
        i32x8::splat(0)
    }

    #[inline(always)]
    fn splat_one() -> Self::Simd {
        i32x8::splat(1)
    }

    #[inline(always)]
    fn splat_scalar(value: Self::Scalar) -> Self::Simd {
        i32x8::splat(value)
    }

    #[inline(always)]
    fn scalar_from_i64(value: i64) -> Self::Scalar {
        value.try_into().expect("value does not fit in i32")
    }

    #[inline(always)]
    fn scalar_to_i64(value: Self::Scalar) -> i64 {
        value as i64
    }

    #[inline(always)]
    fn mask_word_to_scalar(word: u64) -> Self::Scalar {
        word as i32
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
        i32x8::new(arr)
    }

    #[inline(always)]
    fn min(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        lhs.min(rhs)
    }

    #[inline(always)]
    fn blend(mask: Self::Simd, t: Self::Simd, f: Self::Simd) -> Self::Simd {
        mask.blend(t, f)
    }
}

/// Backend using 64-bit limbs with four SIMD lanes, useful for very long queries.
#[derive(Clone, Copy, Debug, Default)]
pub struct I64x4Backend;

impl SimdBackend for I64x4Backend {
    type Simd = i64x4;
    type Scalar = i64;
    type LaneArray = [i64; 4];

    const LANES: usize = 4;
    const LIMB_BITS: usize = 64;
    const CARRY_SHIFT: u32 = 63;
    const MAX_POSITIVE: Self::Scalar = i64::MAX;

    #[inline(always)]
    fn splat_all_ones() -> Self::Simd {
        i64x4::splat(!0)
    }

    #[inline(always)]
    fn splat_zero() -> Self::Simd {
        i64x4::splat(0)
    }

    #[inline(always)]
    fn splat_one() -> Self::Simd {
        i64x4::splat(1)
    }

    #[inline(always)]
    fn splat_scalar(value: Self::Scalar) -> Self::Simd {
        i64x4::splat(value)
    }

    #[inline(always)]
    fn scalar_from_i64(value: i64) -> Self::Scalar {
        value
    }

    #[inline(always)]
    fn scalar_to_i64(value: Self::Scalar) -> i64 {
        value
    }

    #[inline(always)]
    fn mask_word_to_scalar(word: u64) -> Self::Scalar {
        word as i64
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
        i64x4::new(arr)
    }

    #[inline(always)]
    fn min(lhs: Self::Simd, rhs: Self::Simd) -> Self::Simd {
        lhs.min(rhs)
    }

    #[inline(always)]
    fn blend(mask: Self::Simd, t: Self::Simd, f: Self::Simd) -> Self::Simd {
        mask.blend(t, f)
    }
}

// Not sure, maybe U32 vs u32 is a bit confusing?
// maybe, like 32bits? or something
pub type U32 = I32x8Backend;
pub type U64 = I64x4Backend;
pub type DefaultBackend = U32;
