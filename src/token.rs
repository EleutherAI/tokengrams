use funty::Unsigned;
use std::fmt::Debug;

const USIZE_IS_32BIT_OR_LARGER: bool = std::mem::size_of::<usize>() >= 4;

pub trait Token: Unsigned + Copy + Sync + Debug {
    fn from_usize(value: usize) -> Self;
    fn to_usize(self) -> usize;
}

impl Token for u16 {
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        assert!(value <= u16::MAX as usize, "Value too large for u16");
        value as u16
    }

    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl Token for u32 {
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        assert!(value <= u32::MAX as usize, "Value too large for u32");
        value as u32
    }

    #[inline(always)]
    fn to_usize(self) -> usize {
        debug_assert!(USIZE_IS_32BIT_OR_LARGER, "System usize is smaller than u32");
        self as usize
    }
}