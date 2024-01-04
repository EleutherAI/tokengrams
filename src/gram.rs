pub mod token_gram;
pub mod word_gram;

pub use crate::gram::token_gram::TokenGram;
pub use crate::gram::word_gram::WordGram;

use std::boxed::Box;

/// Trait shared by [`TokenGram`] and [`WordGram`]. Grams are glorified Boxed slices.
pub trait Gram: Clone + Eq {
    /// Primitive type for individual characters or tokens.
    type CharType: Clone + Copy;

    /// Creates a [`Gram`] from a byte slice, consuming the slice.
    fn new<T: Into<Box<[Self::CharType]>>>(data: T) -> Self;

    /// Gets the reference to the byte slice.
    fn into_boxed_slice(self) -> Box<[Self::CharType]>;

    /// Pops the last token.
    fn pop_token(&self) -> Option<(Self, Self)>;

    /// Pops the first token.
    fn pop_front_token(&self) -> Option<(Self, Self)>;

    /// Splits the gram into tokens.
    fn to_unigrams(&self) -> Vec<Self>;
}
