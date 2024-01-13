mod identity;
mod yada;

use anyhow::Result;

pub use crate::vocabulary::{
    identity::IdentityVocabulary, yada::DoubleArrayVocabulary,
};
use crate::Gram;

/// Trait for a data structure for mapping tokens to unique identifiers.
pub trait Vocabulary {
    type GramType: Gram;

    /// Creates an empty [`Vocabulary`].
    fn new() -> Self;

    /// Builds a [`Vocabulary`] from a sequence of tokens.
    fn build<T: IntoIterator<Item = Self::GramType>>(tokens: T) -> Result<Self>
    where
        Self: Sized;

    /// Looks up a token.
    fn get(&self, token: Self::GramType) -> Option<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WordGram;

    #[test]
    fn test_basic() {
        let grams = [
            WordGram::from_str("A"),
            WordGram::from_str("D"),
            WordGram::from_str("B"),
        ];

        let vocab = DoubleArrayVocabulary::build(grams).unwrap();
        assert_eq!(vocab.get(WordGram::from_str("A")), Some(0));
        assert_eq!(vocab.get(WordGram::from_str("B")), Some(2));
        assert_eq!(vocab.get(WordGram::from_str("C")), None);
        assert_eq!(vocab.get(WordGram::from_str("D")), Some(1));
    }
}
