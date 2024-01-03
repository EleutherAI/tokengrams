mod simple;
mod yada;

use std::io::{Read, Write};

use anyhow::Result;

pub use crate::vocabulary::{simple::SimpleVocabulary, yada::DoubleArrayVocabulary};
use crate::Gram;

/// Trait for a data structure for mapping tokens to unique identifiers.
pub trait Vocabulary {
    /// Creates an empty [`Vocabulary`].
    fn new() -> Self;

    /// Builds a [`Vocabulary`] from a sequence of tokens.
    fn build(tokens: &[Gram<u8>]) -> Result<Self>
    where
        Self: Sized;

    /// Serializes the data structure into the writer.
    fn serialize_into<W: Write>(&self, writer: W) -> Result<usize>;

    /// Deserializes the data structure from the reader.
    fn deserialize_from<R: Read>(reader: R) -> Result<Self>
    where
        Self: Sized;

    /// Gets the number of bytes to serialize the data structure.
    fn size_in_bytes(&self) -> usize;

    /// Gets breakdowns of memory usages for components.
    fn memory_statistics(&self) -> serde_json::Value;

    /// Looks up a token.
    fn get(&self, token: Gram<u8>) -> Option<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let grams = vec![
            Gram::from_str("A"),
            Gram::from_str("D"),
            Gram::from_str("B"),
        ];

        let vocab = SimpleVocabulary::build(&grams).unwrap();
        assert_eq!(vocab.get(Gram::from_str("A")), Some(0));
        assert_eq!(vocab.get(Gram::from_str("B")), Some(2));
        assert_eq!(vocab.get(Gram::from_str("C")), None);
        assert_eq!(vocab.get(Gram::from_str("D")), Some(1));

        let vocab = DoubleArrayVocabulary::build(&grams).unwrap();
        assert_eq!(vocab.get(Gram::from_str("A")), Some(0));
        assert_eq!(vocab.get(Gram::from_str("B")), Some(2));
        assert_eq!(vocab.get(Gram::from_str("C")), None);
        assert_eq!(vocab.get(Gram::from_str("D")), Some(1));
    }
}
