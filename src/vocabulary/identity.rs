use std::io::{Read, Write};

use anyhow::Result;

use crate::vocabulary::Vocabulary;
use crate::gram::{Gram, TokenGram};

#[derive(Default, Debug)]
pub struct IdentityVocabulary {}

impl Vocabulary for IdentityVocabulary {
    type GramType = TokenGram;

    fn new() -> Self {
        Self { }
    }

    fn build<T: IntoIterator<Item = Self::GramType>>(_: T) -> Result<Self> {
        Ok(Self { })
    }

    fn serialize_into<W>(&self, _: W) -> Result<usize> where W: Write,
    { Ok(0) }

    fn deserialize_from<R>(_: R) -> Result<Self> where R: Read,
    { Ok(Self { }) }

    fn size_in_bytes(&self) -> usize { 0 }

    fn memory_statistics(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    fn get(&self, token: TokenGram) -> Option<usize> {
        token.into_boxed_slice().get(0).map(|&x| x as usize)
    }
}