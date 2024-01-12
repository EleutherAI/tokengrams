use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::gram::{Gram, TokenGram};
use crate::vocabulary::Vocabulary;

#[derive(Default, Debug, Deserialize, Serialize)]
pub struct IdentityVocabulary {}

impl Vocabulary for IdentityVocabulary {
    type GramType = TokenGram;

    fn new() -> Self {
        Self {}
    }

    fn build<T: IntoIterator<Item = Self::GramType>>(_: T) -> Result<Self> {
        Ok(Self {})
    }

    fn get(&self, token: TokenGram) -> Option<usize> {
        token.into_boxed_slice().get(0).map(|&x| x as usize)
    }
}
