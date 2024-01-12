use anyhow::{anyhow, Result};
use yada::{builder::DoubleArrayBuilder, DoubleArray};

use crate::vocabulary::Vocabulary;
use crate::{Gram, WordGram};

/// Compact double-array implementation of [`Vocabulary`].
#[derive(Default, Debug)]
pub struct DoubleArrayVocabulary {
    data: Vec<u8>,
}

impl Vocabulary for DoubleArrayVocabulary {
    type GramType = WordGram;

    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn build<T: IntoIterator<Item = Self::GramType>>(tokens: T) -> Result<Self> {
        let mut keyset = vec![];
        for (id, token) in tokens.into_iter().enumerate() {
            keyset.push((token.into_boxed_slice(), id as u32));
        }

        // Sad that we have to check the length down here instead of checking tokens at
        // the top, but this is a very unlikely case and I want to stay generic
        if (keyset.len() >> 31) != 0 {
            return Err(anyhow!(
                "The number of tokens must be represented in 31 bits."
            ));
        }
        keyset.sort_by(|(g1, _), (g2, _)| g1.cmp(g2));

        for i in 1..keyset.len() {
            if keyset[i - 1].0 == keyset[i].0 {
                let (k, v) = &keyset[i - 1];
                return Err(anyhow!("Duplicated key: {:?} => {}", k, v));
            }
        }

        Ok(Self {
            data: DoubleArrayBuilder::build(&keyset[..]).unwrap(),
        })
    }

    #[inline(always)]
    fn get(&self, token: WordGram) -> Option<usize> {
        let da = DoubleArray::new(&self.data[..]);
        da.exact_match_search(token.into_boxed_slice())
            .map(|x| x as usize)
    }
}
