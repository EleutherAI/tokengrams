use std::io::{Read, Write};

use anyhow::{anyhow, Result};
use sucds::util::VecIO;
use yada::{builder::DoubleArrayBuilder, DoubleArray};

use crate::vocabulary::Vocabulary;
use crate::Gram;

/// Compact double-array implementation of [`Vocabulary`].
#[derive(Default, Debug)]
pub struct DoubleArrayVocabulary {
    data: Vec<u8>,
}

impl Vocabulary for DoubleArrayVocabulary {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn build(tokens: &[Gram<u8>]) -> Result<Self> {
        if (tokens.len() >> 31) != 0 {
            return Err(anyhow!(
                "The number of tokens must be represented in 31 bits."
            ));
        }

        let mut keyset = vec![];
        for (id, token) in tokens.iter().enumerate() {
            keyset.push((token.raw(), id as u32));
        }
        keyset.sort_by(|(g1, _), (g2, _)| g1.cmp(g2));

        for i in 1..keyset.len() {
            if keyset[i - 1].0 == keyset[i].0 {
                let (k, v) = keyset[i - 1];
                return Err(anyhow!("Duplicated key: {:?} => {}", k, v));
            }
        }

        Ok(Self {
            data: DoubleArrayBuilder::build(&keyset[..]).unwrap(),
        })
    }

    fn serialize_into<W>(&self, writer: W) -> Result<usize>
    where
        W: Write,
    {
        self.data.serialize_into(writer)
    }

    fn deserialize_from<R>(reader: R) -> Result<Self>
    where
        R: Read,
    {
        let data = Vec::<u8>::deserialize_from(reader)?;
        Ok(Self { data })
    }

    fn size_in_bytes(&self) -> usize {
        self.data.size_in_bytes()
    }

    fn memory_statistics(&self) -> serde_json::Value {
        let data = self.data.size_in_bytes();
        serde_json::json!({ "data": data })
    }

    #[inline(always)]
    fn get(&self, token: Gram<u8>) -> Option<usize> {
        let da = DoubleArray::new(&self.data[..]);
        da.exact_match_search(token.raw()).map(|x| x as usize)
    }
}
