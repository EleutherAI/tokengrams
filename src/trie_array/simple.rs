use std::io::{Read, Write};

use anyhow::Result;
use sucds::util::VecIO;

use crate::trie_array::TrieArray;

/// Simple implementation of [`TrieArray`] with `Vec<usize>`.
#[derive(Default, Debug)]
pub struct SimpleTrieArray {
    token_ids: Vec<usize>,
    pointers: Vec<usize>,
}

impl TrieArray for SimpleTrieArray {
    fn build(token_ids: Vec<usize>, pointers: Vec<usize>) -> Self {
        Self {
            token_ids,
            pointers,
        }
    }

    fn serialize_into<W>(&self, mut writer: W) -> Result<usize>
    where
        W: Write,
    {
        Ok(self.token_ids.serialize_into(&mut writer)?
            + self.pointers.serialize_into(&mut writer)?)
    }

    fn deserialize_from<R>(mut reader: R) -> Result<Self>
    where
        R: Read,
    {
        let token_ids = Vec::<usize>::deserialize_from(&mut reader)?;
        let pointers = Vec::<usize>::deserialize_from(&mut reader)?;
        Ok(Self {
            token_ids,
            pointers,
        })
    }

    fn size_in_bytes(&self) -> usize {
        self.token_ids.size_in_bytes() + self.pointers.size_in_bytes()
    }

    fn memory_statistics(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    /// Gets the token id with a given index.
    fn token_id(&self, i: usize) -> usize {
        self.token_ids[i]
    }

    fn range(&self, pos: usize) -> (usize, usize) {
        (self.pointers[pos], self.pointers[pos + 1])
    }

    fn find_token(&self, pos: usize, id: usize) -> Option<usize> {
        let (b, e) = self.range(pos);
        self.token_ids[b..e]
            .iter()
            .position(|&x| x == id)
            .map(|i| i + b)
    }

    fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }

    fn num_pointers(&self) -> usize {
        self.pointers.len()
    }
}
