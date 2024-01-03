use std::cmp::Ordering;
use std::io::{Read, Write};

use anyhow::Result;
use sucds::Searial;

use crate::trie_array::TrieArray;

/// Spece-efficient implementation of [`TrieArray`] with Elias-Fano encording.
#[derive(Default)]
pub struct EliasFanoTrieArray {
    token_ids: sucds::EliasFano,
    pointers: sucds::EliasFano,
}

impl TrieArray for EliasFanoTrieArray {
    fn build(token_ids: Vec<usize>, pointers: Vec<usize>) -> Self {
        if token_ids.is_empty() {
            return Self::default();
        }

        let token_ids = Self::build_token_sequence(token_ids, &pointers);
        let pointers = Self::build_pointers(pointers);

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
        let token_ids = sucds::EliasFano::deserialize_from(&mut reader)?;
        let pointers = sucds::EliasFano::deserialize_from(&mut reader)?;
        Ok(Self {
            token_ids,
            pointers,
        })
    }

    fn size_in_bytes(&self) -> usize {
        self.token_ids.size_in_bytes() + self.pointers.size_in_bytes()
    }

    fn memory_statistics(&self) -> serde_json::Value {
        let token_ids = self.token_ids.size_in_bytes();
        let pointers = self.pointers.size_in_bytes();
        serde_json::json!({
            "token_ids": token_ids,
            "pointers": pointers,
        })
    }

    /// Gets the token id with a given index.
    fn token_id(&self, i: usize) -> usize {
        let pos = self.pointers.rank(i + 1) - 1;
        let (b, _) = self.range(pos);
        let base = if b == 0 {
            0
        } else {
            self.token_ids.select(b - 1)
        };
        self.token_ids.select(i) - base
    }

    #[inline(always)]
    fn range(&self, pos: usize) -> (usize, usize) {
        (self.pointers.select(pos), self.pointers.select(pos + 1))
    }

    /// Searches for an element within a given range, returning its index.
    /// TODO: Make faster
    #[inline(always)]
    fn find_token(&self, pos: usize, id: usize) -> Option<usize> {
        let (b, e) = self.range(pos);
        let base = if b == 0 {
            0
        } else {
            self.token_ids.select(b - 1)
        };
        for i in b..e {
            let token_id = self.token_ids.select(i) - base;
            match token_id.cmp(&id) {
                Ordering::Equal => return Some(i),
                Ordering::Greater => break,
                _ => {}
            }
        }
        None
    }

    fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }

    fn num_pointers(&self) -> usize {
        self.pointers.len()
    }
}

impl EliasFanoTrieArray {
    fn build_token_sequence(mut token_ids: Vec<usize>, pointers: &[usize]) -> sucds::EliasFano {
        assert_eq!(token_ids.len(), *pointers.last().unwrap());

        let mut sampled_id = 0;
        for i in 0..pointers.len() - 1 {
            let (b, e) = (pointers[i], pointers[i + 1]);
            debug_assert!(b <= e);

            for token_id in token_ids.iter_mut().take(e).skip(b) {
                *token_id += sampled_id;
            }
            if e != 0 {
                sampled_id = token_ids[e - 1];
            }
        }

        let mut token_efb = sucds::EliasFanoBuilder::new(sampled_id + 1, token_ids.len()).unwrap();
        token_efb.append(&token_ids).unwrap();
        token_efb.build()
    }

    fn build_pointers(pointers: Vec<usize>) -> sucds::EliasFano {
        let mut pointer_efb =
            sucds::EliasFanoBuilder::new(pointers.last().unwrap() + 1, pointers.len()).unwrap();
        pointer_efb.append(&pointers).unwrap();
        pointer_efb.build().enable_rank()
    }
}
