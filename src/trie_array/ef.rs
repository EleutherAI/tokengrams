use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use sucds::mii_sequences::{EliasFano, EliasFanoBuilder};

use crate::sucds_glue;
use crate::trie_array::TrieArray;

/// Spece-efficient implementation of [`TrieArray`] with Elias-Fano encording.
#[derive(Default, Deserialize, Serialize)]
pub struct EliasFanoTrieArray {
    #[serde(with = "sucds_glue")]
    token_ids: EliasFano,

    #[serde(with = "sucds_glue")]
    pointers: EliasFano,
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

    /// Gets the token id with a given index.
    fn token_id(&self, i: usize) -> Option<usize> {
        let pos = self.pointers.rank(i + 1)? - 1;
        let (b, _) = self.range(pos)?;
        let base = if b == 0 {
            0
        } else {
            self.token_ids.select(b - 1)?
        };
        Some(self.token_ids.select(i)? - base)
    }

    #[inline(always)]
    fn range(&self, pos: usize) -> Option<(usize, usize)> {
        Some((self.pointers.select(pos)?, self.pointers.select(pos + 1)?))
    }

    /// Searches for an element within a given range, returning its index.
    /// TODO: Make faster
    #[inline(always)]
    fn find_token(&self, pos: usize, id: usize) -> Option<usize> {
        let (b, e) = self.range(pos)?;
        let base = if b == 0 {
            0
        } else {
            self.token_ids.select(b - 1)?
        };
        for i in b..e {
            let token_id = self.token_ids.select(i)? - base;
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
    fn build_token_sequence(mut token_ids: Vec<usize>, pointers: &[usize]) -> EliasFano {
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

        let mut token_efb = EliasFanoBuilder::new(sampled_id + 1, token_ids.len()).unwrap();
        token_efb.extend(token_ids).unwrap();
        token_efb.build()
    }

    fn build_pointers(pointers: Vec<usize>) -> EliasFano {
        let mut pointer_efb =
            EliasFanoBuilder::new(pointers.last().unwrap() + 1, pointers.len()).unwrap();
        pointer_efb.extend(pointers).unwrap();
        pointer_efb.build().enable_rank()
    }
}
