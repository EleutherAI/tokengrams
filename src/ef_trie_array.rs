use anyhow::Result;
use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use sucds::mii_sequences::{EliasFano, EliasFanoBuilder};

use crate::sucds_glue;

/// Spece-efficient implementation of [`TrieArray`] with Elias-Fano encording.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct EliasFanoTrieArray {
    #[serde(with = "sucds_glue")]
    token_ids: EliasFano,

    #[serde(with = "sucds_glue")]
    pointers: EliasFano,
}

impl EliasFanoTrieArray {
    pub fn build(mut token_ids: Vec<usize>, pointers: Vec<usize>) -> Result<Self> {
        // TODO: Should we actually allow this?
        if token_ids.is_empty() {
            return Ok(Self::default());
        }
        let last_ptr = *pointers.last().unwrap();
        assert_eq!(token_ids.len(), last_ptr);

        let mut sampled_id = 0;
        for (b, e) in pointers.iter().zip(pointers.iter().skip(1)) {
            debug_assert!(b <= e);

            for token_id in token_ids.iter_mut().take(*e).skip(*b) {
                *token_id += sampled_id;
            }
            if *e != 0 {
                sampled_id = token_ids[e - 1];
            }
        }

        let mut token_efb = EliasFanoBuilder::new(sampled_id + 1, token_ids.len())?;
        token_efb.extend(token_ids)?;

        let mut pointer_efb =
            EliasFanoBuilder::new(last_ptr + 1, pointers.len())?;
        pointer_efb.extend(pointers)?;

        Ok(Self {
            token_ids: token_efb.build(),
            pointers: pointer_efb.build().enable_rank(),
        })
    }

    /// Gets the token id with a given index.
    pub fn token_id(&self, i: usize) -> Option<usize> {
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
    pub fn range(&self, pos: usize) -> Option<(usize, usize)> {
        Some((self.pointers.select(pos)?, self.pointers.select(pos + 1)?))
    }

    /// Searches for an element within a given range, returning its index.
    /// TODO: Make faster
    #[inline(always)]
    pub fn find_token(&self, pos: usize, id: usize) -> Option<usize> {
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

    pub fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }

    pub fn num_pointers(&self) -> usize {
        self.pointers.len()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_1() {
        let token_ids = vec![0, 2, 1, 2, 3, 0, 3, 1, 3];
        let pointers = vec![0, 2, 5, 7, 9];
        let ta = EliasFanoTrieArray::build(token_ids.clone(), pointers.clone()).unwrap();

        for (i, &x) in token_ids.iter().enumerate() {
            assert_eq!(ta.token_id(i), Some(x));
        }
        for i in 0..pointers.len() - 1 {
            assert_eq!(ta.range(i).unwrap(), (pointers[i], pointers[i + 1]));
        }

        assert_eq!(ta.find_token(1, 3), Some(4));
        assert_eq!(ta.find_token(1, 1), Some(2));
        assert_eq!(ta.find_token(1, 4), None);

        assert_eq!(ta.num_tokens(), 9);
        assert_eq!(ta.num_pointers(), 5);
    }

    #[test]
    fn test_basic_2() {
        let token_ids = vec![2, 2, 3, 3, 1, 2, 3];
        let pointers = vec![0, 1, 1, 3, 4, 4, 4, 4, 6, 7];
        let ta = EliasFanoTrieArray::build(token_ids.clone(), pointers.clone()).unwrap();

        for (i, &x) in token_ids.iter().enumerate() {
            assert_eq!(ta.token_id(i), Some(x));
        }
        for i in 0..pointers.len() - 1 {
            assert_eq!(ta.range(i).unwrap(), (pointers[i], pointers[i + 1]));
        }

        assert_eq!(ta.find_token(2, 2), Some(1));
        assert_eq!(ta.find_token(2, 3), Some(2));
        assert_eq!(ta.find_token(2, 4), None);

        assert_eq!(ta.num_tokens(), 7);
        assert_eq!(ta.num_pointers(), 10);
    }
}
