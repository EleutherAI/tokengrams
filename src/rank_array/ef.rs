use std::io::{Read, Write};

use anyhow::Result;
use sucds::Searial;

use crate::rank_array::RankArray;

/// Spece-efficient implementation of [`RankArray`] with Elias-Fano gapped encording.
#[derive(Default)]
pub struct EliasFanoRankArray {
    count_ranks: sucds::EliasFanoList,
}

impl RankArray for EliasFanoRankArray {
    fn build(count_ranks: Vec<usize>) -> Self {
        Self {
            count_ranks: sucds::EliasFanoList::from_slice(&count_ranks).unwrap(),
        }
    }

    fn serialize_into<W>(&self, mut writer: W) -> Result<usize>
    where
        W: Write,
    {
        self.count_ranks.serialize_into(&mut writer)
    }

    fn deserialize_from<R>(mut reader: R) -> Result<Self>
    where
        R: Read,
    {
        let count_ranks = sucds::EliasFanoList::deserialize_from(&mut reader)?;
        Ok(Self { count_ranks })
    }

    fn size_in_bytes(&self) -> usize {
        self.count_ranks.size_in_bytes()
    }

    fn memory_statistics(&self) -> serde_json::Value {
        let count_ranks = self.count_ranks.size_in_bytes();
        serde_json::json!({ "count_ranks": count_ranks })
    }

    #[inline(always)]
    fn get(&self, i: usize) -> usize {
        self.count_ranks.get(i)
    }

    fn len(&self) -> usize {
        self.count_ranks.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
