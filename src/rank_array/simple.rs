use std::io::{Read, Write};

use anyhow::Result;
use sucds::util::VecIO;

use crate::rank_array::RankArray;

/// Simple implementation of [`RankArray`] with `Vec<usize>`.
#[derive(Default, Debug)]
pub struct SimpleRankArray {
    count_ranks: Vec<usize>,
}

impl RankArray for SimpleRankArray {
    fn build(count_ranks: Vec<usize>) -> Self {
        Self { count_ranks }
    }

    fn serialize_into<W>(&self, writer: W) -> Result<usize>
    where
        W: Write,
    {
        self.count_ranks.serialize_into(writer)
    }

    fn deserialize_from<R>(reader: R) -> Result<Self>
    where
        R: Read,
    {
        let count_ranks = Vec::<usize>::deserialize_from(reader)?;
        Ok(Self { count_ranks })
    }

    fn size_in_bytes(&self) -> usize {
        self.count_ranks.size_in_bytes()
    }

    fn memory_statistics(&self) -> serde_json::Value {
        let count_ranks = self.count_ranks.size_in_bytes();
        serde_json::json!({ "count_ranks": count_ranks })
    }

    fn get(&self, i: usize) -> usize {
        self.count_ranks[i]
    }

    fn len(&self) -> usize {
        self.count_ranks.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
