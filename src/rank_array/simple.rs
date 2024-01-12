use anyhow::Result;

use crate::rank_array::RankArray;

/// Simple implementation of [`RankArray`] with `Vec<usize>`.
#[derive(Default, Debug)]
pub struct SimpleRankArray {
    count_ranks: Vec<usize>,
}

impl RankArray for SimpleRankArray {
    fn build(count_ranks: Vec<usize>) -> Result<Self> {
        Ok(Self { count_ranks })
    }

    fn get(&self, i: usize) -> Option<usize> {
        self.count_ranks.get(i).copied()
    }

    fn len(&self) -> usize {
        self.count_ranks.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
