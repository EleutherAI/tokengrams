use anyhow::Result;
use serde::{Deserialize, Serialize};
use sucds::int_vectors::{Access, PrefixSummedEliasFano};

use crate::rank_array::RankArray;
use crate::sucds_glue;

/// Spece-efficient implementation of [`RankArray`] with Elias-Fano gapped encording.
#[derive(Default, Deserialize, Serialize)]
pub struct EliasFanoRankArray {
    #[serde(with = "sucds_glue")]
    count_ranks: PrefixSummedEliasFano,
}

impl RankArray for EliasFanoRankArray {
    fn build(count_ranks: Vec<usize>) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            count_ranks: PrefixSummedEliasFano::from_slice(&count_ranks)?,
        })
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Option<usize> {
        self.count_ranks.access(i)
    }

    fn len(&self) -> usize {
        self.count_ranks.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
