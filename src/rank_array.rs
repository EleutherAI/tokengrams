pub mod ef;
pub mod simple;

use anyhow::Result;

pub use crate::rank_array::ef::EliasFanoRankArray;
pub use crate::rank_array::simple::SimpleRankArray;

/// Trait for a data structure for storing count ranks.
pub trait RankArray {
    /// Builds a [`RankArray`] from a sequence of count ranks.
    fn build(count_ranks: Vec<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Gets the `i`-th count rank.
    fn get(&self, i: usize) -> Option<usize>;

    /// Gets the number of count ranks stored.
    fn len(&self) -> usize;

    /// Checks if the data structure is empty.
    fn is_empty(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_basic<A: RankArray>() {
        let count_ranks = vec![3, 0, 0, 0, 1, 2, 0, 1, 1];
        let ra = A::build(count_ranks.clone()).unwrap();
        for (i, &x) in count_ranks.iter().enumerate() {
            assert_eq!(ra.get(i), Some(x));
        }
    }

    #[test]
    fn test_basic_simple() {
        test_basic::<SimpleRankArray>();
    }
}
