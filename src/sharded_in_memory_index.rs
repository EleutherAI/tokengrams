use anyhow::Result;
use funty::Unsigned;
use std::collections::HashMap;
use rayon::prelude::*;

use crate::in_memory_index::InMemoryIndexRs;
use crate::sample::{KneserNeyCache, Sample};
use crate::bindings::sharded_in_memory_index::ShardedInMemoryIndexTrait;
use crate::bindings::in_memory_index::InMemoryIndexTrait;

/// Expose suffix table functionality over text corpora too large to fit in memory.
pub struct ShardedInMemoryIndexRs<T: Unsigned> {
    shards: Vec<InMemoryIndexRs<T>>,
    cache: KneserNeyCache,
}

impl<T: Unsigned> Sample<T> for ShardedInMemoryIndexRs<T> {
    fn get_cache(&self) -> &KneserNeyCache {
        &self.cache
    }

    fn get_mut_cache(&mut self) -> &mut KneserNeyCache {
        &mut self.cache
    }

    fn count_next_slice(&self, query: &[T]) -> Vec<usize> {
        let counts = self
            .shards
            .par_iter()
            .map(|shard| shard.count_next_slice(query))
            .collect::<Vec<_>>();
        (0..counts[0].len())
            .map(|i| counts.iter().map(|count| count[i]).sum())
            .collect()
    }

    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        self.shards.iter().map(|shard| shard.count_ngrams(n)).fold(
            HashMap::new(),
            |mut acc, counts| {
                for (k, v) in counts {
                    *acc.entry(k).or_insert(0) += v;
                }
                acc
            },
        )
    }
}

impl<T: Unsigned> ShardedInMemoryIndexRs<T> {
    pub fn new(paths: Vec<(String, String)>, vocab: usize) -> Result<Self> {
        let shards: Vec<InMemoryIndexRs<T>> = paths
            .into_iter()
            .map(|(text_path, table_path)| {
                InMemoryIndexRs::from_disk(text_path, table_path, vocab).unwrap()
            })
            .collect();

        Ok(ShardedInMemoryIndexRs {
            shards,
            cache: KneserNeyCache::default(),
        })
    }
}

impl<T: Unsigned> ShardedInMemoryIndexTrait for ShardedInMemoryIndexRs<T> {
    fn is_sorted(&self) -> bool {
        self.shards.iter().all(|shard| shard.is_sorted())
    }

    fn contains(&self, query: Vec<usize>) -> bool {
        self.shards
            .iter()
            .any(|shard| shard.contains(query.clone()))
    }

    fn count(&self, query: Vec<usize>) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.count(query.clone()))
            .sum()
    }

    fn count_next(&self, query: Vec<usize>) -> Vec<usize> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();

        let counts = self
            .shards
            .iter()
            .map(|shard| shard.count_next_slice(&query))
            .collect::<Vec<_>>();
        (0..counts[0].len())
            .map(|i| counts.iter().map(|count| count[i]).sum())
            .collect()
    }

    fn batch_count_next(&self, queries: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let batch_counts = self
            .shards
            .iter()
            .map(|shard| shard.batch_count_next(queries.clone()))
            .collect::<Vec<_>>();

        (0..queries.len())
            .map(|i| {
                (0..batch_counts[0][i].len())
                    .map(|j| batch_counts.iter().map(|count| count[i][j]).sum())
                    .collect()
            })
            .collect()
    }

    /// Autoregressively sample num_samples of k characters from an unsmoothed n-gram model."""
    fn sample_unsmoothed(
        &self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();

        let samples_batch =
            <Self as Sample<T>>::sample_unsmoothed(self, &query, n, k, num_samples)?;
        Ok(samples_batch
            .into_iter()
            .map(|samples| {
                samples
                    .into_iter()
                    .filter_map(|sample| {
                        match TryInto::<usize>::try_into(sample) {
                            Ok(value) => Some(value),
                            Err(_) => None, // Silently skip values that can't be converted
                        }
                    })
                    .collect::<Vec<usize>>()
            })
            .collect())
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    fn get_smoothed_probs(&mut self, query: Vec<usize>) -> Vec<f64> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        <Self as Sample<T>>::get_smoothed_probs(self, &query)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    fn batch_get_smoothed_probs(&mut self, queries: Vec<Vec<usize>>) -> Vec<Vec<f64>> {
        let queries: Vec<Vec<T>> = queries
            .into_iter()
            .map(|query| {
                query
                    .iter()
                    .filter_map(|&item| T::try_from(item).ok())
                    .collect()
            })
            .collect();
        <Self as Sample<T>>::batch_get_smoothed_probs(self, &queries)
    }

    /// Autoregressively sample num_samples of k characters from a Kneser-Ney smoothed n-gram model.
    fn sample_smoothed(
        &mut self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();

        let samples_batch = <Self as Sample<T>>::sample_smoothed(self, &query, n, k, num_samples)?;
        Ok(samples_batch
            .into_iter()
            .map(|samples| {
                samples
                    .into_iter()
                    .filter_map(|sample| {
                        match TryInto::<usize>::try_into(sample) {
                            Ok(value) => Some(value),
                            Err(_) => None, // Silently skip values that can't be converted
                        }
                    })
                    .collect::<Vec<usize>>()
            })
            .collect())
    }

    /// Warning: O(k**n) where k is vocabulary size, use with caution.
    /// Improve smoothed model quality by replacing the default delta hyperparameters
    /// for models of order n and below with improved estimates over the entire index.
    /// <https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf/>, page 16.
    fn estimate_deltas(&mut self, n: usize) {
        <Self as Sample<T>>::estimate_deltas(self, n);
    }
}
