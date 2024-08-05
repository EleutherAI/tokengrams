use anyhow::Result;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::sample::{KneserNeyCache, Sample};
use crate::{MemmapIndexU16, MemmapIndexU32};

/// Expose suffix table functionality over text corpora too large to fit in memory.
macro_rules! create_interface {
    ($name: ident, $type: ident, $memmap_type: ident) => {
        #[pyclass]
        pub struct $name {
            shards: Vec<$memmap_type>,
            cache: KneserNeyCache,
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(_py: Python, files: Vec<(String, String)>) -> PyResult<Self> {
                let shards: Vec<$memmap_type> = files
                    .into_iter()
                    .map(|(text_path, table_path)| $memmap_type::new(_py, text_path, table_path).unwrap())
                    .collect();
        
                Ok($name {
                    shards,
                    cache: KneserNeyCache::default(),
                })
            }
        
            #[staticmethod]
            #[pyo3(signature = (paths, verbose=false))]
            pub fn build(paths: Vec<(String, String)>, verbose: bool) -> PyResult<Self> {
                let shards: Vec<$memmap_type> = paths
                    .into_iter()
                    .map(|(token_paths, index_paths)| {
                        $memmap_type::build(token_paths, index_paths, verbose).unwrap()
                    })
                    .collect();
        
                Ok($name {
                    shards,
                    cache: KneserNeyCache::default(),
                })
            }
        
            pub fn is_sorted(&self) -> bool {
                self.shards.iter().all(|shard| shard.is_sorted())
            }
        
            pub fn contains(&self, query: Vec<$type>) -> bool {
                self.shards
                    .iter()
                    .any(|shard| shard.contains(query.clone()))
            }
        
            pub fn count(&self, query: Vec<$type>) -> usize {
                self.shards
                    .iter()
                    .map(|shard| shard.count(query.clone()))
                    .sum()
            }
        
            #[pyo3(signature = (query, vocab=None))]
            pub fn count_next(&self, query: Vec<$type>, vocab: Option<usize>) -> Vec<usize> {
                let counts = self
                    .shards
                    .iter()
                    .map(|shard| shard.count_next_slice(&query, vocab))
                    .collect::<Vec<_>>();
                (0..counts[0].len())
                    .map(|i| counts.iter().map(|count| count[i]).sum())
                    .collect()
            }
        
            #[pyo3(signature = (queries, vocab=None))]
            pub fn batch_count_next(&self, queries: Vec<Vec<$type>>, vocab: Option<usize>) -> Vec<Vec<usize>> {
                let batch_counts = self
                    .shards
                    .iter()
                    .map(|shard| shard.batch_count_next(queries.clone(), vocab))
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
            #[pyo3(signature = (query, n, k, num_samples, vocab=None))]
            pub fn sample_unsmoothed(
                &self,
                query: Vec<$type>,
                n: usize,
                k: usize,
                num_samples: usize,
                vocab: Option<usize>,
            ) -> Result<Vec<Vec<$type>>> {
                self.sample_unsmoothed_rs(&query, n, k, num_samples, vocab)
            }
        
            /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
            /// tokens in the query.
            #[pyo3(signature = (query, vocab=None))]
            pub fn get_smoothed_probs(&mut self, query: Vec<$type>, vocab: Option<usize>) -> Vec<f64> {
                self.get_smoothed_probs_rs(&query, vocab)
            }
        
            /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
            /// tokens in the query.
            #[pyo3(signature = (queries, vocab=None))]
            pub fn batch_get_smoothed_probs(
                &mut self,
                queries: Vec<Vec<$type>>,
                vocab: Option<usize>,
            ) -> Vec<Vec<f64>> {
                self.batch_get_smoothed_probs_rs(&queries, vocab)
            }
        
            /// Autoregressively sample num_samples of k characters from a Kneser-Ney smoothed n-gram model.
            #[pyo3(signature = (query, n, k, num_samples, vocab=None))]
            pub fn sample_smoothed(
                &mut self,
                query: Vec<$type>,
                n: usize,
                k: usize,
                num_samples: usize,
                vocab: Option<usize>,
            ) -> Result<Vec<Vec<$type>>> {
                self.sample_smoothed_rs(&query, n, k, num_samples, vocab)
            }
    
            /// Warning: O(k**n) where k is vocabulary size, use with caution.
            /// Improve smoothed model quality by replacing the default delta hyperparameters
            /// for models of order n and below with improved estimates over the entire index.
            /// https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf, page 16."""
            pub fn estimate_deltas(&mut self, n: usize) {
                self.estimate_deltas_rs(n);
            }
        }
        
        impl Sample<$type> for $name {
            fn get_cache(&self) -> &KneserNeyCache {
                &self.cache
            }
        
            fn get_mut_cache(&mut self) -> &mut KneserNeyCache {
                &mut self.cache
            }
        
            fn count_next_slice(&self, query: &[$type], vocab: Option<usize>) -> Vec<usize> {
                let counts = self
                    .shards
                    .iter()
                    .map(|shard| shard.count_next_slice(query, vocab))
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
    };
}

create_interface!(ShardedMemmapIndexU16, u16, MemmapIndexU16);
create_interface!(ShardedMemmapIndexU32, u32, MemmapIndexU32);
