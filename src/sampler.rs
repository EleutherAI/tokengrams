extern crate utf16_literal;

use pyo3::prelude::*;
use pyo3::{Py, Python};
use anyhow::Result;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{ops::Mul, u64};

use crate::countable::CountableIndex;

#[pyclass]
pub struct Sampler {
    index: Py<CountableIndex>,
    cache: KneserNeyCache,
}

#[derive(Clone, Deserialize, Serialize)]
struct KneserNeyCache {
    unigram_probs: Option<Vec<f64>>,
    n_delta: HashMap<usize, f64>,
}

#[pymethods]
impl Sampler {
    #[new]
    pub fn new_py(_py: Python, index: Py<CountableIndex>) -> Self {
        Sampler {
            index: index,
            cache: KneserNeyCache {
                unigram_probs: None,
                n_delta: HashMap::new(),
            },
        }
    }

    // pub fn sampleable_index(index: SampleableIndex) -> Self {
    //     Sampler {
    //         index: index,
    //         cache: KneserNeyCache {
    //             unigram_probs: None,
    //             n_delta: HashMap::new(),
    //         },
    //     }
    // }
    // pub fn memmap_index(index: MemmapIndex) -> Self {
    //     Sampler {
    //         index: SampleableIndex::Memmap(index),
    //         cache: KneserNeyCache {
    //             unigram_probs: None,
    //             n_delta: HashMap::new(),
    //         },
    //     }
    // }
    // pub fn in_memory_index(index: InMemoryIndex) -> Self {
    //     Sampler {
    //         index: SampleableIndex::InMemory(index),
    //         cache: KneserNeyCache {
    //             unigram_probs: None,
    //             n_delta: HashMap::new(),
    //         },
    //     }
    // }
    // pub fn sharded_memmap_index(index: ShardedMemmapIndex) -> Self {
    //     Sampler {
    //         index: SampleableIndex::ShardedMemmap(index),
    //         cache: KneserNeyCache {
    //             unigram_probs: None,
    //             n_delta: HashMap::new(),
    //         },
    //     }
    // }
    // pub fn suffix_table(suffix_table: SuffixTable) -> Self {
    //     Sampler {
    //         index: SampleableIndex::Countable(suffix_table),
    //         cache: KneserNeyCache {
    //             unigram_probs: None,
    //             n_delta: HashMap::new(),
    //         },
    //     }
    // }

    /// Autoregressively sample num_samples of k characters from an unsmoothed n-gram model."""
    pub fn sample_unsmoothed(
        &self,
        query: Vec<u16>,
        n: usize,
        k: usize,
        num_samples: usize,
        vocab: Option<u16>,
    ) -> Result<Vec<Vec<u16>>> {
        (0..num_samples)
            .into_par_iter()
            .map(|_| self.sample(query.clone(), n, k, vocab))
            .collect()
    }

    //// Autoregressively sample a sequence of k characters from an unsmoothed n-gram model."""
    fn sample(&self, query: Vec<u16>, n: usize, k: usize, vocab: Option<u16>) -> Result<Vec<u16>> {
        let mut rng = thread_rng();
        let mut sequence = Vec::from(query);

        for _ in 0..k {
            // look at the previous (n - 1) characters to predict the n-gram completion
            let start = sequence.len().saturating_sub(n - 1);
            let prev = &sequence[start..];
            
            let counts = Python::with_gil(
                |py| self.index.as_ref(py).borrow().count_next_slice(prev, vocab)
            );
            let dist = WeightedIndex::new(&counts)?;
            let sampled_index = dist.sample(&mut rng);

            sequence.push(sampled_index as u16);
        }

        Ok(sequence)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    pub fn get_smoothed_probs(&mut self, query: Vec<u16>, vocab: Option<u16>) -> Vec<f64> {
        self.estimate_deltas(1);
        self.compute_smoothed_unigram_probs(vocab);
        self.smoothed_probs(query, vocab)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    pub fn batch_get_smoothed_probs(&mut self, queries: Vec<Vec<u16>>, vocab: Option<u16>) -> Vec<Vec<f64>> {
        self.estimate_deltas(1);
        self.compute_smoothed_unigram_probs(vocab);

        queries
            .into_par_iter()
            .map(|query| self.smoothed_probs(query, vocab))
            .collect()
    }

    /// Autoregressively sample num_samples of k characters from a Kneser-Ney smoothed n-gram model.
    pub fn sample_smoothed(
        &mut self,
        query: Vec<u16>,
        n: usize,
        k: usize,
        num_samples: usize,
        vocab: Option<u16>,
    ) -> Result<Vec<Vec<u16>>> {
        self.estimate_deltas(1);
        self.compute_smoothed_unigram_probs(vocab);

        (0..num_samples)
            .into_par_iter()
            .map(|_| self.kn_sample(query.clone(), n, k, vocab))
            .collect()
    }
        
    /// Returns the Kneser-Ney smoothed token probability distribution for a query 
    /// continuation using absolute discounting as described in
    /// "On structuring probabilistic dependences in stochastic language modelling", page 25,
    /// doi:10.1006/csla.1994.1001
    fn smoothed_probs(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<f64> {
        let p_continuations = if query.is_empty() {
            self.get_cached_smoothed_unigram_probs().to_vec()
        } else {
            self.smoothed_probs(query[1..].to_vec(), vocab)
        };
        
        let counts = Python::with_gil(|py| self.index.as_ref(py).borrow().count_next_slice(&query, vocab));
        let suffix_count_recip = {
            let suffix_count: usize = counts.iter().sum();
            if suffix_count == 0 {
                return p_continuations;
            }
            1.0 / suffix_count as f64
        };

        let (gt_zero_count, eq_one_count) = self.get_occurrence_counts(counts.clone());
        let used_suffix_count = gt_zero_count as f64;
        let used_once_suffix_count = eq_one_count as f64;

        // Interpolation budget to be distributed according to lower order n-gram distribution
        let delta = self.get_cached_delta(query.len() + 1);
        let lambda = if delta < 1.0 {
            delta.mul(used_suffix_count).mul(suffix_count_recip)
        } else {
            used_once_suffix_count
                + delta
                    .mul(used_suffix_count - used_once_suffix_count)
                    .mul(suffix_count_recip)
        };
        
        let mut probs = Vec::with_capacity(counts.len());
        counts
            .iter()
            .zip(p_continuations.iter())
            .for_each(|(&count, &p_continuation)| {
                let prob =
                    (count as f64 - delta).max(0.0).mul(suffix_count_recip) + lambda.mul(p_continuation);
                probs.push(prob);
            });
        probs
    }

    /// Returns tuple of the number of elements that are greater than 0 and the number of elements that equal 1.
    fn get_occurrence_counts(&self, slice: Vec<usize>) -> (usize, usize) {
        slice
            .iter()
            .fold((0, 0), |(gt_zero_count, eq_one_count), &c| {
                let gt_zero_count = gt_zero_count + (c > 0) as usize;
                let eq_one_count = eq_one_count + (c == 1) as usize;
                (gt_zero_count, eq_one_count)
            })
    }

    /// Autoregressively sample k characters from a Kneser-Ney smoothed n-gram model.
    fn kn_sample(&self, query: Vec<u16>, n: usize, k: usize, vocab: Option<u16>) -> Result<Vec<u16>> {
        let mut rng = thread_rng();
        let mut sequence = Vec::from(query);

        for _ in 0..k {
            let start = sequence.len().saturating_sub(n - 1);
            let prev = &sequence[start..];
            let probs = self.smoothed_probs(prev.to_vec(), vocab);
            let dist = WeightedIndex::new(&probs)?;
            let sampled_index = dist.sample(&mut rng);

            sequence.push(sampled_index as u16);
        }

        Ok(sequence)
    }
    
    /// Warning: O(k**n) where k is vocabulary size, use with caution.
    /// Improve smoothed model quality by replacing the default delta hyperparameters
    /// for models of order n and below with improved estimates over the entire index.
    /// https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf, page 16."""
    pub fn estimate_deltas(&mut self, n: usize) {
        for i in 1..n + 1 {
            if self.cache.n_delta.contains_key(&i) {
                continue;
            }
            
            let count_map = Python::with_gil(
                |py| self.index.as_ref(py).borrow().count_ngrams(i)
            );
            let n1 = *count_map.get(&1).unwrap_or(&0) as f64;
            let n2 = *count_map.get(&2).unwrap_or(&0) as f64;

            // n1 and n2 are greater than 0 for non-trivial datasets
            let delta = if n1 == 0. || n2 == 0. {
                1.
            } else {
                n1 / (n1 + n2.mul(2.))
            };

            self.cache.n_delta.insert(i, delta);
        }
    }

    fn get_cached_delta(&self, n: usize) -> f64 {
        *self.cache.n_delta.get(&n).unwrap_or(&0.5)
    }

    /// Returns unigram probabilities with additive smoothing applied.
    fn compute_smoothed_unigram_probs(&mut self, vocab: Option<u16>) {
        if let Some(_) = &self.cache.unigram_probs {
            return;
        }

        let eps = 1e-9;
        let max_vocab = u16::MAX as usize + 1;
        let vocab_size = match vocab {
            Some(size) => size as usize,
            None => max_vocab,
        };

        // Count the number of unique bigrams that end with each token
        let counts = Python::with_gil(
            |py| self.index.as_ref(py).borrow().count_next_slice(&[], vocab)
        );

        let total_count: usize = counts.iter().sum();
        let adjusted_total_count = total_count as f64 + eps.mul(vocab_size as f64);
        let unigram_probs: Vec<f64> = counts
            .iter()
            .map(|&count| {
                (count as f64 + eps) / adjusted_total_count
            })
            .collect();

        self.cache.unigram_probs = Some(unigram_probs);
    }

    fn get_cached_smoothed_unigram_probs(&self) -> Vec<f64> {
        self.cache.unigram_probs.as_ref().unwrap().clone()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use utf16_literal::utf16;

//     fn sais(text: &str) -> SuffixTable {
//         SuffixTable::new(text.encode_utf16().collect::<Vec<_>>(), false)
//     }
//     #[test]
//     fn unigram_probs_exists() {
//         let mut sa = sais("aaab");
//         let sampler = Sampler::new()
//         sa.compute_smoothed_unigram_probs(Some(100));
//         let sa = sa;

//         let unigram_probs = sa.get_cached_smoothed_unigram_probs();
//         let residual = (unigram_probs.iter().sum::<f64>() - 1.0).abs();
//         assert!(residual < 1e-4);

//         // Additive smoothing results in non-zero probability on unused vocabulary
//         assert!(unigram_probs[0] > 0.0);
//     }
//     #[test]
//     fn compute_ngram_counts_exists() {
//         let sa = sais("aaabbbaaa");
//         let count_map = sa.count_ngrams(3);

//         // Every 3-gram except aaa occurs once
//         let mut expected_map = std::collections::HashMap::new();
//         expected_map.insert(1, 5);
//         expected_map.insert(2, 1);

//         assert_eq!(count_map, expected_map);
//     }

//     #[test]
//     fn sample_query_exists() {
//         let sa = sais("aaa");
//         let a = utf16!("a");
//         let tokens = sa.sample(a, 3, 10, None).unwrap();

//         assert_eq!(*tokens.last().unwrap(), a[0]);
//     }

//     #[test]
//     fn sample_empty_query_exists() {
//         let sa = sais("aaa");
//         let a = utf16!("a");
//         let empty_query = utf16!("");
//         let tokens = sa.sample(empty_query, 3, 10, None).unwrap();

//         assert_eq!(*tokens.last().unwrap(), a[0]);
//     }
// }