extern crate utf16_literal;

use pyo3::prelude::*;
use anyhow::Result;

use crate::countable_index::CountableIndex;
use crate::sampler_rs::SamplerRs;
use crate::MemmapIndex;
use crate::InMemoryIndex;
use crate::ShardedMemmapIndex;

#[pyclass]
pub struct Sampler {
    sampler: SamplerRs
}

#[pymethods]
impl Sampler {
    #[staticmethod]
    pub fn from_sharded_memmap_index(paths: Vec<(String, String)>, verbose: bool) -> PyResult<Sampler> {
        let index = ShardedMemmapIndex::build(paths, verbose)?;
        let countable_index: CountableIndex = CountableIndex::new(Box::new(index));
        Ok(
            Sampler { 
                sampler: SamplerRs::new(countable_index),
            }
        )
    }

    #[staticmethod]
    pub fn from_memmap_index(text_path: String, table_path: String, verbose: bool) -> PyResult<Sampler> {
        let index = MemmapIndex::build(text_path, table_path, verbose)?;
        let countable_index: CountableIndex = CountableIndex::new(Box::new(index));
        Ok(
            Sampler { 
                sampler: SamplerRs::new(countable_index),
            }
        )
    }

    #[staticmethod]
    pub fn from_in_memory_index(path: String) -> PyResult<Sampler> {
        let index = InMemoryIndex::from_pretrained(path)?;
        let countable_index = CountableIndex::new(Box::new(index));
        Ok(
            Sampler { 
                sampler: SamplerRs::new(countable_index),
            }
        )
    }

    /// Autoregressively sample num_samples of k characters from an unsmoothed n-gram model."""
    #[pyo3(signature = (query, n, k, num_samples, vocab=None))]
    pub fn sample_unsmoothed(
        &self,
        query: Vec<u16>,
        n: usize,
        k: usize,
        num_samples: usize,
        vocab: Option<u16>,
    ) -> Result<Vec<Vec<u16>>> {
        self.sampler.sample_unsmoothed(&query, n, k, num_samples, vocab)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    #[pyo3(signature = (query, vocab=None))]
    pub fn get_smoothed_probs(&mut self, query: Vec<u16>, vocab: Option<u16>) -> Vec<f64> {
        self.sampler.get_smoothed_probs(&query, vocab)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    #[pyo3(signature = (queries, vocab=None))]
    pub fn batch_get_smoothed_probs(&mut self, queries: Vec<Vec<u16>>, vocab: Option<u16>) -> Vec<Vec<f64>> {
        self.sampler.batch_get_smoothed_probs(&queries, vocab)
    }

    /// Autoregressively sample num_samples of k characters from a Kneser-Ney smoothed n-gram model.
    #[pyo3(signature = (query, n, k, num_samples, vocab=None))]
    pub fn sample_smoothed(
        &mut self,
        query: Vec<u16>,
        n: usize,
        k: usize,
        num_samples: usize,
        vocab: Option<u16>,
    ) -> Result<Vec<Vec<u16>>> {
        self.sampler.sample_smoothed(&query, n, k, num_samples, vocab)
    }

    /// Warning: O(k**n) where k is vocabulary size, use with caution.
    /// Improve smoothed model quality by replacing the default delta hyperparameters
    /// for models of order n and below with improved estimates over the entire index.
    /// https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf, page 16."""
    pub fn estimate_deltas(&mut self, n: usize) {
        self.sampler.estimate_deltas(n);
    }
}