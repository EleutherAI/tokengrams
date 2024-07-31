use bincode::{deserialize, serialize};
use std::collections::HashMap;
use pyo3::prelude::*;
use std::fs::File;
use std::io::Read;
use anyhow::Result;

use crate::table::SuffixTable;
use crate::sample::Sample;
use crate::util::transmute_slice;
use crate::sample::KneserNeyCache;

/// An in-memory index exposes suffix table functionality over text corpora small enough to fit in memory.
#[pyclass]
pub struct InMemoryIndex {
    table: SuffixTable,
    cache: KneserNeyCache,
}

#[pymethods]
impl InMemoryIndex {
    #[new]
    #[pyo3(signature = (tokens, verbose=false))]
    pub fn new_py(_py: Python, tokens: Vec<u16>, verbose: bool) -> Self {
        InMemoryIndex {
            table: SuffixTable::new(tokens, verbose),
            cache: KneserNeyCache::default(),
        }
    }

    #[staticmethod]
    pub fn from_pretrained(path: String) -> PyResult<Self> {
        // TODO: handle errors here
        let table: SuffixTable = deserialize(&std::fs::read(path)?).unwrap();
        Ok(InMemoryIndex { 
            table,
            cache: KneserNeyCache::default(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, verbose=false, token_limit=None))]
    pub fn from_token_file(
        path: String,
        verbose: bool,
        token_limit: Option<usize>,
    ) -> PyResult<Self> {
        let mut buffer = Vec::new();
        let mut file = File::open(&path)?;

        if let Some(max_tokens) = token_limit {
            // Limit on the number of tokens to consider is provided
            let max_bytes = max_tokens * std::mem::size_of::<u16>();
            file.take(max_bytes as u64).read_to_end(&mut buffer)?;
        } else {
            file.read_to_end(&mut buffer)?;
        };

        Ok(InMemoryIndex {
            table: SuffixTable::new(transmute_slice(buffer.as_slice()), verbose),
            cache: KneserNeyCache::default(),
        })
    }

    pub fn is_sorted(&self) -> bool {
        self.table.is_sorted()
    }

    pub fn contains(&self, query: Vec<u16>) -> bool {
        self.table.contains(&query)
    }

    pub fn positions(&self, query: Vec<u16>) -> Vec<u64> {
        self.table.positions(&query).to_vec()
    }

    pub fn count(&self, query: Vec<u16>) -> usize {
        self.table.positions(&query).len()
    }

    #[pyo3(signature = (query, vocab=None))]
    pub fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize> {
        self.table.count_next(&query, vocab)
    }

    #[pyo3(signature = (queries, vocab=None))]
    pub fn batch_count_next(&self, queries: Vec<Vec<u16>>, vocab: Option<u16>) -> Vec<Vec<usize>> {
        self.table.batch_count_next(&queries, vocab)
    }

    pub fn save(&self, path: String) -> PyResult<()> {
        // TODO: handle errors here
        let bytes = serialize(&self.table).unwrap();
        std::fs::write(&path, bytes)?;
        Ok(())
    }

    /// Autoregressively sample num_samples of k characters from an unsmoothed n-gram model."""
    #[pyo3(signature = (query, n, k, num_samples, vocab=None))]
    pub fn sample_unsmoothed(
        &self, query: Vec<u16>, n: usize, k: usize, num_samples: usize, vocab: Option<u16>,
    ) -> Result<Vec<Vec<u16>>> {
        self.sample_unsmoothed_rs(&query, n, k, num_samples, vocab)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    #[pyo3(signature = (query, vocab=None))]
    pub fn get_smoothed_probs(&mut self, query: Vec<u16>, vocab: Option<u16>) -> Vec<f64> {
        self.get_smoothed_probs_rs(&query, vocab)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    #[pyo3(signature = (queries, vocab=None))]
    pub fn batch_get_smoothed_probs(&mut self, queries: Vec<Vec<u16>>, vocab: Option<u16>) -> Vec<Vec<f64>> {
        self.batch_get_smoothed_probs_rs(&queries, vocab)
    }

    /// Autoregressively sample num_samples of k characters from a Kneser-Ney smoothed n-gram model.
    #[pyo3(signature = (query, n, k, num_samples, vocab=None))]
    pub fn sample_smoothed(
        &mut self, query: Vec<u16>, n: usize, k: usize, num_samples: usize, vocab: Option<u16>,
    ) -> Result<Vec<Vec<u16>>> {
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

impl Sample for InMemoryIndex {
    fn get_cache(& self) -> & KneserNeyCache {
        &self.cache
    }

    fn get_mut_cache(&mut self) -> &mut KneserNeyCache {
        &mut self.cache
    }

    fn count_next_slice(&self, query: &[u16], vocab: Option<u16>) -> Vec<usize> {
        self.table.count_next(&query, vocab)
    }

    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        self.table.count_ngrams(n)
    }
}

impl InMemoryIndex {
    pub fn new(tokens: Vec<u16>, verbose: bool) -> Self {
        InMemoryIndex {
            table: SuffixTable::new(tokens, verbose),
            cache: KneserNeyCache::default(),
        }
    }
}
