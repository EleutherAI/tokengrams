use anyhow::Result;
use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::fs::OpenOptions;

use crate::mmap_slice::{MmapSliceMut, MmapSlice};
use crate::sample::{KneserNeyCache, Sample};
use crate::table::SuffixTable;
use crate::util::transmute_slice;
use crate::table::InMemoryTable;

/// An in-memory index exposes suffix table functionality over text corpora small enough to fit in memory.
#[pyclass]
pub struct InMemoryIndex {
    table: Box<dyn InMemoryTable + Send + Sync>,
    cache: KneserNeyCache
}

impl InMemoryIndex {
    pub fn new(tokens: Vec<usize>, vocab: Option<usize>, verbose: bool) -> Self {
        let vocab = vocab.unwrap_or(u16::MAX as usize + 1);

        let table: Box<dyn InMemoryTable + Send + Sync> = if vocab <= u16::MAX as usize + 1 {
            let tokens: Vec<u16> = tokens.iter().map(|&x| x as u16).collect();
            Box::new(SuffixTable::<Box<[u16]>, Box<[u64]>>::new(tokens, Some(vocab), verbose))
        } else {
            let tokens: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
            Box::new(SuffixTable::<Box<[u32]>, Box<[u64]>>::new(tokens, Some(vocab), verbose))
        };

        debug_assert!(table.is_sorted());

        InMemoryIndex {
            table,
            cache: KneserNeyCache::default(),
        }
    }
}

impl Sample for InMemoryIndex {
    fn get_cache(&self) -> &KneserNeyCache {
        &self.cache
    }

    fn get_mut_cache(&mut self) -> &mut KneserNeyCache {
        &mut self.cache
    }

    fn count_next_slice(&self, query: &[usize]) -> Vec<usize> {
        self.table.count_next(query)
    }

    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        self.table.count_ngrams(n)
    }
}

#[pymethods]
impl InMemoryIndex {
    #[new]
    #[pyo3(signature = (tokens, vocab=u16::MAX as usize + 1, verbose=false))]
    pub fn new_py(_py: Python, tokens: Vec<usize>, vocab: usize, verbose: bool) -> Self {
        let table: Box<dyn InMemoryTable + Send + Sync> = if vocab <= u16::MAX as usize + 1 {
            let tokens: Vec<u16> = tokens.iter().map(|&x| x as u16).collect();
            Box::new(SuffixTable::<Box<[u16]>, Box<[u64]>>::new(tokens, Some(vocab), verbose))
        } else {
            let tokens: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
            Box::new(SuffixTable::<Box<[u32]>, Box<[u64]>>::new(tokens, Some(vocab), verbose))
        };

        debug_assert!(table.is_sorted());

        InMemoryIndex {
            table,
            cache: KneserNeyCache::default()
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, vocab=u16::MAX as usize + 1))]
    pub fn from_pretrained(path: String, vocab: usize) -> PyResult<Self> {
        // TODO: handle errors here
        if vocab <= u16::MAX as usize + 1 {
            let table: SuffixTable<Box<[u16]>> = deserialize(&std::fs::read(path)?).unwrap();
            debug_assert!(table.is_sorted());
            Ok(InMemoryIndex {
                table: Box::new(table),
                cache: KneserNeyCache::default()
            })
        } else {
            let table: SuffixTable<Box<[u32]>> = deserialize(&std::fs::read(path)?).unwrap();
            debug_assert!(table.is_sorted());
            Ok(InMemoryIndex {
                table: Box::new(table),
                cache: KneserNeyCache::default()
            })
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, token_limit=None, vocab=u16::MAX as usize + 1, verbose=false))]
    pub fn from_token_file(
        path: String,
        token_limit: Option<usize>,
        vocab: usize,
        verbose: bool,
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

        let table: Box<dyn InMemoryTable + Send + Sync> = if vocab <= u16::MAX as usize + 1 {
            let tokens = transmute_slice::<u8, u16>(buffer.as_slice());
            Box::new(SuffixTable::new(tokens, Some(vocab), verbose))
        } else {
            let tokens = transmute_slice::<u8, u32>(buffer.as_slice());
            Box::new(SuffixTable::new(tokens, Some(vocab), verbose))
        };
        debug_assert!(table.is_sorted());

        Ok(InMemoryIndex {
            table,
            cache: KneserNeyCache::default()
        })
    }

    pub fn save(&self, path: String) -> PyResult<()> {
        // TODO: handle errors here
        let bytes = serialize(&self.table).unwrap();
        std::fs::write(&path, bytes)?;
        Ok(())
    }

    pub fn save_table(&self, table_path: String) -> Result<()> {
        let table = self.table.get_table();
        println!("table len: {}, {:?}", table.len(), table);
        let table_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&table_path)?;

        table_file.set_len((table.len() * 8) as u64)?;

        let mut table_mmap = MmapSliceMut::<u64>::new(&table_file)?;
        table_mmap.copy_from_slice(table);
        table_mmap.flush()?;

        Ok(())
    }

    pub fn is_sorted(&self) -> bool {
        self.table.is_sorted()
    }

    pub fn contains(&self, query: Vec<usize>) -> bool {
        self.table.contains(&query)
    }

    pub fn positions(&self, query: Vec<usize>) -> Vec<u64> {
        self.table.positions(&query).to_vec()
    }

    pub fn count(&self, query: Vec<usize>) -> usize {
        self.table.positions(&query).len()
    }

    pub fn count_next(&self, query: Vec<usize>) -> Vec<usize> {
        self.table.count_next(&query)
    }

    pub fn batch_count_next(&self, queries: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        self.table.batch_count_next(&queries)
    }

    /// Autoregressively sample num_samples of k characters from an unsmoothed n-gram model."""
    pub fn sample_unsmoothed(
        &self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize
    ) -> Result<Vec<Vec<usize>>> {
        self.sample_unsmoothed_rs(&query, n, k, num_samples)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    pub fn get_smoothed_probs(&mut self, query: Vec<usize>) -> Vec<f64> {
        self.get_smoothed_probs_rs(&query)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    pub fn batch_get_smoothed_probs(
        &mut self,
        queries: Vec<Vec<usize>>
    ) -> Vec<Vec<f64>> {
        self.batch_get_smoothed_probs_rs(&queries)
    }

    /// Autoregressively sample num_samples of k characters from a Kneser-Ney smoothed n-gram model.
    pub fn sample_smoothed(
        &mut self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize
    ) -> Result<Vec<Vec<usize>>> {
        self.sample_smoothed_rs(&query, n, k, num_samples)
    }

    /// Warning: O(k**n) where k is vocabulary size, use with caution.
    /// Improve smoothed model quality by replacing the default delta hyperparameters
    /// for models of order n and below with improved estimates over the entire index.
    /// https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf, page 16."""
    pub fn estimate_deltas(&mut self, n: usize) {
        self.estimate_deltas_rs(n);
    }
}