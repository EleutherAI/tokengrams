use anyhow::Result;
use pyo3::prelude::*;
use crate::in_memory_index::InMemoryIndexRs;

/// An in-memory index exposes suffix table functionality over text corpora small enough to fit in memory.
/// Non-generic PyO3 wrapper over InMemoryIndexRs.
#[pyclass]
pub struct InMemoryIndex {
    index: Box<dyn InMemoryIndexTrait + Send + Sync>
}

/// This trait is non-generic for PyO3 compatibility. Implementing structs may cast data
/// to other unsigned integer types.
pub trait InMemoryIndexTrait {
    fn save_text(&self, path: String) -> Result<()>;
    fn save_table(&self, path: String) -> Result<()>;
    fn is_sorted(&self) -> bool;
    fn contains(&self, query: Vec<usize>) -> bool;
    fn positions(&self, query: Vec<usize>) -> Vec<u64>;
    fn count(&self, query: Vec<usize>) -> usize;
    fn count_next(&self, query: Vec<usize>) -> Vec<usize>;
    fn batch_count_next(&self, queries: Vec<Vec<usize>>) -> Vec<Vec<usize>>;
    fn sample_unsmoothed(
        &self, query: Vec<usize>, n: usize, k: usize, num_samples: usize
    ) -> Result<Vec<Vec<usize>>>;
    fn sample_smoothed(
        &mut self, query: Vec<usize>, n: usize, k: usize, num_samples: usize
    ) -> Result<Vec<Vec<usize>>>;
    fn get_smoothed_probs(&mut self, query: Vec<usize>) -> Vec<f64>;
    fn batch_get_smoothed_probs(
        &mut self, queries: Vec<Vec<usize>>
    ) -> Vec<Vec<f64>>;
    fn estimate_deltas(&mut self, n: usize);
}

impl InMemoryIndex {
    pub fn new(tokens: Vec<usize>, vocab: Option<usize>, verbose: bool) -> Self {
        let vocab = vocab.unwrap_or(u16::MAX as usize + 1);
    
        let index: Box<dyn InMemoryIndexTrait + Send + Sync> = if vocab <= u16::MAX as usize + 1 {
            let tokens: Vec<u16> = tokens.iter().map(|&x| x as u16).collect();
            Box::new(InMemoryIndexRs::<u16>::new(tokens, Some(vocab), verbose))
        } else {
            let tokens: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
            Box::new(InMemoryIndexRs::<u32>::new(tokens, Some(vocab), verbose))
        };

        InMemoryIndex {
            index,
        }
    }
}

#[pymethods]
impl InMemoryIndex {
    #[new]
    #[pyo3(signature = (tokens, vocab=u16::MAX as usize + 1, verbose=false))]
    pub fn new_py(_py: Python, tokens: Vec<usize>, vocab: usize, verbose: bool) -> Self {
    
        let index: Box<dyn InMemoryIndexTrait + Send + Sync> = if vocab <= u16::MAX as usize + 1 {
            let tokens: Vec<u16> = tokens.iter().map(|&x| x as u16).collect();
            Box::new(InMemoryIndexRs::<u16>::new(tokens, Some(vocab), verbose))
        } else {
            let tokens: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
            Box::new(InMemoryIndexRs::<u32>::new(tokens, Some(vocab), verbose))
        };

        InMemoryIndex {
            index,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, token_limit=None, vocab=u16::MAX as usize + 1, verbose=false))]
    pub fn from_token_file(path: String, token_limit: Option<usize>, vocab: usize, verbose: bool) -> Result<Self> {
        if vocab <= u16::MAX as usize + 1 {
            Ok(InMemoryIndex {
                index: Box::new(InMemoryIndexRs::<u16>::from_token_file(path, token_limit, vocab, verbose)?)
            })
        } else {
            Ok(InMemoryIndex {
                index: Box::new(InMemoryIndexRs::<u32>::from_token_file(path, token_limit, vocab, verbose)?)
            })
        }
    }

    #[staticmethod]
    #[pyo3(signature = (text_path, table_path, vocab=u16::MAX as usize + 1))]
    pub fn from_disk(text_path: String, table_path: String, vocab: usize,) -> Result<Self> {
        if vocab <= u16::MAX as usize + 1 {
            Ok(InMemoryIndex {
                index: Box::new(InMemoryIndexRs::<u16>::from_disk(text_path, table_path, vocab)?)
            })
        } else {
            Ok(InMemoryIndex {
                index: Box::new(InMemoryIndexRs::<u32>::from_disk(text_path, table_path, vocab)?)
            })
        }
    }

    pub fn save_text(&self, path: String) -> Result<()> {
        self.index.save_text(path)
    }

    pub fn save_table(&self, path: String) -> Result<()> {
        self.index.save_table(path)
    }

    pub fn is_sorted(&self) -> bool {
        self.index.is_sorted()
    }

    pub fn contains(&self, query: Vec<usize>) -> bool {
        self.index.contains(query)
    }

    pub fn positions(&self, query: Vec<usize>) -> Vec<u64> {
        self.index.positions(query).to_vec()
    }

    pub fn count(&self, query: Vec<usize>) -> usize {
        self.index.count(query)
    }

    pub fn count_next(&self, query: Vec<usize>) -> Vec<usize> {
        self.index.count_next(query)
    }

    pub fn batch_count_next(&self, queries: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        self.index.batch_count_next(queries)
    }

    /// Autoregressively sample num_samples of k characters from an unsmoothed n-gram model."""
    pub fn sample_unsmoothed(
        &self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize
    ) -> Result<Vec<Vec<usize>>> {
        self.index.sample_unsmoothed(query, n, k, num_samples)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    pub fn get_smoothed_probs(&mut self, query: Vec<usize>) -> Vec<f64> {
        self.index.get_smoothed_probs(query)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    pub fn batch_get_smoothed_probs(
        &mut self,
        queries: Vec<Vec<usize>>
    ) -> Vec<Vec<f64>> {
        self.index.batch_get_smoothed_probs(queries)
    }

    /// Autoregressively sample num_samples of k characters from a Kneser-Ney smoothed n-gram model.
    pub fn sample_smoothed(
        &mut self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize
    ) -> Result<Vec<Vec<usize>>> {
        self.index.sample_smoothed(query, n, k, num_samples)
    }

    /// Warning: O(k**n) where k is vocabulary size, use with caution.
    /// Improve smoothed model quality by replacing the default delta hyperparameters
    /// for models of order n and below with improved estimates over the entire index.
    /// https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf, page 16."""
    pub fn estimate_deltas(&mut self, n: usize) {
        self.index.estimate_deltas(n);
    }
}