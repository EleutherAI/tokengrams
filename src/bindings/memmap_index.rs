use crate::memmap_index::MemmapIndexRs;
use anyhow::Result;
use pyo3::prelude::*;

/// A memmap index exposes suffix table functionality over text corpora too large to fit in memory.
#[pyclass]
pub struct MemmapIndex {
    index: Box<dyn MemmapIndexTrait + Send + Sync>,
}

/// This trait is non-generic for PyO3 compatibility. Implementing structs may cast data
/// to other unsigned integer types.
pub trait MemmapIndexTrait {
    fn is_sorted(&self) -> bool;
    fn contains(&self, query: Vec<usize>) -> bool;
    fn positions(&self, query: Vec<usize>) -> Vec<u64>;
    fn count(&self, query: Vec<usize>) -> usize;
    fn count_next(&self, query: Vec<usize>) -> Vec<usize>;
    fn batch_count_next(&self, queries: Vec<Vec<usize>>) -> Vec<Vec<usize>>;
    fn sample_unsmoothed(
        &self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize,
    ) -> Result<Vec<Vec<usize>>>;
    fn sample_smoothed(
        &mut self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize,
    ) -> Result<Vec<Vec<usize>>>;
    fn get_smoothed_probs(&mut self, query: Vec<usize>) -> Vec<f64>;
    fn batch_get_smoothed_probs(&mut self, queries: Vec<Vec<usize>>) -> Vec<Vec<f64>>;
    fn estimate_deltas(&mut self, n: usize);
}

#[pymethods]
impl MemmapIndex {
    #[new]
    #[pyo3(signature = (text_path, table_path, vocab=u16::MAX as usize + 1))]
    pub fn new(
        _py: Python,
        text_path: String,
        table_path: String,
        vocab: usize,
    ) -> PyResult<Self> {
        if vocab <= u16::MAX as usize + 1 {
            Ok(MemmapIndex {
                index: Box::new(MemmapIndexRs::<u16>::new(text_path, table_path, vocab)?),
            })
        } else {
            Ok(MemmapIndex {
                index: Box::new(MemmapIndexRs::<u32>::new(text_path, table_path, vocab)?),
            })
        }
    }

    #[staticmethod]
    #[pyo3(signature = (text_path, table_path, vocab=u16::MAX as usize + 1, verbose=false))]
    pub fn build(
        text_path: String,
        table_path: String,
        vocab: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        if vocab <= u16::MAX as usize + 1 {
            Ok(MemmapIndex {
                index: Box::new(MemmapIndexRs::<u16>::build(
                    text_path, table_path, vocab, verbose,
                )?),
            })
        } else {
            Ok(MemmapIndex {
                index: Box::new(MemmapIndexRs::<u32>::build(
                    text_path, table_path, vocab, verbose,
                )?),
            })
        }
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
        self.index.positions(query).len()
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
        num_samples: usize,
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
    pub fn batch_get_smoothed_probs(&mut self, queries: Vec<Vec<usize>>) -> Vec<Vec<f64>> {
        self.index.batch_get_smoothed_probs(queries)
    }

    /// Autoregressively sample num_samples of k characters from a Kneser-Ney smoothed n-gram model.
    pub fn sample_smoothed(
        &mut self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize,
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
