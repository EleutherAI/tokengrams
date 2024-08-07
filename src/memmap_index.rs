use anyhow::Result;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::time::Instant;

use crate::mmap_slice::{MmapSlice, MmapSliceMut};
use crate::par_quicksort::par_sort_unstable_by_key;
use crate::sample::{KneserNeyCache, Sample};
use crate::table::SuffixTable;
use crate::table::Table;
use crate::token::Token;

/// A memmap index exposes suffix table functionality over text corpora too large to fit in memory.
#[pyclass]
pub struct MemmapIndex {
    table: Box<dyn Table + Send + Sync>,
    cache: KneserNeyCache
}

impl MemmapIndex {
    fn build_typed<T: Token>(
        text_path: String, 
        table_path: String,
        vocab: Option<usize>, 
        verbose: bool
    ) -> PyResult<Self> {
        // Memory map the text as read-only
        let text_mmap = MmapSlice::<T>::new(&File::open(&text_path).unwrap()).unwrap();

        let table_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&table_path)?;

        // Allocate space on disk for the table
        let table_size = text_mmap.len() * 8;
        table_file.set_len(table_size as u64)?;

        if verbose {
            println!("Writing indices to disk...");
        }
        let start = Instant::now();
        let mut table_mmap = MmapSliceMut::<u64>::new(&table_file)?;
        table_mmap
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = i as u64);

        assert_eq!(table_mmap.len(), text_mmap.len());
        if verbose {
            println!("Time elapsed: {:?}", start.elapsed());
        }
        let start = Instant::now();

        // TODO: Be even smarter about this? We may need to take into account the number of CPUs
        // available as well. These magic numbers were tuned on a server with 48 physical cores.
        // Empirically we start getting stack overflows between 5B and 10B tokens when using the
        // default stack size of 2MB. We scale the stack size as log2(n) * 8MB to avoid this.
        let scale = (text_mmap.len() as f64) / 5e9; // 5B tokens
        let stack_size = scale.log2().max(1.0) * 8e6; // 8MB

        rayon::ThreadPoolBuilder::new()
            .stack_size(stack_size as usize)
            .build()
            .unwrap()
            .install(|| {
                // Sort the indices by the suffixes they point to.
                // The unstable algorithm is critical for avoiding out-of-memory errors, since it does
                // not allocate any more memory than the input and output slices.
                println!("Sorting indices...");
                par_sort_unstable_by_key(
                    table_mmap.as_slice_mut(),
                    |&i| &text_mmap[i as usize..],
                    verbose,
                );
            });
        if verbose {
            println!("Time elapsed: {:?}", start.elapsed());
        }

        // Re-open the table as read-only
        let table_mmap = MmapSlice::new(&table_file)?;
        let table = SuffixTable::from_parts(text_mmap, table_mmap, vocab);
        assert!(table.is_sorted());

        Ok(MemmapIndex {
            table: Box::new(table),
            cache: KneserNeyCache::default(),
        })

    }
}

#[pymethods]
impl MemmapIndex {
    #[new]
    #[pyo3(signature = (text_path, table_path, vocab=u16::MAX as usize + 1))]
    pub fn new(_py: Python, text_path: String, table_path: String, vocab: usize) -> PyResult<Self> {
        let text_file = File::open(&text_path)?;
        let table_file = File::open(&table_path)?;

        let table: Box<dyn Table + Send + Sync> = if vocab <= u16::MAX as usize + 1 {
            Box::new(SuffixTable::from_parts(
                MmapSlice::<u16>::new(&text_file)?,
                MmapSlice::new(&table_file)?,
                Some(vocab)
            ))
        } else {
            Box::new(SuffixTable::from_parts(
                MmapSlice::<u32>::new(&text_file)?,
                MmapSlice::new(&table_file)?,
                Some(vocab)
            ))
        };
        assert!(table.is_sorted());

        Ok(MemmapIndex {
            table,
            cache: KneserNeyCache::default()
        })
    }

    #[staticmethod]
    #[pyo3(signature = (text_path, table_path, vocab=u16::MAX as usize + 1, verbose=false))]
    pub fn build(text_path: String, table_path: String, vocab: usize, verbose: bool) -> PyResult<Self> {
        if vocab <= u16::MAX as usize + 1 {
            MemmapIndex::build_typed::<u16>(text_path, table_path, Some(vocab), verbose)
        } else {
            MemmapIndex::build_typed::<u32>(text_path, table_path, Some(vocab), verbose)
        }
    }

    pub fn positions(&self, query: Vec<usize>) -> Vec<u64> {
        self.table.positions(&query).to_vec()
    }

    pub fn is_sorted(&self) -> bool {
        self.table.is_sorted()
    }

    pub fn contains(&self, query: Vec<usize>) -> bool {
        self.table.contains(&query)
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
    #[pyo3(signature = (query, n, k, num_samples))]
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

impl Sample for MemmapIndex {
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
