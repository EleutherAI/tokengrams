use anyhow::Result;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::time::Instant;
use funty::Unsigned;

use crate::mmap_slice::{MmapSlice, MmapSliceMut};
use crate::par_quicksort::par_sort_unstable_by_key;
use crate::sample::{KneserNeyCache, Sample};
use crate::table::SuffixTable;

/// A memmap index exposes suffix table functionality over text corpora too large to fit in memory.
// #[pyclass]
pub struct MemmapIndex<T: Unsigned> {
    table: SuffixTable<MmapSlice<T>, MmapSlice<u64>>,
    cache: KneserNeyCache,
}


macro_rules! create_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            inner: MemmapIndex<$type>,
        }
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(_py: Python, text_path: String, table_path: String, data: $type) -> PyResult<Self> {
                let text_file = File::open(&text_path)?;
                let table_file = File::open(&table_path)?;

                Ok(Self {
                    inner: MemmapIndex { 
                        table: SuffixTable::from_parts(
                            MmapSlice::<$type>::new(&text_file)?,
                            MmapSlice::new(&table_file)?,
                        ),
                        cache: KneserNeyCache::default(),
                    },
                })
            }
        }
    };
}

create_interface!(U16, u16);
create_interface!(U32, u32);

#[pymethods]
impl MemmapIndex {
    #[new]
    pub fn new(_py: Python, text_path: String, table_path: String) -> PyResult<Self> {
        let text_file = File::open(&text_path)?;
        let table_file = File::open(&table_path)?;

        Ok(MemmapIndex {
            table: SuffixTable::from_parts(
                MmapSlice::new(&text_file)?,
                MmapSlice::new(&table_file)?,
            ),
            cache: KneserNeyCache::default(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (text_path, table_path, verbose=false))]
    pub fn build(text_path: String, table_path: String, verbose: bool) -> PyResult<Self> {
        // Memory map the text as read-only
        let text_mmap = MmapSlice::new(&File::open(&text_path)?)?;

        // Create the table file
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
        Ok(MemmapIndex {
            table: SuffixTable::from_parts(text_mmap, table_mmap),
            cache: KneserNeyCache::default(),
        })
    }

    pub fn positions(&self, query: Vec<u16>) -> Vec<u64> {
        self.table.positions(&query).to_vec()
    }

    pub fn is_sorted(&self) -> bool {
        self.table.is_sorted()
    }

    pub fn contains(&self, query: Vec<u16>) -> bool {
        self.table.contains(&query)
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
    pub fn batch_get_smoothed_probs(
        &mut self,
        queries: Vec<Vec<u16>>,
        vocab: Option<u16>,
    ) -> Vec<Vec<f64>> {
        self.batch_get_smoothed_probs_rs(&queries, vocab)
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

impl Sample for MemmapIndex {
    fn get_cache(&self) -> &KneserNeyCache {
        &self.cache
    }

    fn get_mut_cache(&mut self) -> &mut KneserNeyCache {
        &mut self.cache
    }

    fn count_next_slice(&self, query: &[u16], vocab: Option<u16>) -> Vec<usize> {
        self.table.count_next(query, vocab)
    }

    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        self.table.count_ngrams(n)
    }
}
