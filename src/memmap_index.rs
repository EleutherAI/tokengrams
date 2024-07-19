use pyo3::prelude::*;
use std::fs::{File, OpenOptions};
use std::time::Instant;
use std::collections::HashMap;

use crate::mmap_slice::{MmapSlice, MmapSliceMut};
use crate::par_quicksort::par_sort_unstable_by_key;
use crate::table::SuffixTable;
use crate::countable_index::Countable;

/// A memmap index exposes suffix table functionality over text corpora too large to fit in memory.
#[pyclass]
pub struct MemmapIndex {
    table: SuffixTable<MmapSlice<u16>, MmapSlice<u64>>,
}

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
        })
    }

    #[staticmethod]
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

    pub fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize> {
        self.table.count_next(&query, vocab)
    }

    pub fn batch_count_next(&self, queries: Vec<Vec<u16>>, vocab: Option<u16>) -> Vec<Vec<usize>> {
        self.table.batch_count_next(&queries, vocab)
    }
}

impl Countable for MemmapIndex {
    fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize> {
        self.table.count_next(&query, vocab)
    }

    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        self.table.count_ngrams(n)
    }
}