use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::{File, OpenOptions};
use std::time::Instant;
use std::collections::HashSet;

use crate::mmap_slice::{MmapSlice, MmapSliceMut};
use crate::table::SuffixTable;
use crate::MemmapIndex;

/// A memmap index exposes suffix table functionality over text corpora too large to fit in memory.
#[pyclass]
pub struct ShardedMemmapIndex {
    shards: HashSet<SuffixTable<MmapSlice<u16>, MmapSlice<u64>>>,
}

#[pymethods]
impl ShardedMemmapIndex {
    #[new]
    pub fn new(_py: Python, files: Vec<(String, String)>) -> PyResult<Self> {
        let mut shards = HashSet::new();
        for (text_path, table_path) in files {
            let text_file = File::open(&text_path)?;
            let table_file = File::open(&table_path)?;

            shards.insert(SuffixTable::from_parts(
                MmapSlice::new(&text_file)?,
                MmapSlice::new(&table_file)?,
            ));
        }

        Ok(ShardedIndex { shards })
    }

    #[staticmethod]
    pub fn build(paths: Vec<(String, String)>, verbose: bool) -> PyResult<Self> {
        for (token_paths, index_paths) in paths {
            MemmapIndex.build(token_paths, index_paths, verbose);
        }
        ShardedMemmapIndex(paths)
    }

    pub fn is_sorted(&self) -> bool {
        self.shards.iter().all(|shard| shard.is_sorted())
    }

    pub fn contains(&self, query: Vec<u16>) -> bool {
        self.shards.iter().any(|shard| shard.contains(&query))
    }

    pub fn count(&self, query: Vec<u16>) -> usize {
        self.shards.iter().map(|shard| shard.count(query)).sum()
    }

    pub fn positions(&self, query: Vec<u16>) -> Vec<u64> {
        self.table.positions(&query).to_vec()
    }

    pub fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize> {
        let counts = self.shards.iter().map(|shard| {
            shard.count_next(&query, vocab)
        }).collect::<Vec<_>>();
        (0..counts[0].len()).map(|i| counts.iter().map(|count| count[i]).sum()).collect()
    }
}

    pub fn batch_count_next(&self, queries: Vec<Vec<u16>>, vocab: Option<u16>) -> Vec<Vec<usize>> {
        let batch_counts = self.shards.iter().map(|shard| {
            shard.count_next(&query, vocab)
        }).collect::<Vec<_>>();
        (0..batch_counts[0].len()).map(|i| batch_counts.iter().map(|count| count[i]).sum()).collect()
    }

    pub fn sample(&self, query: Vec<u16>, n: usize, k: usize) -> Result<Vec<u16>, PyErr> {
        self.table
            .sample(&query, n, k)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    pub fn batch_sample(
        &self,
        query: Vec<u16>,
        n: usize,
        k: usize,
        num_samples: usize,
    ) -> Result<Vec<Vec<u16>>, PyErr> {
        self.table
            .batch_sample(&query, n, k, num_samples)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    
}
