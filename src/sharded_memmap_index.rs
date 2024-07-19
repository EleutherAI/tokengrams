use pyo3::prelude::*;
use crate::countable_index::Countable;
use crate::MemmapIndex;
use std::collections::HashMap;

/// A memmap index exposes suffix table functionality over text corpora too large to fit in memory.
#[pyclass]
pub struct ShardedMemmapIndex {
    shards: Vec<MemmapIndex>,
}

#[pymethods]
impl ShardedMemmapIndex {
    #[new]
    pub fn new(_py: Python, files: Vec<(String, String)>) -> PyResult<Self> {
        let shards: Vec<MemmapIndex> = files.into_iter()
            .map(|(text_path, table_path)| MemmapIndex::new(_py, text_path, table_path).unwrap())
            .collect();

        Ok(ShardedMemmapIndex { shards })
    }

    #[staticmethod]
    pub fn build(paths: Vec<(String, String)>, verbose: bool) -> PyResult<Self> {
        let shards: Vec<MemmapIndex> = paths.into_iter()
            .map(|(token_paths, index_paths)| MemmapIndex::build(token_paths, index_paths, verbose).unwrap())
            .collect();

        Ok(ShardedMemmapIndex { shards })
    }

    pub fn is_sorted(&self) -> bool {
        self.shards.iter().all(|shard| shard.is_sorted())
    }

    pub fn contains(&self, query: Vec<u16>) -> bool {
        self.shards.iter().any(|shard| shard.contains(query.clone()))
    }

    pub fn count(&self, query: Vec<u16>) -> usize {
        self.shards.iter().map(|shard| shard.count(query.clone())).sum()
    }

    pub fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize> {
        let counts = self.shards.iter().map(|shard| {
            shard.count_next(query.clone(), vocab)
        }).collect::<Vec<_>>();
        (0..counts[0].len()).map(|i| counts.iter().map(|count| count[i]).sum()).collect()
    }

    pub fn batch_count_next(&self, queries: Vec<Vec<u16>>, vocab: Option<u16>) -> Vec<Vec<usize>> {
        let batch_counts = self.shards.iter().map(|shard| {
            shard.batch_count_next(queries.clone(), vocab)
        }).collect::<Vec<_>>();

        (0..queries.len()).map(|i| {
            (0..batch_counts[0][i].len()).map(|j| {
                batch_counts.iter().map(|count| count[i][j]).sum()
            }).collect()
        }).collect()
    }
}

impl Countable for ShardedMemmapIndex {
    fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize> {
        let counts = self.shards.iter().map(|shard| {
            shard.count_next(query.clone(), vocab)
        }).collect::<Vec<_>>();
        (0..counts[0].len()).map(|i| counts.iter().map(|count| count[i]).sum()).collect()
    }

    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        self.shards.iter().map(|shard| shard.count_ngrams(n)).fold(HashMap::new(), |mut acc, counts| {
            for (k, v) in counts {
                *acc.entry(k).or_insert(0) += v;
            }
            acc
        })
    }
}