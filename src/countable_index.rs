use std::collections::HashMap;
use pyo3::pyclass;
use crate::in_memory_index::InMemoryIndex;
use crate::memmap_index::MemmapIndex;
use crate::sharded_memmap_index::ShardedMemmapIndex;

pub trait Countable: Send + Sync {
    fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize>;
    
    /// Generate a frequency map from an occurrence frequency 
    /// to the number of n-grams in the data structure with that 
    /// frequency.
    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize>;
}

#[pyclass]
pub struct CountableIndex {
    index: Box<dyn Countable + Send + Sync>
}

impl CountableIndex {
    pub fn in_memory_index(index: InMemoryIndex) -> Self {
        CountableIndex { index: Box::new(index) }
    }

    pub fn memmap_index(index: MemmapIndex) -> Self {
        CountableIndex { index: Box::new(index) }
    }

    pub fn sharded_memmap_index(index: ShardedMemmapIndex) -> Self {
        CountableIndex { index: Box::new(index) }
    }

    pub fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize> {
        self.index.count_next(query, vocab)
    }

    pub fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        self.index.count_ngrams(n)
    }
}