use std::collections::HashMap;
use pyo3::prelude::*;
use crate::SuffixTable;
use crate::countable::Countable;

#[pyclass(frozen)]
pub struct CountableIndex {
    index: Box<dyn Countable + Send + Sync>
}

unsafe impl Send for CountableIndex {}

impl CountableIndex {
    pub fn new(index: Box<dyn Countable + Send + Sync>) -> Self {
        CountableIndex { index }
    }

    pub fn from_str(text: &str) -> Self {
        CountableIndex { 
            index: Box::new(SuffixTable::new(text.encode_utf16().collect::<Vec<_>>(), false))
        }
    }
}

impl Countable for CountableIndex {
    fn count_next_slice(&self, query: &[u16], vocab: Option<u16>) -> Vec<usize> {
        self.index.count_next_slice(query, vocab)
    }

    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        self.index.count_ngrams(n)
    }
}