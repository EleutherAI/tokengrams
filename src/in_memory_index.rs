use bincode::{deserialize, serialize};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::File;
use std::io::Read;

use crate::table::SuffixTable;
use crate::util::transmute_slice;

/// An in-memory index exposes suffix table functionality over text corpora small enough to fit in memory.
#[pyclass]
pub struct InMemoryIndex {
    table: SuffixTable,
}

#[pymethods]
impl InMemoryIndex {
    #[new]
    pub fn new(_py: Python, tokens: Vec<u16>) -> Self {
        InMemoryIndex {
            table: SuffixTable::new(tokens),
        }
    }

    #[staticmethod]
    pub fn from_pretrained(path: String) -> PyResult<Self> {
        // TODO: handle errors here
        let table: SuffixTable = deserialize(&std::fs::read(path)?).unwrap();
        Ok(InMemoryIndex { table })
    }

    #[staticmethod]
    pub fn from_token_file(
        path: String,
        token_limit: Option<usize>,
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

        Ok(InMemoryIndex {
            table: SuffixTable::new(transmute_slice(buffer.as_slice())),
        })
    }

    #[staticmethod]
    pub fn from_token_file_sais(
        path: String,
        token_limit: Option<usize>,
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

        Ok(InMemoryIndex {
            table: SuffixTable::new_sais(transmute_slice(buffer.as_slice())),
        })
    }

    #[staticmethod]
    pub fn sais(tokens: Vec<u16>) -> Self {
        InMemoryIndex {
            table: SuffixTable::new_sais(tokens),
        }
    }

    pub fn contains(&self, query: Vec<u16>) -> bool {
        self.table.contains(&query)
    }

    pub fn count(&self, query: Vec<u16>) -> usize {
        self.table.positions(&query).len()
    }

    pub fn positions(&self, query: Vec<u16>) -> Vec<u64> {
        self.table.positions(&query).to_vec()
    }

    pub fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize> {
        self.table.count_next(&query, vocab)
    }

    pub fn batch_count_next(&self, queries: Vec<Vec<u16>>, vocab: Option<u16>) -> Vec<Vec<usize>> {
        self.table.batch_count_next(&queries, vocab)
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

    pub fn is_sorted(&self) -> bool {
        self.table.is_sorted()
    }

    pub fn save(&self, path: String) -> PyResult<()> {
        // TODO: handle errors here
        let bytes = serialize(&self.table).unwrap();
        std::fs::write(&path, bytes)?;
        Ok(())
    }
}
