use bincode::{deserialize, serialize};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::File;
use std::io::Read;

use crate::table::SuffixTable;
use crate::util::transmute_slice;

#[pyclass]
pub struct InMemoryIndex {
    table: SuffixTable,
}

#[pymethods]
impl InMemoryIndex {
    #[new]
    fn new(_py: Python, tokens: Vec<u16>) -> Self {
        InMemoryIndex {
            table: SuffixTable::new(tokens),
        }
    }

    #[staticmethod]
    fn from_pretrained(path: String) -> PyResult<Self> {
        // TODO: handle errors here
        let table: SuffixTable = deserialize(&std::fs::read(path)?).unwrap();
        Ok(InMemoryIndex { table })
    }

    #[staticmethod]
    fn from_token_file(path: String, token_limit: Option<usize>) -> PyResult<Self> {
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

    fn contains(&self, query: Vec<u16>) -> bool {
        self.table.contains(&query)
    }

    fn count(&self, query: Vec<u16>) -> usize {
        self.table.positions(&query).len()
    }

    pub fn positions(&self, query: Vec<u16>) -> Vec<u64> {
        self.table.positions(&query).to_vec()
    }

    fn batch_next_token_counts(&self, queries: Vec<Vec<u16>>, vocab: Option<u16>) -> Vec<Vec<usize>> {
        self.table.batch_next_token_counts(&queries, vocab)
    }

    fn sample(&self, query: Vec<u16>, n: usize, k: usize) -> Result<Vec<u16>, PyErr> {
        self.table.sample(&query, n, k)
            .map_err(|error| PyValueError::new_err(error.to_string()))  
    }

    fn batch_sample(&self, query: Vec<u16>, n: usize, k: usize, num_samples: usize) -> Result<Vec<Vec<u16>>, PyErr> {
        self.table.batch_sample(&query, n, k, num_samples)
            .map_err(|error| PyValueError::new_err(error.to_string()))  
    }

    fn is_sorted(&self) -> bool {
        self.table.is_sorted()
    }

    fn save(&self, path: String) -> PyResult<()> {
        // TODO: handle errors here
        let bytes = serialize(&self.table).unwrap();
        std::fs::write(&path, bytes)?;
        Ok(())
    }
}
