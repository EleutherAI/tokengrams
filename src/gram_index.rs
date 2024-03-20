use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use std::fs::File;
use std::io::Read;

use crate::table::SuffixTable;
use crate::util::transmute_slice;


#[pyclass]
pub struct GramIndex {
    table: SuffixTable,
}

#[pymethods]
impl GramIndex {
    #[new]
    fn new(_py: Python, tokens: Vec<u16>) -> Self {
        GramIndex {
            table: SuffixTable::par_new(tokens),
        }
    }

    #[staticmethod]
    fn from_pretrained(path: String) -> PyResult<Self> {
        // TODO: handle errors here
        let table: SuffixTable = deserialize(&std::fs::read(path)?).unwrap();
        Ok(GramIndex { table })
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

        Ok(GramIndex {
            table: SuffixTable::par_new(transmute_slice(buffer.as_slice())),
        })
    }

    fn contains(&self, query: Vec<u16>) -> bool {
        self.table.contains(&query)
    }

    fn count(&self, query: Vec<u16>) -> usize {
        self.table.positions(&query).len()
    }

    fn save(&self, path: String) -> PyResult<()> {
        // TODO: handle errors here
        let bytes = serialize(&self.table).unwrap();
        std::fs::write(&path, bytes)?;
        Ok(())
    }
}