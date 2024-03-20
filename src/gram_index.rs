use pyo3::prelude::*;

use crate::table::SuffixTable;


#[pyclass]
pub struct GramIndex {
    table: SuffixTable,
}

#[pymethods]
impl GramIndex {
    #[new]
    fn new(_py: Python, tokens: Vec<u16>) -> Self {
        GramIndex {
            table: SuffixTable::new(tokens),
        }
    }

    /// Count the number of occurrences of a query in the index
    fn count(&self, query: Vec<u16>) -> usize {
        self.table.positions(&query).len()
    }

    /// Search for the positions of a query in the index
    fn search(&self, query: Vec<u16>) -> Vec<u64> {
        self.table.positions(&query).into()
    }
}