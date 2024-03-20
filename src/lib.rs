pub mod table;
pub mod util;

pub use table::SuffixTable;

/// Python bindings
use pyo3::prelude::*;

mod gram_index;
use gram_index::GramIndex;

#[pymodule]
fn tokengrams(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GramIndex>()?;
    Ok(())
}