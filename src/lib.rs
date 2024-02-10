pub mod sucds_glue;
pub mod table;
pub mod util;

pub use table::SuffixTable;

/*/// Python bindings
use pyo3::prelude::*;

#[pymodule]
fn tokengrams(_py: Python, m: &PyModule) -> PyResult<()> {
    Ok(())
}
*/