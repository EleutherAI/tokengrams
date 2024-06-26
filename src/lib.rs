pub mod mmap_slice;
pub use in_memory_index::InMemoryIndex;
pub use memmap_index::MemmapIndex;
pub use table::SuffixTable;

/// Python bindings
use pyo3::prelude::*;

mod in_memory_index;
mod memmap_index;
mod par_quicksort;
mod table;
mod util;
mod merge;

#[pymodule]
fn tokengrams(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<InMemoryIndex>()?;
    m.add_class::<MemmapIndex>()?;
    Ok(())
}
