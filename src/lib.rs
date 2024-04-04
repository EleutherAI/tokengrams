pub mod mmap_slice;
pub use in_memory_index::InMemoryIndex;
pub use memmap_index::MemmapIndex;
pub use table::SuffixTable;

/// Python bindings
use pyo3::prelude::*;

mod table;
mod util;
mod in_memory_index;
mod memmap_index;
mod par_quicksort;

#[pymodule]
fn tokengrams(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<InMemoryIndex>()?;
    m.add_class::<MemmapIndex>()?;
    Ok(())
}
