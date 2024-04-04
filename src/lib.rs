pub mod mmap_slice;
pub mod table;
pub mod util;

pub use mmap_slice::MmapSlice;
pub use table::SuffixTable;

/// Python bindings
use pyo3::prelude::*;

mod in_memory_index;
mod memmap_index;
mod par_quicksort;
use in_memory_index::InMemoryIndex;
use memmap_index::MemmapIndex;

#[pymodule]
fn tokengrams(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<InMemoryIndex>()?;
    m.add_class::<MemmapIndex>()?;
    Ok(())
}
