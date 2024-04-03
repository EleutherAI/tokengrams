pub mod mmap_slice;
pub mod table;
pub mod util;
pub mod par_quicksort;

pub use mmap_slice::MmapSlice;
pub use table::SuffixTable;
pub use par_quicksort::par_quicksort;

/// Python bindings
use pyo3::prelude::*;

mod in_memory_index;
mod memmap_index;
use in_memory_index::InMemoryIndex;
use memmap_index::MemmapIndex;

#[pymodule]
fn tokengrams(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<InMemoryIndex>()?;
    m.add_class::<MemmapIndex>()?;
    Ok(())
}
