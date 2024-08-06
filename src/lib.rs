pub mod mmap_slice;
pub use in_memory_index::InMemoryIndex;
pub use memmap_index::MemmapIndex;
pub use sharded_memmap_index::ShardedMemmapIndex;
pub use table::{SuffixTable, Table};

/// Python bindings
use pyo3::prelude::*;

mod in_memory_index;
mod memmap_index;
mod par_quicksort;
mod sample;
mod sharded_memmap_index;
mod table;
mod util;
mod token;

#[pymodule]
fn tokengrams(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InMemoryIndex>()?;
    m.add_class::<MemmapIndex>()?;
    m.add_class::<ShardedMemmapIndex>()?;
    Ok(())
}
