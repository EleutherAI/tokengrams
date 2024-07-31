pub mod mmap_slice;
pub use in_memory_index::InMemoryIndex;
pub use memmap_index::MemmapIndex;
pub use sharded_memmap_index::ShardedMemmapIndex;
pub use table::SuffixTable;

/// Python bindings
use pyo3::prelude::*;

mod sharded_memmap_index;
mod in_memory_index;
mod memmap_index;
mod sample;
mod table;
mod par_quicksort;
mod util;

#[pymodule]
fn tokengrams(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InMemoryIndex>()?;
    m.add_class::<MemmapIndex>()?;
    m.add_class::<ShardedMemmapIndex>()?;
    Ok(())
}
