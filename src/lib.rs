pub mod mmap_slice;
pub use bindings::in_memory_index::InMemoryIndex;
pub use bindings::memmap_index::MemmapIndex;
pub use bindings::sharded_memmap_index::ShardedMemmapIndex;
pub use bindings::sharded_in_memory_index::ShardedInMemoryIndex;
pub use sharded_in_memory_index::ShardedInMemoryIndexRs;
pub use in_memory_index::InMemoryIndexRs;
pub use sample::Sample;

pub use table::SuffixTable;

/// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;

mod bindings;
mod in_memory_index;
mod memmap_index;
mod par_quicksort;
mod sample;
mod sharded_memmap_index;
mod sharded_in_memory_index;
mod table;
mod util;

#[cfg(feature = "python")]
#[pymodule]
fn tokengrams(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InMemoryIndex>()?;
    m.add_class::<MemmapIndex>()?;
    m.add_class::<ShardedMemmapIndex>()?;
    m.add_class::<ShardedInMemoryIndex>()?;
    Ok(())
}
