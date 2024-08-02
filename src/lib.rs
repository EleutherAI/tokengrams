pub mod mmap_slice;
pub use in_memory_index::InMemoryIndex;
pub use memmap_index::MemmapIndex;
pub use memmap_index::{MemmapIndexU16, MemmapIndexU32};
pub use sharded_memmap_index::ShardedMemmapIndex;
pub use tmp::{DataHolder, DataType, DynamicData};
pub use table::SuffixTable;

/// Python bindings
use pyo3::prelude::*;

mod in_memory_index;
mod memmap_index;
mod par_quicksort;
mod tmp;
mod sample;
mod sharded_memmap_index;
mod table;
mod util;

#[pymodule]
fn tokengrams(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_class::<InMemoryIndexU16>()?;
    m.add_class::<DynamicData>()?;
    // m.add_class::<InMemoryIndexU32>()?;
    m.add_class::<MemmapIndexU16>()?;
    m.add_class::<MemmapIndexU32>()?;
    // m.add_class::<ShardedMemmapIndexU16>()?;
    // m.add_class::<ShardedMemmapIndexU32>()?;
    Ok(())
}
