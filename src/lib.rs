pub mod mmap_slice;
pub use table::SuffixTable;
pub use in_memory_index::{InMemoryIndexU16, InMemoryIndexU32};
pub use memmap_index::{MemmapIndexU16, MemmapIndexU32};
pub use sharded_memmap_index::{ShardedMemmapIndexU16, ShardedMemmapIndexU32};

/// Python bindings
use pyo3::prelude::*;

mod in_memory_index;
mod memmap_index;
mod par_quicksort;
mod sample;
mod sharded_memmap_index;
mod table;
mod util;

#[pymodule]
fn tokengrams(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InMemoryIndexU16>()?;
    m.add_class::<InMemoryIndexU32>()?;
    m.add_class::<MemmapIndexU16>()?;
    m.add_class::<MemmapIndexU32>()?;
    m.add_class::<ShardedMemmapIndexU16>()?;
    m.add_class::<ShardedMemmapIndexU32>()?;
    Ok(())
}
