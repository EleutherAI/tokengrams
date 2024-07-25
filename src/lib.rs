pub mod mmap_slice;
pub use sampler::Sampler;
pub use in_memory_index::InMemoryIndex;
pub use memmap_index::MemmapIndex;
pub use sharded_memmap_index::ShardedMemmapIndex;
pub use table::SuffixTable;
pub use countable::CountableIndex;

/// Python bindings
use pyo3::prelude::*;

mod sharded_memmap_index;
mod in_memory_index;
mod memmap_index;
mod countable;
mod sampler;
mod table;
mod par_quicksort;
mod util;

#[macro_use]
extern crate derive_builder;

#[pymodule]
fn tokengrams(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<InMemoryIndex>()?;
    m.add_class::<MemmapIndex>()?;
    m.add_class::<ShardedMemmapIndex>()?;
    m.add_class::<Sampler>()?;
    Ok(())
}
