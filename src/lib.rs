pub mod mmap_slice;
pub use in_memory_index::InMemoryIndex;
pub use memmap_index::MemmapIndex;
pub use sharded_memmap_index::ShardedMemmapIndex;
pub use sampler_by_ref::SamplerByRef;
pub use table::SuffixTable;
pub use countable_index::CountableIndex;
pub use sampler_rs::SamplerRs;
pub use sampler::Sampler;

/// Python bindings
use pyo3::prelude::*;

mod sharded_memmap_index;
mod in_memory_index;
mod memmap_index;
mod sampler_by_ref;
mod countable;
mod countable_index;
mod sampler_rs;
mod sampler;
mod table;
mod par_quicksort;
mod util;

#[pymodule]
fn tokengrams(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InMemoryIndex>()?;
    m.add_class::<MemmapIndex>()?;
    m.add_class::<ShardedMemmapIndex>()?;
    m.add_class::<Sampler>()?;
    Ok(())
}
