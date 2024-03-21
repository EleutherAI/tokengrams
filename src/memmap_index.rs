use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
//use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::{File, OpenOptions};

use crate::mmap_slice::{MmapSlice, MmapSliceMut};
use crate::table::SuffixTable;

#[pyclass]
pub struct MemmapIndex {
    table: SuffixTable<MmapSlice<u16>, MmapSlice<u64>>,
}

#[pymethods]
impl MemmapIndex {
    #[new]
    fn new(_py: Python, text_path: String, table_path: String) -> PyResult<Self> {
        let text_file = File::open(&text_path)?;
        let table_file = File::open(&table_path)?;

        Ok(MemmapIndex {
            table: SuffixTable::from_parts(
                MmapSlice::new(&text_file)?,
                MmapSlice::new(&table_file)?,
            ),
        })
    }

    #[staticmethod]
    fn build(text_path: String, table_path: String) -> PyResult<Self> {
        // Memory map the text as read-only
        let text_mmap = MmapSlice::new(&File::open(&text_path)?)?;
        let table_size = text_mmap.len() * std::mem::size_of::<u64>();

        // Memory map the table as read-write and write indices 0..n to the table
        let table_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&table_path)?;
        table_file.set_len(table_size as u64)?;
        println!("Writing indices to disk...");

        let pbar = ProgressBar::new(table_size as u64);
        pbar.set_style(ProgressStyle::with_template("{human_pos}/{human_len}").unwrap());

        let mut table_mmap = MmapSliceMut::<u64>::new(&table_file)?;
        table_mmap
            .par_iter_mut()
            .progress_with(pbar)
            .enumerate()
            .for_each(|(i, index)| {
                *index = i as u64;
            });

        // Sort the indices by the suffixes they point to.
        // The unstable algorithm is critical for avoiding out-of-memory errors, since it does
        // not allocate any more memory than the input and output slices.
        table_mmap.par_sort_unstable_by_key(|&i| &text_mmap[i as usize..]);

        Ok(MemmapIndex {
            table: SuffixTable::from_parts(text_mmap, table_mmap.into_read_only()?),
        })
    }

    fn contains(&self, query: Vec<u16>) -> bool {
        self.table.contains(&query)
    }

    fn count(&self, query: Vec<u16>) -> usize {
        self.table.positions(&query).len()
    }
}
