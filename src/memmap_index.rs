use anyhow::Result;
use funty::Unsigned;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::time::Instant;

use crate::bindings::memmap_index::MemmapIndexTrait;
use crate::mmap_slice::{MmapSlice, MmapSliceMut};
use crate::par_quicksort::par_sort_unstable_by_key;
use crate::sample::{KneserNeyCache, Sample};
use crate::table::SuffixTable;

/// A memmap index exposes suffix table functionality over text corpora too large to fit in memory.
pub struct MemmapIndexRs<T: Unsigned> {
    table: SuffixTable<MmapSlice<T>, MmapSlice<u64>>,
    cache: KneserNeyCache,
}

impl<T: Unsigned> MemmapIndexRs<T> {
    pub fn new(text_path: String, table_path: String, vocab: usize) -> Result<Self> {
        let text_file = File::open(&text_path)?;
        let table_file = File::open(&table_path)?;

        let table = SuffixTable::from_parts(
            MmapSlice::<T>::new(&text_file)?,
            MmapSlice::new(&table_file)?,
            Some(vocab),
        );
        assert!(table.is_sorted());

        Ok(MemmapIndexRs {
            table,
            cache: KneserNeyCache::default(),
        })
    }

    pub fn build(
        text_path: String,
        table_path: String,
        vocab: usize,
        verbose: bool,
    ) -> Result<Self> {
        // Memory map the text as read-only
        let text_mmap = MmapSlice::<T>::new(&File::open(&text_path).unwrap()).unwrap();

        let table_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&table_path)?;

        // Allocate space on disk for the table
        let table_size = text_mmap.len() * 8;
        table_file.set_len(table_size as u64)?;

        if verbose {
            println!("Writing indices to disk...");
        }
        let start = Instant::now();
        let mut table_mmap = MmapSliceMut::<u64>::new(&table_file)?;
        table_mmap
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = i as u64);

        assert_eq!(table_mmap.len(), text_mmap.len());
        if verbose {
            println!("Time elapsed: {:?}", start.elapsed());
        }
        let start = Instant::now();

        // TODO: Be even smarter about this? We may need to take into account the number of CPUs
        // available as well. These magic numbers were tuned on a server with 48 physical cores.
        // Empirically we start getting stack overflows between 5B and 10B tokens when using the
        // default stack size of 2MB. We scale the stack size as log2(n) * 8MB to avoid this.
        let scale = (text_mmap.len() as f64) / 5e9; // 5B tokens
        let stack_size = scale.log2().max(1.0) * 8e6; // 8MB

        rayon::ThreadPoolBuilder::new()
            .stack_size(stack_size as usize)
            .build()
            .unwrap()
            .install(|| {
                // Sort the indices by the suffixes they point to.
                // The unstable algorithm is critical for avoiding out-of-memory errors, since it does
                // not allocate any more memory than the input and output slices.
                println!("Sorting indices...");
                par_sort_unstable_by_key(
                    table_mmap.as_slice_mut(),
                    |&i| &text_mmap[i as usize..],
                    verbose,
                );
            });
        if verbose {
            println!("Time elapsed: {:?}", start.elapsed());
        }

        // Re-open the table as read-only
        let table_mmap = MmapSlice::new(&table_file)?;
        let table = SuffixTable::from_parts(text_mmap, table_mmap, Some(vocab));
        debug_assert!(table.is_sorted());

        Ok(MemmapIndexRs {
            table,
            cache: KneserNeyCache::default(),
        })
    }
}

impl<T: Unsigned> Sample<T> for MemmapIndexRs<T> {
    fn get_cache(&self) -> &KneserNeyCache {
        &self.cache
    }

    fn get_mut_cache(&mut self) -> &mut KneserNeyCache {
        &mut self.cache
    }

    fn count_next_slice(&self, query: &[T]) -> Vec<usize> {
        self.table.count_next(query)
    }

    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        self.table.count_ngrams(n)
    }
}

impl<T> MemmapIndexTrait for MemmapIndexRs<T>
where
    T: Unsigned,
{
    fn positions(&self, query: Vec<usize>) -> Vec<u64> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        self.table.positions(&query).to_vec()
    }

    fn is_sorted(&self) -> bool {
        self.table.is_sorted()
    }

    fn contains(&self, query: Vec<usize>) -> bool {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        self.table.contains(&query)
    }

    fn count(&self, query: Vec<usize>) -> usize {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        self.table.positions(&query).len()
    }

    fn count_next(&self, query: Vec<usize>) -> Vec<usize> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        self.table.count_next(&query)
    }

    fn batch_count_next(&self, queries: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        queries
            .into_par_iter()
            .map(|query| self.count_next(query))
            .collect()
    }

    fn sample_smoothed(
        &mut self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();

        let samples_batch = <Self as Sample<T>>::sample_smoothed(self, &query, n, k, num_samples)?;
        Ok(samples_batch
            .into_iter()
            .map(|samples| {
                samples
                    .into_iter()
                    .filter_map(|sample| {
                        match TryInto::<usize>::try_into(sample) {
                            Ok(value) => Some(value),
                            Err(_) => None, // Silently skip values that can't be converted
                        }
                    })
                    .collect::<Vec<usize>>()
            })
            .collect())
    }

    fn sample_unsmoothed(
        &self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();

        let samples_batch =
            <Self as Sample<T>>::sample_unsmoothed(self, &query, n, k, num_samples)?;
        Ok(samples_batch
            .into_iter()
            .map(|samples| {
                samples
                    .into_iter()
                    .filter_map(|sample| {
                        match TryInto::<usize>::try_into(sample) {
                            Ok(value) => Some(value),
                            Err(_) => None, // Silently skip values that can't be converted
                        }
                    })
                    .collect::<Vec<usize>>()
            })
            .collect())
    }

    fn get_smoothed_probs(&mut self, query: Vec<usize>) -> Vec<f64> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();

        <Self as Sample<T>>::get_smoothed_probs(self, &query)
    }

    fn batch_get_smoothed_probs(&mut self, queries: Vec<Vec<usize>>) -> Vec<Vec<f64>> {
        let queries: Vec<Vec<T>> = queries
            .into_iter()
            .map(|query| {
                query
                    .iter()
                    .filter_map(|&item| T::try_from(item).ok())
                    .collect()
            })
            .collect();
        <Self as Sample<T>>::batch_get_smoothed_probs(self, &queries)
    }

    fn estimate_deltas(&mut self, n: usize) {
        <Self as Sample<T>>::estimate_deltas(self, n)
    }
}
