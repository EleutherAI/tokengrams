use anyhow::Result;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use funty::Unsigned;
use std::io::Write;
use std::io::Read;
use rayon::prelude::*;
use std::fmt::Debug;

use crate::sample::{KneserNeyCache, Sample};
use crate::table::SuffixTable;
use crate::util::transmute_slice;

pub trait InMemoryIndexTrait {
    fn save(&self, text_path: String, table_path: String) -> PyResult<()>;
    fn is_sorted(&self) -> bool;
    fn contains(&self, query: Vec<usize>) -> bool;
    fn positions(&self, query: Vec<usize>) -> Vec<u64>;
    fn count(&self, query: Vec<usize>) -> usize;
    fn count_next(&self, query: Vec<usize>) -> Vec<usize>;
    fn batch_count_next(&self, queries: Vec<Vec<usize>>) -> Vec<Vec<usize>>;
    fn sample_unsmoothed(
        &self, query: Vec<usize>, n: usize, k: usize, num_samples: usize
    ) -> Result<Vec<Vec<usize>>>;
    fn sample_smoothed(
        &mut self, query: Vec<usize>, n: usize, k: usize, num_samples: usize
    ) -> Result<Vec<Vec<usize>>>;
    fn get_smoothed_probs(&mut self, query: Vec<usize>) -> Vec<f64>;
    fn batch_get_smoothed_probs(
        &mut self, queries: Vec<Vec<usize>>
    ) -> Vec<Vec<f64>>;
    fn estimate_deltas(&mut self, n: usize);
}

/// An in-memory index exposes suffix table functionality over text corpora small enough to fit in memory.
pub struct InMemoryIndexRs<T: Unsigned> {
    table: SuffixTable<Box<[T]>, Box<[u64]>>,
    cache: KneserNeyCache
}

impl<T: Unsigned + Debug> InMemoryIndexRs<T> {
    pub fn new(tokens: Vec<T>, vocab: Option<usize>, verbose: bool) -> Self {
        let vocab = vocab.unwrap_or(u16::MAX as usize + 1);
    
        let table = SuffixTable::new(tokens, Some(vocab), verbose);
        debug_assert!(table.is_sorted());

        InMemoryIndexRs {
            table,
            cache: KneserNeyCache::default(),
        }
    }

    pub fn from_token_file(
        path: String,
        token_limit: Option<usize>,
        vocab: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let mut buffer = Vec::new();
        let mut file = File::open(&path)?;

        if let Some(max_tokens) = token_limit {
            // Limit on the number of tokens to consider is provided
            let max_bytes = max_tokens * std::mem::size_of::<T>();
            file.take(max_bytes as u64).read_to_end(&mut buffer)?;
        } else {
            file.read_to_end(&mut buffer)?;
        };

        let tokens = transmute_slice::<u8, T>(buffer.as_slice());
        let table = SuffixTable::new(tokens, Some(vocab), verbose);
        debug_assert!(table.is_sorted());

        Ok(InMemoryIndexRs {
            table,
            cache: KneserNeyCache::default()
        })
    }

    pub fn from_parts(text: Vec<T>, table: Vec<u64>, vocab: Option<usize>) -> Self {
        let vocab = vocab.unwrap_or(u16::MAX as usize + 1);
    
        let table = SuffixTable::from_parts(text.into_boxed_slice(), table.into_boxed_slice(), Some(vocab));
        debug_assert!(table.is_sorted());
    
        InMemoryIndexRs {
            table,
            cache: KneserNeyCache::default(),
        }
    }

    pub fn save(&self, text_path: String, table_path: String) -> PyResult<()> {
        fn write_data<T>(path: &str, data: &[T]) -> PyResult<()> {
            let mut file = File::create(path)?;
            let bytes = transmute_slice::<T, u8>(data);
            file.write_all(&bytes)?;
            Ok(())
        }
    
        write_data(&table_path, self.table.get_table())?;
        write_data(&text_path, self.table.get_text())?;
    
        println!("Table of len {} saved to {}", self.table.get_table().len(), table_path);
        println!("Text of len {} saved to {}", self.table.get_text().len(), text_path);
    
        Ok(())
    }
}

impl<T: Unsigned> Sample<T> for InMemoryIndexRs<T> {
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

impl<T: Unsigned> InMemoryIndexTrait for InMemoryIndexRs<T> {
    fn save(&self, text_path: String, table_path: String) -> PyResult<()> {
        self.save(text_path, table_path)
    }

    fn is_sorted(&self) -> bool {
        self.table.is_sorted()
    }

    fn contains(&self, query: Vec<usize>) -> bool {
        let query: Vec<T> = query.iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        self.table.contains(&query)
    }

    fn positions(&self, query: Vec<usize>) -> Vec<u64> {
        let query: Vec<T> = query.iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        self.table.positions(&query).to_vec()
    }

    fn count(&self, query: Vec<usize>) -> usize {
        let query: Vec<T> = query.iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        self.table.positions(&query).len()
    }

    fn count_next(&self, query: Vec<usize>) -> Vec<usize> {
        let query: Vec<T> = query.iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        self.table.count_next(&query)
    }

    fn batch_count_next(&self, queries: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        queries
            .into_par_iter()
            .map(|query| {
                self.count_next(query)
            })
            .collect()
    }

    fn sample_smoothed(
        &mut self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize
    ) -> Result<Vec<Vec<usize>>> {
        let query: Vec<T> = query.iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();

        let samples_batch = <Self as Sample<T>>::sample_smoothed(self, &query, n, k, num_samples)?;
        Ok(samples_batch.into_iter().map(|samples| {
            samples.into_iter()
                .filter_map(|sample| {
                    match TryInto::<usize>::try_into(sample) {
                        Ok(value) => Some(value),
                        Err(_) => None, // Silently skip values that can't be converted
                    }
                })
                .collect::<Vec<usize>>()
        }).collect())
    }

    fn sample_unsmoothed(
        &self,
        query: Vec<usize>,
        n: usize,
        k: usize,
        num_samples: usize
    ) -> Result<Vec<Vec<usize>>> {
        let query: Vec<T> = query.iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();

        let samples_batch = <Self as Sample<T>>::sample_unsmoothed(self, &query, n, k, num_samples)?;
        Ok(samples_batch.into_iter().map(|samples| {
            samples.into_iter()
                .filter_map(|sample| {
                    match TryInto::<usize>::try_into(sample) {
                        Ok(value) => Some(value),
                        Err(_) => None, // Silently skip values that can't be converted
                    }
                })
                .collect::<Vec<usize>>()
        }).collect())
    }

    fn get_smoothed_probs(&mut self, query: Vec<usize>) -> Vec<f64> {
        let query: Vec<T> = query.iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();

            <Self as Sample<T>>::get_smoothed_probs(self, &query)
    }

    fn batch_get_smoothed_probs(
        &mut self,
        queries: Vec<Vec<usize>>
    ) -> Vec<Vec<f64>> {
        let queries: Vec<Vec<T>> = queries.into_iter()
            .map(|query| {
                query.iter()
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