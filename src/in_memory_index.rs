use anyhow::Result;
use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use crate::sample::{KneserNeyCache, Sample};
use crate::table::SuffixTable;
use crate::util::transmute_slice;

/// An in-memory index exposes suffix table functionality over text corpora small enough to fit in memory.
macro_rules! create_in_memory_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            table: SuffixTable<Box<[$type]>, Box<[u64]>>,
            cache: KneserNeyCache,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (tokens, verbose=false))]
            pub fn new_py(_py: Python, tokens: Vec<$type>, verbose: bool) -> Self {
                $name {
                    table: SuffixTable::<Box<[$type]>>::new(tokens, verbose),
                    cache: KneserNeyCache::default(),
                }
            }

            #[staticmethod]
            pub fn from_pretrained(path: String) -> PyResult<Self> {
                let table: SuffixTable<Box<[$type]>, Box<[u64]>> = deserialize(&std::fs::read(path)?).unwrap();
                Ok($name {
                    table,
                    cache: KneserNeyCache::default(),
                })
            }

            #[staticmethod]
            #[pyo3(signature = (path, verbose=false, token_limit=None))]
            pub fn from_token_file(
                path: String,
                verbose: bool,
                token_limit: Option<usize>,
            ) -> PyResult<Self> {
                let mut buffer = Vec::new();
                let mut file = File::open(&path)?;

                if let Some(max_tokens) = token_limit {
                    let max_bytes = max_tokens * std::mem::size_of::<$type>();
                    file.take(max_bytes as u64).read_to_end(&mut buffer)?;
                } else {
                    file.read_to_end(&mut buffer)?;
                };

                Ok($name {
                    table: SuffixTable::new(transmute_slice(buffer.as_slice()), verbose),
                    cache: KneserNeyCache::default(),
                })
            }

            pub fn is_sorted(&self) -> bool {
                self.table.is_sorted()
            }

            pub fn contains(&self, query: Vec<$type>) -> bool {
                self.table.contains(&query)
            }

            pub fn positions(&self, query: Vec<$type>) -> Vec<u64> {
                self.table.positions(&query).to_vec()
            }

            pub fn count(&self, query: Vec<$type>) -> usize {
                self.table.positions(&query).len()
            }

            #[pyo3(signature = (query, vocab=None))]
            pub fn count_next(&self, query: Vec<$type>, vocab: Option<usize>) -> Vec<usize> {
                self.table.count_next(&query, vocab)
            }

            #[pyo3(signature = (queries, vocab=None))]
            pub fn batch_count_next(&self, queries: Vec<Vec<$type>>, vocab: Option<usize>) -> Vec<Vec<usize>> {
                self.table.batch_count_next(&queries, vocab)
            }

            pub fn save(&self, path: String) -> PyResult<()> {
                let bytes = serialize(&self.table).unwrap();
                std::fs::write(&path, bytes)?;
                Ok(())
            }

            #[pyo3(signature = (query, n, k, num_samples, vocab=None))]
            pub fn sample_unsmoothed(
                &self,
                query: Vec<$type>,
                n: usize,
                k: usize,
                num_samples: usize,
                vocab: Option<usize>,
            ) -> Result<Vec<Vec<$type>>> {
                self.sample_unsmoothed_rs(&query, n, k, num_samples, vocab)
            }

            #[pyo3(signature = (query, vocab=None))]
            pub fn get_smoothed_probs(&mut self, query: Vec<$type>, vocab: Option<usize>) -> Vec<f64> {
                self.get_smoothed_probs_rs(&query, vocab)
            }

            #[pyo3(signature = (queries, vocab=None))]
            pub fn batch_get_smoothed_probs(
                &mut self,
                queries: Vec<Vec<$type>>,
                vocab: Option<usize>,
            ) -> Vec<Vec<f64>> {
                self.batch_get_smoothed_probs_rs(&queries, vocab)
            }

            #[pyo3(signature = (query, n, k, num_samples, vocab=None))]
            pub fn sample_smoothed(
                &mut self,
                query: Vec<$type>,
                n: usize,
                k: usize,
                num_samples: usize,
                vocab: Option<usize>,
            ) -> Result<Vec<Vec<$type>>> {
                self.sample_smoothed_rs(&query, n, k, num_samples, vocab)
            }

            pub fn estimate_deltas(&mut self, n: usize) {
                self.estimate_deltas_rs(n);
            }
        }

        impl Sample<$type> for $name {
            fn get_cache(&self) -> &KneserNeyCache {
                &self.cache
            }

            fn get_mut_cache(&mut self) -> &mut KneserNeyCache {
                &mut self.cache
            }

            fn count_next_slice(&self, query: &[$type], vocab: Option<usize>) -> Vec<usize> {
                self.table.count_next(&query, vocab)
            }

            fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
                self.table.count_ngrams(n)
            }
        }

        impl $name {
            pub fn new(tokens: Vec<$type>, verbose: bool) -> Self {
                $name {
                    table: SuffixTable::new(tokens, verbose),
                    cache: KneserNeyCache::default(),
                }
            }
        }
    };
}

create_in_memory_interface!(InMemoryIndexU16, u16);
create_in_memory_interface!(InMemoryIndexU32, u32);
