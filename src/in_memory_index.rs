use anyhow::Result;
use funty::Unsigned;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::Read;

use crate::bindings::in_memory_index::InMemoryIndexTrait;
use crate::mmap_slice::MmapSliceMut;
use crate::sample::{KneserNeyCache, Sample};
use crate::table::SuffixTable;
use crate::util::transmute_slice;

/// An in-memory index exposes suffix table functionality over text corpora small enough to fit in memory.
pub struct InMemoryIndexRs<T: Unsigned> {
    table: SuffixTable<Box<[T]>, Box<[u64]>>,
    cache: KneserNeyCache,
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
            cache: KneserNeyCache::default(),
        })
    }

    fn read_file_to_boxed_slice<E: Unsigned>(path: &str) -> Result<Box<[E]>> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;

        // Ensure file size is a multiple of size of E
        if file_len % std::mem::size_of::<T>() != 0 {
            anyhow::bail!("File size is not a multiple of element size");
        }

        let num_elements = file_len / std::mem::size_of::<E>();
        let mut vec: Vec<E> = Vec::with_capacity(num_elements);
        unsafe {
            let buf = std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, file_len);
            file.read_exact(buf)?;
            vec.set_len(num_elements);
        }

        Ok(vec.into_boxed_slice())
    }

    pub fn from_disk(text_path: String, table_path: String, vocab: usize) -> PyResult<Self> {
        let text = Self::read_file_to_boxed_slice::<T>(&text_path)?;
        let table = Self::read_file_to_boxed_slice::<u64>(&table_path)?;

        let suffix_table = SuffixTable::from_parts(text, table, Some(vocab));
        debug_assert!(suffix_table.is_sorted());

        Ok(InMemoryIndexRs {
            table: suffix_table,
            cache: KneserNeyCache::default(),
        })
    }

    pub fn save_text(&self, path: String) -> Result<()> {
        let text = self.table.get_text();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;

        let file_len = text.len() * std::mem::size_of::<T>();
        file.set_len(file_len as u64)?;

        let mut mmap = MmapSliceMut::<T>::new(&file)?;
        mmap.copy_from_slice(text);
        mmap.flush()?;

        Ok(())
    }

    pub fn save_table(&self, path: String) -> Result<()> {
        let table = self.table.get_table();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;

        file.set_len((table.len() * 8) as u64)?;

        let mut mmap = MmapSliceMut::<u64>::new(&file)?;
        mmap.copy_from_slice(table);
        mmap.flush()?;

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
    fn save_table(&self, table_path: String) -> Result<()> {
        self.save_table(table_path)
    }

    fn save_text(&self, text_path: String) -> Result<()> {
        self.save_text(text_path)
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

    fn positions(&self, query: Vec<usize>) -> Vec<u64> {
        let query: Vec<T> = query
            .iter()
            .filter_map(|&item| T::try_from(item).ok())
            .collect();
        self.table.positions(&query).to_vec()
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


#[cfg(test)]
pub mod tests {
    use super::*;
    use utf16_literal::utf16;
    use crate::table::SuffixTable;

    fn sais(text: &str) -> SuffixTable {
        SuffixTable::new(text.encode_utf16().collect::<Vec<_>>(), None, false)
    }

    fn utf16_as_usize(s: &str) -> Vec<usize> {
        s.encode_utf16().map(|x| x as usize).collect()
    }

    #[test]
    fn sample_unsmoothed_empty_query_exists() {
        let s = utf16!("aaa");
        let index: Box<dyn Sample<u16>> = Box::new(InMemoryIndexRs::new(s.to_vec(), None, false));

        let seqs = index.sample_unsmoothed(&[], 3, 10, 1).unwrap();

        assert_eq!(*seqs[0].last().unwrap(), s[0]);
    }

    #[test]
    fn sample_unsmoothed_u16_exists() {
        let s = utf16!("aaaa");
        let a = &s[0..1];
        let index: Box<dyn Sample<u16>> = Box::new(InMemoryIndexRs::new(s.to_vec(), None, false));

        let seqs = index.sample_unsmoothed(a, 3, 10, 1).unwrap();

        assert_eq!(*seqs[0].last().unwrap(), a[0]);
    }

    #[test]
    fn sample_unsmoothed_u32_exists() {
        let s: Vec<u32> = "aaaa".encode_utf16().map(|c| c as u32).collect();
        let u32_vocab = Some(u16::MAX as usize + 2);
        let index: Box<dyn Sample<u32>> = Box::new(InMemoryIndexRs::<u32>::new(s.clone(), u32_vocab, false));

        let seqs = index.sample_unsmoothed(&s[0..1], 3, 10, 1).unwrap();

        assert_eq!(*seqs[0].last().unwrap(), s[0]);
    }

    #[test]
    fn sample_unsmoothed_usize_exists() {
        let s = utf16_as_usize("aaaa");
        let index: Box<dyn InMemoryIndexTrait> = Box::new(InMemoryIndexRs::new(s.to_vec(), None, false));

        let seqs = index.sample_unsmoothed(s[0..1].to_vec(), 3, 10, 1).unwrap();

        assert_eq!(*seqs[0].last().unwrap(), s[0]);
    }

    #[test]
    fn sample_smoothed_exists() {
        let s = utf16!("aabbccabccba");
        let mut index: Box<dyn Sample<u16>> = Box::new(InMemoryIndexRs::new(s.to_vec(), None, false));

        let tokens = &index.sample_smoothed(&s[0..1], 3, 10, 1).unwrap()[0];

        assert_eq!(tokens.len(), 11);
    }

    #[test]
    fn sample_smoothed_empty_query_exists() {
        let s: Vec<u16> = "aabbccabccba".encode_utf16().collect();
        let mut index: Box<dyn Sample<u16>> = Box::new(InMemoryIndexRs::new(s, None, false));

        let tokens = &index.sample_smoothed(&[], 1, 10, 10).unwrap()[0];

        assert_eq!(tokens.len(), 10);
    }

    #[test]
    fn smoothed_probs_exists() {
        let tokens = "aaaaaaaabc".to_string();
        let tokens_vec: Vec<u16> = tokens.encode_utf16().collect();
        let query: Vec<_> = vec![utf16!("b")[0]];

        // Get unsmoothed probs for query
        let sa: SuffixTable = sais(&tokens);
        let bigram_counts = sa.count_next(&query);
        let unsmoothed_probs = bigram_counts
            .iter()
            .map(|&x| x as f64 / bigram_counts.iter().sum::<usize>() as f64)
            .collect::<Vec<f64>>();

        // Get smoothed probs for query
        let mut index: Box<dyn Sample<u16>> = Box::new(InMemoryIndexRs::new(tokens_vec, None, false));
        let smoothed_probs = index.get_smoothed_probs(&query);

        // Compare unsmoothed and smoothed probabilities
        let a = utf16!("a")[0] as usize;
        let c = utf16!("c")[0] as usize;

        // The naive bigram probability for query 'b' is p(c) = 1.0.
        assert!(unsmoothed_probs[a] == 0.0);
        assert!(unsmoothed_probs[c] == 1.0);

        // The smoothed bigram probabilities interpolate with the lower-order unigram
        // probabilities where p(a) is high, lowering p(c)
        assert!(smoothed_probs[a] > 0.1);
        assert!(smoothed_probs[c] < 1.0);
    }
}
