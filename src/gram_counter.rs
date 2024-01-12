use core::fmt::Debug;
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::iter::Iterator;

use ahash::AHashMap;
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::exceptions::PyValueError;
use pyo3::ffi::PyErr_CheckSignals;
use pyo3::prelude::*;

use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::gram::TokenGram;
use crate::loader::GramSource;
use crate::record::CountRecord;
use crate::Gram;

/// Counter of N-grams of type T for a fixed N.
#[derive(Debug, Clone)]
pub struct GramCounter<T: Copy + Eq + Hash, const N: usize> {
    /// Map from n-gram to count
    table: AHashMap<[T; N], u32>,
}

impl<T: Copy + Eq + Hash, const N: usize> GramCounter<T, N> {
    const TOKEN_SIZE: usize = std::mem::size_of::<T>();

    /// Create a new counter.
    pub fn new() -> Self {
        Self {
            table: AHashMap::new(),
        }
    }

    /// Get immutable reference to the count of a gram.
    pub fn get(&self, gram: &[T; N]) -> &u32 {
        self.table.get(gram).unwrap_or(&0)
    }

    /// Get a mutable reference to the count of a gram. Unlike .get(), this will insert
    /// into the underlying table if the gram is not present.
    pub fn entry(&mut self, gram: &[T; N]) -> &mut u32 {
        self.table.entry(*gram).or_insert(0)
    }

    /// Iterate over all n-grams and their counts.
    pub fn iter(&self) -> impl Iterator<Item = (&[T; N], &u32)> {
        self.table.iter()
    }

    /// The number of n-grams in the counter.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn merge(&mut self, other: &Self) {
        for (gram, count) in other.iter() {
            *self.entry(gram) += count;
        }
    }

    /// Count n-grams from a stream, consuming it.
    pub fn update_from<I: Iterator<Item = T>>(&mut self, mut stream: I) {
        // Fill buffer with first n-gram and count it.
        let buf: Vec<_> = stream.by_ref().take(N).collect();
        let mut arr: [T; N] = match buf.try_into() {
            Ok(arr) => arr,
            Err(_) => return, // Do nothing if stream is too short.
        };
        *self.entry(&arr) += 1;

        // Count n-grams.
        for x in stream {
            // Update buffer of last n elements.
            arr.rotate_left(1);
            arr[N - 1] = x;

            *self.entry(&arr) += 1;
        }
    }

    /// Read a file of raw u16s into the counter.
    pub fn count_file(&mut self, path: &str, max_tokens: Option<u64>) -> std::io::Result<()> {
        let file = File::open(path)?;
        let total_tokens = std::cmp::min(
            file.metadata()?.len() / Self::TOKEN_SIZE as u64,
            max_tokens.unwrap_or(u64::MAX),
        );

        let mut reader = BufReader::new(file);
        let pb = ProgressBar::new(total_tokens);
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed}/{eta}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap(),
        );

        let mut tokens_processed = 0;
        while tokens_processed < total_tokens {
            let buffer = reader.fill_buf()?;
            let num_tokens = buffer.len() / Self::TOKEN_SIZE;

            // Couldn't read a full token
            if num_tokens == 0 {
                break;
            }
            // SAFETY: we used floor division above so we can't read past the end of the buffer
            let tokens =
                unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, num_tokens) };
            self.update_from(tokens.iter().copied()); // TODO: Maybe get rid of the copy here?

            // All bytes consumed from the buffer
            // should not be read again.
            reader.consume(num_tokens * Self::TOKEN_SIZE);
            tokens_processed += num_tokens as u64;

            // Check for Ctrl-C
            if unsafe { PyErr_CheckSignals() } != 0 {
                break;
            }

            let msg = format!("{} n-grams", self.len());
            pb.inc(num_tokens as u64);
            pb.set_message(msg);
        }
        Ok(())
    }
}

pub struct GramCounterIter<'a, T: Copy + Eq + Hash, const N: usize> {
    iter: std::collections::hash_map::Iter<'a, [T; N], u32>,
}

impl<'a, T: Copy + Eq + Hash, const N: usize> Iterator for GramCounterIter<'a, T, N> {
    type Item = anyhow::Result<CountRecord<TokenGram<T>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let (gram, count) = self.iter.next()?;
        Some(Ok(CountRecord {
            gram: TokenGram::new(*gram),
            count: *count as usize,
        }))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: Copy + Eq + Hash, const N: usize> GramSource for &'a GramCounter<T, N> {
    type GramType = TokenGram<T>;
    type Iter = GramCounterIter<'a, T, N>;

    fn iter(&self) -> anyhow::Result<Self::Iter> {
        Ok(GramCounterIter {
            iter: self.table.iter(),
        })
    }
}

#[cfg(test)]
mod tests {
    use rand::RngCore;

    use super::*;

    struct FakeStream {
        /// The number of unique in the stream.
        num_distinct: u16,
        num_total: u32,

        cursor: u32,
    }

    impl Iterator for FakeStream {
        type Item = u16;

        fn next(&mut self) -> Option<u16> {
            if self.cursor >= self.num_total {
                return None;
            }

            let mut rng = rand::thread_rng();
            Some((rng.next_u32() as u16) % self.num_distinct)
        }
    }

    /// Macro for testing a GramCounter with a given type and n-gram order.
    macro_rules! range_test {
        ($type:ty, $window_size:expr) => {
            let mut counter = GramCounter::<$type, $window_size>::new();

            let stream: Vec<_> = (0..=32000).collect();
            counter.update_from(stream.clone().into_iter());

            for w in stream.as_slice().windows($window_size) {
                let arr = w.try_into().unwrap();
                assert_eq!(*counter.get(arr), 1);
            }
            counter.update_from(stream.clone().into_iter());

            for w in stream.as_slice().windows($window_size) {
                let arr = w.try_into().unwrap();
                assert_eq!(*counter.get(arr), 2);
            }
        };
    }

    #[test]
    fn test_gram_counter() {
        range_test!(u16, 1);
        range_test!(u16, 2);
        range_test!(u16, 3);
    }
}

/// Python bindings
macro_rules! ngram_counter {
    ($name:ident, $size:expr) => {
        #[pyclass]
        pub struct $name {
            inner: GramCounter<u16, $size>,
        }

        #[pymethods]
        impl $name {
            #[staticmethod]
            pub fn from_file(path: &str, max_tokens: Option<u64>) -> PyResult<Self> {
                let mut counter = GramCounter::<u16, $size>::new();
                counter.count_file(path, max_tokens)?;
                Ok(Self { inner: counter })
            }

            pub fn count(&self, gram: Vec<u16>) -> PyResult<u32> {
                let arr = gram
                    .try_into()
                    .map_err(|_| PyValueError::new_err(format!("gram must be a {}-gram", $size)))?;
                Ok(*self.inner.get(&arr))
            }

            pub fn increment(&mut self, gram: Vec<u16>) -> PyResult<()> {
                let arr = gram
                    .try_into()
                    .map_err(|_| PyValueError::new_err(format!("gram must be a {}-gram", $size)))?;
                *self.inner.entry(&arr) += 1;
                Ok(())
            }

            pub fn __len__(&self) -> usize {
                self.inner.len()
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                writeln!(f, "{} grams", self.inner.len())?;
                Ok(())
            }
        }
    };
}

ngram_counter!(BigramCounter, 2);
ngram_counter!(TrigramCounter, 3);
ngram_counter!(FourgramCounter, 4);
ngram_counter!(FivegramCounter, 5);
ngram_counter!(SixgramCounter, 6);
