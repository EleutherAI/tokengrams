use core::fmt::Debug;
use std::hash::Hash;
use std::iter::Iterator;
use std::fmt::{Display, Formatter};

use ahash::AHashMap;
use bincode::{deserialize_from, serialize_into}2;
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::exceptions::PyValueError;
use pyo3::ffi::PyErr_CheckSignals;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};


use memmap2::Mmap;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::marker::PhantomData;

pub struct TypedMmap<T> {
    inner: Mmap,

    // Workaround for the lack of generic associated types
    _marker: PhantomData<T>,
}

impl<T> TypedMmap<T> {
    const SIZE: usize = std::mem::size_of::<T>();

    pub fn new(file: File) -> std::io::Result<Self> {
        let inner = unsafe { Mmap::map(&file)? };

        if inner.len() % Self::SIZE != 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "mmap size is not a multiple of the type size",
            ));
        }
        Ok(Self { inner, _marker: PhantomData, })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len() / Self::SIZE
    }

    pub fn as_slice(&self) -> &[T] {
        let ptr = self.inner.as_ptr() as *const T;

        // SAFETY: We checked that the length is a multiple of the type size
        unsafe { std::slice::from_raw_parts(ptr, self.len()) }
    }
}


/// Counter of N-grams of type T for a fixed N.
#[derive(Debug, Clone)]
pub struct GramCounter<T: Copy + Eq + Hash, const N: usize> {
    /// Grams so frequent that they overflow to 64 bits.
    frequent: AHashMap<[T; N], u64>,

    /// Grams that fit in 32 bits.
    common: AHashMap<[T; N], u32>,

    /// Grams that fit in 16 bits.
    rare: AHashMap<[T; N], u16>,
}

impl<T: Copy + Eq + Hash, const N: usize> GramCounter<T, N> {
    /// The number of bytes in an n-gram.
    const GRAM_SIZE: usize = std::mem::size_of::<T>() * N;

    /// Create a new counter.
    pub fn new() -> Self {
        Self {
            frequent: AHashMap::new(),
            common: AHashMap::new(),
            rare: AHashMap::new(),
        }
    }

    /// Get the count of a gram.
    pub fn count(&self, gram: &[T; N]) -> u64 {
        // Check if the gram is in the frequent, common, or rare maps.
        if let Some(count) = self.frequent.get(gram) {
            *count as u64
        } else if let Some(count) = self.common.get(gram) {
            *count as u64
        } else if let Some(count) = self.rare.get(gram) {
            *count as u64
        } else {
            0
        }
    }

    /// Increment the count of a gram by 1.
    pub fn increment(&mut self, gram: &[T; N]) {
        // Check the maps in reverse order: rare, common, frequent.
        // For N > 1, the "rare" map is actually the most common.
        if let Some(count) = self.rare.get_mut(gram) {
            // Try to increment the count, checking for overflow.
            if let Some(res) = count.checked_add(1) {
                *count = res;
            } else {
                // If overflow, remove from rare map and add to common map.
                let count = self.rare.remove(gram).unwrap();
                self.common.insert(*gram, count as u32 + 1);
            }
        } else if let Some(count) = self.common.get_mut(gram) {
            // Try to increment the count, checking for overflow.
            if let Some(res) = count.checked_add(1) {
                *count = res;
            } else {
                // If overflow, remove from common map and add to frequent map.
                let count = self.common.remove(gram).unwrap();
                self.frequent.insert(*gram, count as u64 + 1);
            }
        }
        else if let Some(count) = self.frequent.get_mut(gram) {
            // If `gram` is in the frequent map, we don't worry about overflow.
            *count += 1;
        } else {
            // If not, add it to the rare map.
            self.rare.insert(*gram, 1);
        }
    }

    /// Iterate over all n-grams and their counts.
    pub fn iter(&self) -> impl Iterator<Item = (&[T; N], u64)> {
        self.frequent.iter().map(|(k, v)| (k, *v))
            .chain(self.common.iter().map(|(k, v)| (k, *v as u64)))
            .chain(self.rare.iter().map(|(k, v)| (k, *v as u64)))
    }

    /// The number of n-grams in the counter.
    pub fn len(&self) -> usize {
        self.frequent.len() + self.common.len() + self.rare.len()
    }

    /*pub fn merge(&mut self, other: &Self) {
        // Don't have to worry about overflow for the frequent map.
        for (gram, count) in other.iter() {
            *self.frequent.entry(*gram).or_insert(0) += count;
        }
        for (gram, count) in &other.frequent {
            *self.frequent.entry(*gram).or_insert(0) += count;
        }
        for (gram, count) in &other.common {
            self.common.entry(*gram).and_modify(|c| *c += count);
        }e
        for (gram, count) in &other.rare {
            self.rare.entry(*gram).and_modify(|c| *c += count);
        }
    }*/

    /// Count n-grams from a stream, consuming it.
    pub fn update_from<I: Iterator<Item = T>>(&mut self, mut stream: I) {
        // Fill buffer with first n-gram and count it.
        let buf: Vec<_> = stream.by_ref().take(N).collect();
        let mut arr: [T; N] = match buf.try_into() {
            Ok(arr) => arr,
            Err(_) => return, // Do nothing if stream is too short.
        };
        self.increment(&arr);

        let pb = ProgressBar::new(600_000_000_000);
        pb.set_style(
            ProgressStyle::with_template("[{elapsed_precise}/{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap()
        );

        // Count n-grams.
        for (i, x) in stream.enumerate() {
            // Update buffer of last n elements.
            arr.rotate_left(1);
            arr[N - 1] = x;

            self.increment(&arr);

            if i % 100_000 == 0 {
                pb.inc(100_000);

                let msg = format!("{} common, {} rare", self.common.len(), self.rare.len());
                pb.set_message(msg);

                // Check for Ctrl-C
                if unsafe { PyErr_CheckSignals() } != 0 {
                    break;
                }
            }
        }
    }
}

impl<T: Copy + Eq + Hash, const N: usize> Serialize for GramCounter<T, N> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::new();
        serialize_into(&mut bytes, self).map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de, T: Copy + Eq + Hash, const N: usize> Deserialize<'de> for GramCounter<T, N> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = <&[u8]>::deserialize(deserializer)?;
        deserialize_from(bytes).map_err(serde::de::Error::custom)
    }
}

impl<const N: usize> GramCounter<u16, N> {
    pub fn count_file(path: &str) -> std::io::Result<()> {
        let mmap = TypedMmap::<u16>::new(File::open(path)?)?;
        self.update_from(mmap.as_slice().iter().copied());
        Ok(())
    }

    /// Read a file of raw u16s into the counter.
    pub fn read_file(&mut self, path: &str) -> std::io::Result<()> {
        let mmap = TypedMmap::<u16>::new(File::open(path)?)?;
        self.update_from(mmap.as_slice().iter().copied());
        Ok(())
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
                assert_eq!(counter.count(arr), 1);
            }
            counter.update_from(stream.clone().into_iter());
    
            for w in stream.as_slice().windows($window_size) {
                let arr = w.try_into().unwrap();
                assert_eq!(counter.count(arr), 2);
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
        #[derive(Deserialize, Serialize)]
        pub struct $name {
            inner: GramCounter<u16, $size>,
        }

        #[pymethods]
        impl $name {
            #[staticmethod]
            pub fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
                deserialize_from(&bytes[..]).map_err(|e| PyValueError::new_err(e.to_string()))
            }

            #[staticmethod]
            pub fn from_file(path: &str) -> PyResult<Self> {
                let mut counter = GramCounter::<u16, $size>::new();
                counter.read_file(path)?;
                Ok(Self { inner: counter })
            }

            pub fn count(&self, gram: Vec<u16>) -> PyResult<u64> {
                let arr = &gram.try_into().map_err(|_| PyValueError::new_err(format!("gram must be a {}-gram", $size)))?;
                Ok(self.inner.count(arr))
            }

            pub fn increment(&mut self, gram: Vec<u16>) -> PyResult<()> {
                let arr = &gram.try_into().map_err(|_| PyValueError::new_err(format!("gram must be a {}-gram", $size)))?;
                self.inner.increment(arr);
                Ok(())
            }

            pub fn __len__(&self) -> usize {
                self.inner.len()
            }

            pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
                let mut bytes = Vec::new();
                serialize_into(&mut bytes, &self.inner).map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(bytes)
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
