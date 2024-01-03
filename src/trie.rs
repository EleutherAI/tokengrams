use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::loader::{GramsFileFormats, GramsLoader, GramsTextLoader};
use crate::trie_array::EliasFanoTrieArray;
use crate::trie_count_lm::{TrieCountLm, TrieCountLmBuilder, TrieCountLmLookuper};
use crate::rank_array::EliasFanoRankArray;
use crate::vocabulary::DoubleArrayVocabulary;

/// Elias-Fano Trie implementation of [`TrieCountLm`].
/// This configuration is similar to `ef_trie_PSEF_ranks_count_lm` in the original `tongrams`.
pub type EliasFanoTrieCountLm =
    TrieCountLm<EliasFanoTrieArray, DoubleArrayVocabulary, EliasFanoRankArray>;

#[pyclass(frozen)]
pub struct Trie {
    trie: EliasFanoTrieCountLm,
}

#[pymethods]
impl Trie {
    #[staticmethod]
    pub fn from_files(filepaths: Vec<PathBuf>, fmt: &str) -> PyResult<Self> {
        Ok(Self {
            trie: EliasFanoTrieCountLm::from_files(&filepaths, match fmt {
                "plain" => GramsFileFormats::Plain,
                "gzip" => GramsFileFormats::Gzip,
                _ => return Err(PyValueError::new_err("Invalid format")),
            })?,
        })
    }

    #[staticmethod]
    pub fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
        Ok(Self {
            trie: EliasFanoTrieCountLm::deserialize_from(&bytes[..])?,
        })
    }

    #[staticmethod]
    pub fn from_texts(texts: Vec<String>) -> PyResult<Self> {
        let mut loaders = Vec::with_capacity(texts.len());
        for text in texts {
            // Hack to make GramsTextLoader work with a static lifetime.
            let leaked = Box::leak(text.into_boxed_str());
            let loader: Box<dyn GramsLoader<_>> = Box::new(GramsTextLoader::new(leaked.as_bytes()));
            loaders.push(loader);
        }
        Ok(Self {
            trie: TrieCountLmBuilder::new(loaders)?.build()?,
        })
    }

    pub fn find(&self, gram: &str) -> Option<usize> {
        let mut lookuper = TrieCountLmLookuper::new(&self.trie);
        lookuper.with_str(gram)
    }

    pub fn num_orders(&self) -> usize {
        self.trie.num_orders()
    }

    pub fn num_grams(&self) -> usize {
        self.trie.num_grams()
    }

    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let mut bytes = Vec::new();
        self.trie.serialize_into(&mut bytes)?;
        Ok(bytes)
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn tokengrams(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Trie>()?;
    Ok(())
}
