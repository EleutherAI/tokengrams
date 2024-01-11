use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::WordGram;
use crate::gram::{Gram, TokenGram};
use crate::loader::{GramsFileFormats, GramsLoader, GramsTextLoader};
use crate::trie_array::EliasFanoTrieArray;
use crate::trie_lm::{TrieLm, TrieLmBuilder, TrieLmLookuper};
use crate::rank_array::EliasFanoRankArray;
use crate::vocabulary::{DoubleArrayVocabulary, IdentityVocabulary};

pub type TokenTrieLm = TrieLm<EliasFanoTrieArray, IdentityVocabulary, EliasFanoRankArray>;
pub type WordTrieLm =
    TrieLm<EliasFanoTrieArray, DoubleArrayVocabulary, EliasFanoRankArray>;


#[pyclass(frozen)]
pub struct TokenTrie {
    trie: TokenTrieLm,
}

#[pymethods]
impl TokenTrie {
    #[staticmethod]
    pub fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
        Ok(Self {
            trie: TokenTrieLm::deserialize_from(&bytes[..])?,
        })
    }

    pub fn find(&self, gram: Vec<u16>) -> Option<usize> {
        let mut lookuper = TrieLmLookuper::new(&self.trie);
        lookuper.with_gram(TokenGram::new(gram))
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


#[pyclass(frozen)]
pub struct WordTrie {
    trie: WordTrieLm,
}

#[pymethods]
impl WordTrie {
    #[staticmethod]
    pub fn from_files(filepaths: Vec<PathBuf>, fmt: &str) -> PyResult<Self> {
        Ok(Self {
            trie: WordTrieLm::from_files(&filepaths, match fmt {
                "plain" => GramsFileFormats::Plain,
                "gzip" => GramsFileFormats::Gzip,
                _ => return Err(PyValueError::new_err("Invalid format")),
            })?,
        })
    }

    #[staticmethod]
    pub fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
        Ok(Self {
            trie: WordTrieLm::deserialize_from(&bytes[..])?,
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
            trie: TrieLmBuilder::new(loaders)?.build()?,
        })
    }

    pub fn find(&self, gram: &str) -> Option<usize> {
        let mut lookuper = TrieLmLookuper::new(&self.trie);
        lookuper.with_gram(WordGram::from_str(gram))
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