use std::path::PathBuf;

use bincode::{deserialize_from, serialize_into, serialized_size};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::gram::{Gram, TokenGram};
use crate::gram_counter::GramCounter;
use crate::loader::{GramsFileFormats, GramsTextLoader};
use crate::rank_array::EliasFanoRankArray;
use crate::trie_array::EliasFanoTrieArray;
use crate::trie_lm::{TrieLm, TrieLmBuilder, TrieLmLookuper};
use crate::vocabulary::{DoubleArrayVocabulary, IdentityVocabulary};
use crate::WordGram;

pub type TokenTrieLm = TrieLm<EliasFanoTrieArray, IdentityVocabulary, EliasFanoRankArray>;
pub type WordTrieLm = TrieLm<EliasFanoTrieArray, DoubleArrayVocabulary, EliasFanoRankArray>;

#[pyclass(frozen)]
pub struct TokenTrie {
    trie: TokenTrieLm,
}

#[pymethods]
impl TokenTrie {
    #[staticmethod]
    pub fn from_file(path: String, max_tokens: Option<u64>) -> PyResult<Self> {
        let mut counter = GramCounter::<u16, 3>::new();
        counter.count_file(&path, max_tokens)?;

        Ok(Self {
            trie: TrieLmBuilder::new(vec![&counter])?.build()?,
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

    pub fn __repr__(&self) -> String {
        format!(
            "TokenTrie: {} bytes",
            serialized_size(&self.trie).unwrap_or(0)
        )
    }

    /// Loads a serialized trie from a file.
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let mut file = std::fs::File::open(path)?;
        Ok(Self {
            trie: deserialize_from(&mut file).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Serializes the trie to a file.
    pub fn save(&self, path: String) -> PyResult<()> {
        let file = std::fs::File::create(path)?;
        serialize_into(file, &self.trie).map_err(|e| PyValueError::new_err(e.to_string()))
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
            trie: WordTrieLm::from_files(
                &filepaths,
                match fmt {
                    "plain" => GramsFileFormats::Plain,
                    "gzip" => GramsFileFormats::Gzip,
                    _ => return Err(PyValueError::new_err("Invalid format")),
                },
            )?,
        })
    }

    #[staticmethod]
    pub fn from_texts(texts: Vec<String>) -> PyResult<Self> {
        let mut loaders = Vec::with_capacity(texts.len());
        for text in texts {
            // Hack to make GramsTextLoader work with a static lifetime.
            let leaked = Box::leak(text.into_boxed_str());
            let loader = GramsTextLoader::new(leaked.as_bytes());
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
}
