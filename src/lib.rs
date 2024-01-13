pub mod ef_trie_array;
pub mod gram;
pub mod gram_counter;
pub mod loader;
pub mod parser;
pub mod record;
pub mod sucds_glue;
pub mod trie;
pub mod trie_lm;
pub mod util;
pub mod vocabulary;

mod mappers;

/// The maximum order of *N*-grams (i.e., `1 <= N <= 8`).
pub const MAX_ORDER: usize = 8;
/// The separator for tokens.
pub const TOKEN_SEPARATOR: u8 = b' ';
/// The separator for grams and count.
pub const GRAM_COUNT_SEPARATOR: u8 = b'\t';

pub use gram::{Gram, WordGram};
pub use gram_counter::{
    BigramCounter, FivegramCounter, FourgramCounter, GramCounter, SixgramCounter, TrigramCounter,
};
pub use record::CountRecord;
pub use trie::{TokenTrie, TokenTrieLm, WordTrie, WordTrieLm};
pub use trie_lm::{TrieLm, TrieLmBuilder, TrieLmLookuper};

pub use loader::{GramSource, GramsFileFormats, GramsTextLoader};
pub use parser::GramsParser;

pub use ef_trie_array::EliasFanoTrieArray;
pub use vocabulary::{DoubleArrayVocabulary, IdentityVocabulary, Vocabulary};

/// Python bindings
use pyo3::prelude::*;

#[pymodule]
fn tokengrams(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BigramCounter>()?;
    m.add_class::<TrigramCounter>()?;
    m.add_class::<FourgramCounter>()?;
    m.add_class::<FivegramCounter>()?;
    m.add_class::<SixgramCounter>()?;
    m.add_class::<TokenTrie>()?;
    m.add_class::<WordTrie>()?;
    Ok(())
}
