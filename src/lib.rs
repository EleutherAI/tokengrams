pub mod gram_counter;
pub mod gram;
pub mod loader;
pub mod parser;
pub mod record;
pub mod trie;
pub mod trie_lm;
pub mod util;
pub mod vocabulary;

mod mappers;
mod rank_array;
mod trie_array;

/// The maximum order of *N*-grams (i.e., `1 <= N <= 8`).
pub const MAX_ORDER: usize = 8;
/// The separator for tokens.
pub const TOKEN_SEPARATOR: u8 = b' ';
/// The separator for grams and count.
pub const GRAM_COUNT_SEPARATOR: u8 = b'\t';

//pub use typed_mmap::TypedMmap;

pub use gram_counter::{GramCounter, BigramCounter, TrigramCounter, FourgramCounter, FivegramCounter, SixgramCounter};
pub use gram::{Gram, WordGram};
pub use record::CountRecord;
pub use trie_lm::{TrieLm, TrieLmBuilder, TrieLmLookuper};
pub use trie::{TokenTrie, TokenTrieLm, WordTrie, WordTrieLm};

pub use loader::{GramsFileFormats, GramsLoader, GramsTextLoader};
pub use parser::GramsParser;

pub use rank_array::{EliasFanoRankArray, RankArray, SimpleRankArray};
pub use trie_array::{EliasFanoTrieArray, SimpleTrieArray, TrieArray};
pub use vocabulary::{DoubleArrayVocabulary, IdentityVocabulary, SimpleVocabulary, Vocabulary};

/// Simple implementation of [`TrieLm`].
/// Note that this is for debug, and do NOT use it for storing massive datasets.
pub type SimpleTrieLm = TrieLm<SimpleTrieArray, SimpleVocabulary, SimpleRankArray>;


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
