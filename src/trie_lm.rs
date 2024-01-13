mod builder;
mod lookuper;

use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sucds::int_vectors::{CompactVector, PrefixSummedEliasFano};

use crate::ef_trie_array::EliasFanoTrieArray;
use crate::loader::{GramsFileLoader, GramsGzFileLoader, GramsTextLoader};
use crate::sucds_glue;
use crate::vocabulary::Vocabulary;
use crate::GramsFileFormats;
use crate::WordGram;

pub use crate::trie_lm::builder::TrieLmBuilder;
pub use crate::trie_lm::lookuper::TrieLmLookuper;

/// Elias-Fano trie for indexing *N*-grams with their frequency counts.
#[derive(Deserialize, Default, Debug, Serialize)]
pub struct TrieLm<V> {
    vocab: V,
    arrays: Vec<EliasFanoTrieArray>,

    #[serde(with = "sucds_glue")]
    count_ranks: Vec<PrefixSummedEliasFano>,

    #[serde(with = "sucds_glue")]
    counts: Vec<CompactVector>,
}

impl<V> TrieLm<V>
where
    V: Vocabulary<GramType = WordGram>,
{
    /// Builds the index from *N*-gram counts files.
    ///
    /// # Arguments
    ///
    ///  - `filepaths`: Paths of *N*-gram counts files that should be sorted by *N* = 1, 2, ...
    ///  - `fmt`: File format.
    pub fn from_files<P>(filepaths: &[P], fmt: GramsFileFormats) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        match fmt {
            GramsFileFormats::Plain => Self::from_plain_files(filepaths),
            GramsFileFormats::Gzip => Self::from_gz_files(filepaths),
        }
    }

    /// Builds the index from *N*-gram counts files in a plain text format.
    ///
    /// # Arguments
    ///
    ///  - `filepaths`: Paths of *N*-gram counts files that should be sorted by *N* = 1, 2, ...
    pub fn from_plain_files<P>(filepaths: &[P]) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let mut loaders = Vec::with_capacity(filepaths.len());
        for filepath in filepaths {
            let loader = GramsFileLoader::new(filepath);
            loaders.push(loader);
        }
        TrieLmBuilder::new(loaders)?.build()
    }

    /// Builds the index from *N*-gram counts files in a gzip compressed format.
    ///
    /// # Arguments
    ///
    ///  - `filepaths`: Paths of *N*-gram counts files that should be sorted by *N* = 1, 2, ...
    pub fn from_gz_files<P>(filepaths: &[P]) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let mut loaders = Vec::with_capacity(filepaths.len());
        for filepath in filepaths {
            let loader = GramsGzFileLoader::new(filepath);
            loaders.push(loader);
        }
        TrieLmBuilder::new(loaders)?.build()
    }

    /// Builds the index from *N*-gram counts of raw texts (for debug).
    #[doc(hidden)]
    pub fn from_texts(texts: Vec<&'static str>) -> Result<Self> {
        let mut loaders = Vec::with_capacity(texts.len());
        for text in texts {
            let loader = GramsTextLoader::new(text.as_bytes());
            loaders.push(loader);
        }
        TrieLmBuilder::new(loaders)?.build()
    }
}

// Methods which work for TokenGram as well as WordGram.
impl<V> TrieLm<V>
where
    V: Vocabulary,
{
    /// Makes the lookuper.
    pub fn lookuper(&self) -> TrieLmLookuper<V> {
        TrieLmLookuper::new(self)
    }

    /// Gets the maximum of *N*.
    pub fn num_orders(&self) -> usize {
        self.count_ranks.len()
    }

    /// Gets the number of stored grams.
    pub fn num_grams(&self) -> usize {
        self.count_ranks.iter().fold(0, |acc, x| acc + x.len())
    }
}

#[cfg(test)]
mod tests {
    use sucds::int_vectors::Access;

    use super::*;
    use crate::{GramSource, WordGram, WordTrieLm};

    const GRAMS_1: &'static str = "4
A\t10
B\t7
C\t1
D\t1
";

    const GRAMS_2: &'static str = "9
A A\t5
A C\t2
B B\t2
B C\t2
B D\t1
C A\t3
C D\t2
D B\t1
D D\t1
";

    const GRAMS_3: &'static str = "7
A A C\t4
B B C\t2
B B D\t1
B C D\t1
D B B\t2
D B C\t1
D D D\t1
";

    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    const D: usize = 3;

    fn test_vocabulary<V: Vocabulary<GramType = WordGram>>(vocab: &V) {
        assert_eq!(vocab.get(WordGram::from_str("A")), Some(A));
        assert_eq!(vocab.get(WordGram::from_str("B")), Some(B));
        assert_eq!(vocab.get(WordGram::from_str("C")), Some(C));
        assert_eq!(vocab.get(WordGram::from_str("D")), Some(D));
    }

    fn test_unigrams(ra: &PrefixSummedEliasFano) {
        for (i, &count_rank) in [2, 1, 0, 0].iter().enumerate() {
            assert_eq!(ra.access(i), Some(count_rank));
        }
    }

    fn test_bigrams(ta: &EliasFanoTrieArray, ra: &PrefixSummedEliasFano) {
        for (i, &token_id) in [A, C, B, C, D, A, D, B, D].iter().enumerate() {
            assert_eq!(ta.token_id(i), Some(token_id));
        }
        for (i, &range) in [(0, 2), (2, 5), (5, 7), (7, 9)].iter().enumerate() {
            assert_eq!(ta.range(i), Some(range));
        }
        for (i, &count_rank) in [3, 0, 0, 0, 1, 2, 0, 1, 1].iter().enumerate() {
            assert_eq!(ra.access(i), Some(count_rank));
        }
    }

    fn test_trigrams(ta: &EliasFanoTrieArray, ra: &PrefixSummedEliasFano) {
        for (i, &token_id) in [C, C, D, D, B, C, D].iter().enumerate() {
            assert_eq!(ta.token_id(i), Some(token_id));
        }
        for (i, &range) in [
            (0, 1),
            (1, 1),
            (1, 3),
            (3, 4),
            (4, 4),
            (4, 4),
            (4, 4),
            (4, 6),
            (6, 7),
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(ta.range(i).unwrap(), range);
        }
        for (i, &count_rank) in [2, 1, 0, 0, 1, 0, 0].iter().enumerate() {
            assert_eq!(ra.access(i), Some(count_rank));
        }
    }

    #[test]
    fn test_ef_components() {
        let lm = WordTrieLm::from_texts(vec![GRAMS_1, GRAMS_2, GRAMS_3]).unwrap();
        test_vocabulary(&lm.vocab);
        test_unigrams(&lm.count_ranks[0]);
        test_bigrams(&lm.arrays[0], &lm.count_ranks[1]);
        test_trigrams(&lm.arrays[1], &lm.count_ranks[2]);
    }

    #[test]
    fn test_ef_lookup() {
        let lm = WordTrieLm::from_texts(vec![GRAMS_1, GRAMS_2, GRAMS_3]).unwrap();
        let mut lookuper = lm.lookuper();

        let loader = GramsTextLoader::new(GRAMS_1.as_bytes());
        let mut gp = loader.iter().unwrap();
        while let Some(rec) = gp.next_record() {
            let rec = rec.unwrap();
            assert_eq!(lookuper.with_gram(rec.gram), Some(rec.count));
        }

        let loader = GramsTextLoader::new(GRAMS_2.as_bytes());
        let mut gp = loader.iter().unwrap();
        while let Some(rec) = gp.next_record() {
            let rec = rec.unwrap();
            assert_eq!(lookuper.with_gram(rec.gram), Some(rec.count));
        }

        let loader = GramsTextLoader::new(GRAMS_3.as_bytes());
        let mut gp = loader.iter().unwrap();
        while let Some(rec) = gp.next_record() {
            let rec = rec.unwrap();
            assert_eq!(lookuper.with_gram(rec.gram), Some(rec.count));
        }

        assert_eq!(lookuper.with_gram(WordGram::from_str("E")), None);
        assert_eq!(lookuper.with_gram(WordGram::from_str("B A")), None);
        assert_eq!(lookuper.with_gram(WordGram::from_str("B B A")), None);
    }

    #[test]
    fn test_marginalization() {
        let lm = WordTrieLm::from_texts(vec![GRAMS_1, GRAMS_2, GRAMS_3]).unwrap();
        test_vocabulary(&lm.vocab);
        test_unigrams(&lm.count_ranks[0]);
    }
}
