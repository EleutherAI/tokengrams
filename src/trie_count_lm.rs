mod builder;
mod lookuper;

use std::io::{Read, Write};
use std::path::Path;

use anyhow::Result;
use sucds::{util::IntIO, Searial};

use crate::loader::{GramsFileLoader, GramsGzFileLoader, GramsLoader, GramsTextLoader};
use crate::rank_array::RankArray;
use crate::trie_array::TrieArray;
use crate::vocabulary::Vocabulary;
use crate::GramsFileFormats;

pub use crate::trie_count_lm::builder::TrieCountLmBuilder;
pub use crate::trie_count_lm::lookuper::TrieCountLmLookuper;

/// Elias-Fano trie for indexing *N*-grams with their frequency counts.
#[derive(Default, Debug)]
pub struct TrieCountLm<T, V, A> {
    vocab: V,
    arrays: Vec<T>,
    count_ranks: Vec<A>,
    counts: Vec<sucds::CompactVector>,
}

impl<T, V, A> TrieCountLm<T, V, A>
where
    T: TrieArray,
    V: Vocabulary,
    A: RankArray,
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
            let loader: Box<dyn GramsLoader<_>> = Box::new(GramsFileLoader::new(filepath));
            loaders.push(loader);
        }
        TrieCountLmBuilder::new(loaders)?.build()
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
            let loader: Box<dyn GramsLoader<_>> = Box::new(GramsGzFileLoader::new(filepath));
            loaders.push(loader);
        }
        TrieCountLmBuilder::new(loaders)?.build()
    }

    /// Builds the index from *N*-gram counts of raw texts (for debug).
    #[doc(hidden)]
    pub fn from_texts(texts: Vec<&'static str>) -> Result<Self> {
        let mut loaders = Vec::with_capacity(texts.len());
        for text in texts {
            let loader: Box<dyn GramsLoader<_>> = Box::new(GramsTextLoader::new(text.as_bytes()));
            loaders.push(loader);
        }
        TrieCountLmBuilder::new(loaders)?.build()
    }

    /// Serializes the index into the writer.
    pub fn serialize_into<W>(&self, mut writer: W) -> Result<usize>
    where
        W: Write,
    {
        let mut mem = 0;
        // vocab
        mem += self.vocab.serialize_into(&mut writer)?;
        // arrays
        mem += self.arrays.len().serialize_into(&mut writer)?;
        for array in &self.arrays {
            mem += array.serialize_into(&mut writer)?;
        }
        // count_ranks
        mem += self.count_ranks.len().serialize_into(&mut writer)?;
        for count_rank in &self.count_ranks {
            mem += count_rank.serialize_into(&mut writer)?;
        }
        // counts
        mem += self.counts.len().serialize_into(&mut writer)?;
        for count in &self.counts {
            mem += count.serialize_into(&mut writer)?;
        }
        Ok(mem)
    }

    /// Deserializes the index from the reader.
    pub fn deserialize_from<R>(mut reader: R) -> Result<Self>
    where
        R: Read,
    {
        let vocab = V::deserialize_from(&mut reader)?;
        let arrays = {
            let len = usize::deserialize_from(&mut reader)?;
            let mut arrays = Vec::with_capacity(len);
            for _ in 0..len {
                arrays.push(T::deserialize_from(&mut reader)?);
            }
            arrays
        };
        let count_ranks = {
            let len = usize::deserialize_from(&mut reader)?;
            let mut count_ranks = Vec::with_capacity(len);
            for _ in 0..len {
                count_ranks.push(A::deserialize_from(&mut reader)?);
            }
            count_ranks
        };
        let counts = {
            let len = usize::deserialize_from(&mut reader)?;
            let mut counts = Vec::with_capacity(len);
            for _ in 0..len {
                counts.push(sucds::CompactVector::deserialize_from(&mut reader)?);
            }
            counts
        };
        Ok(Self {
            vocab,
            arrays,
            count_ranks,
            counts,
        })
    }

    /// Gets the number of bytes to serialize the index.
    pub fn size_in_bytes(&self) -> usize {
        let mut mem = 0;
        // vocab
        mem += self.vocab.size_in_bytes();
        // arrays
        mem += usize::size_in_bytes();
        for array in &self.arrays {
            mem += array.size_in_bytes();
        }
        // count_ranks
        mem += usize::size_in_bytes();
        for count_rank in &self.count_ranks {
            mem += count_rank.size_in_bytes();
        }
        // counts
        mem += usize::size_in_bytes();
        for count in &self.counts {
            mem += count.size_in_bytes();
        }
        mem
    }

    /// Gets breakdowns of memory usages for components.
    pub fn memory_statistics(&self) -> serde_json::Value {
        let vocab = self.vocab.memory_statistics();
        let arrays = {
            let mut arrays = vec![];
            for array in &self.arrays {
                arrays.push(array.memory_statistics());
            }
            arrays
        };
        let count_ranks = {
            let mut count_ranks = vec![];
            for count_rank in &self.count_ranks {
                count_ranks.push(count_rank.memory_statistics());
            }
            count_ranks
        };
        let counts = {
            let mut counts = vec![];
            for count in &self.counts {
                counts.push(serde_json::json!({"count": count.size_in_bytes()}));
            }
            counts
        };
        serde_json::json!({
            "vocab": vocab,
            "arrays": arrays,
            "count_ranks": count_ranks,
            "counts": counts,
        })
    }

    /// Makes the lookuper.
    pub fn lookuper(&self) -> TrieCountLmLookuper<T, V, A> {
        TrieCountLmLookuper::new(self)
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
    use super::*;
    use crate::{EliasFanoTrieCountLm, Gram, SimpleTrieCountLm};

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

    fn test_vocabulary<V: Vocabulary>(vocab: &V) {
        assert_eq!(vocab.get(Gram::from_str("A")), Some(A));
        assert_eq!(vocab.get(Gram::from_str("B")), Some(B));
        assert_eq!(vocab.get(Gram::from_str("C")), Some(C));
        assert_eq!(vocab.get(Gram::from_str("D")), Some(D));
    }

    fn test_unigrams<A: RankArray>(ra: &A) {
        for (i, &count_rank) in [2, 1, 0, 0].iter().enumerate() {
            assert_eq!(ra.get(i), count_rank);
        }
    }

    fn test_bigrams<T: TrieArray, A: RankArray>(ta: &T, ra: &A) {
        for (i, &token_id) in [A, C, B, C, D, A, D, B, D].iter().enumerate() {
            assert_eq!(ta.token_id(i), token_id);
        }
        for (i, &range) in [(0, 2), (2, 5), (5, 7), (7, 9)].iter().enumerate() {
            assert_eq!(ta.range(i), range);
        }
        for (i, &count_rank) in [3, 0, 0, 0, 1, 2, 0, 1, 1].iter().enumerate() {
            assert_eq!(ra.get(i), count_rank);
        }
    }

    fn test_trigrams<T: TrieArray, A: RankArray>(ta: &T, ra: &A) {
        for (i, &token_id) in [C, C, D, D, B, C, D].iter().enumerate() {
            assert_eq!(ta.token_id(i), token_id);
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
            assert_eq!(ta.range(i), range);
        }
        for (i, &count_rank) in [2, 1, 0, 0, 1, 0, 0].iter().enumerate() {
            assert_eq!(ra.get(i), count_rank);
        }
    }

    #[test]
    fn test_simple_components() {
        let lm = SimpleTrieCountLm::from_texts(vec![GRAMS_1, GRAMS_2, GRAMS_3]).unwrap();
        test_vocabulary(&lm.vocab);
        test_unigrams(&lm.count_ranks[0]);
        test_bigrams(&lm.arrays[0], &lm.count_ranks[1]);
        test_trigrams(&lm.arrays[1], &lm.count_ranks[2]);
    }

    #[test]
    fn test_ef_components() {
        let lm = EliasFanoTrieCountLm::from_texts(vec![GRAMS_1, GRAMS_2, GRAMS_3]).unwrap();
        test_vocabulary(&lm.vocab);
        test_unigrams(&lm.count_ranks[0]);
        test_bigrams(&lm.arrays[0], &lm.count_ranks[1]);
        test_trigrams(&lm.arrays[1], &lm.count_ranks[2]);
    }

    #[test]
    fn test_simple_lookup() {
        let lm = SimpleTrieCountLm::from_texts(vec![GRAMS_1, GRAMS_2, GRAMS_3]).unwrap();
        let mut lookuper = lm.lookuper();

        let loader = GramsTextLoader::new(GRAMS_1.as_bytes());
        let mut gp = loader.parser().unwrap();
        while let Some(rec) = gp.next_count_record() {
            let rec = rec.unwrap();
            assert_eq!(lookuper.with_gram(rec.gram()), Some(rec.count()));
        }

        let loader = GramsTextLoader::new(GRAMS_2.as_bytes());
        let mut gp = loader.parser().unwrap();
        while let Some(rec) = gp.next_count_record() {
            let rec = rec.unwrap();
            assert_eq!(lookuper.with_gram(rec.gram()), Some(rec.count()));
        }

        let loader = GramsTextLoader::new(GRAMS_3.as_bytes());
        let mut gp = loader.parser().unwrap();
        while let Some(rec) = gp.next_count_record() {
            let rec = rec.unwrap();
            assert_eq!(lookuper.with_gram(rec.gram()), Some(rec.count()));
        }

        assert_eq!(lookuper.with_gram(Gram::from_str("E")), None);
        assert_eq!(lookuper.with_gram(Gram::from_str("B A")), None);
        assert_eq!(lookuper.with_gram(Gram::from_str("B B A")), None);
    }

    #[test]
    fn test_ef_lookup() {
        let lm = EliasFanoTrieCountLm::from_texts(vec![GRAMS_1, GRAMS_2, GRAMS_3]).unwrap();
        let mut lookuper = lm.lookuper();

        let loader = GramsTextLoader::new(GRAMS_1.as_bytes());
        let mut gp = loader.parser().unwrap();
        while let Some(rec) = gp.next_count_record() {
            let rec = rec.unwrap();
            assert_eq!(lookuper.with_gram(rec.gram()), Some(rec.count()));
        }

        let loader = GramsTextLoader::new(GRAMS_2.as_bytes());
        let mut gp = loader.parser().unwrap();
        while let Some(rec) = gp.next_count_record() {
            let rec = rec.unwrap();
            assert_eq!(lookuper.with_gram(rec.gram()), Some(rec.count()));
        }

        let loader = GramsTextLoader::new(GRAMS_3.as_bytes());
        let mut gp = loader.parser().unwrap();
        while let Some(rec) = gp.next_count_record() {
            let rec = rec.unwrap();
            assert_eq!(lookuper.with_gram(rec.gram()), Some(rec.count()));
        }

        assert_eq!(lookuper.with_gram(Gram::from_str("E")), None);
        assert_eq!(lookuper.with_gram(Gram::from_str("B A")), None);
        assert_eq!(lookuper.with_gram(Gram::from_str("B B A")), None);
    }
}
