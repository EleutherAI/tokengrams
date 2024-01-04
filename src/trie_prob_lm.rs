mod builder;

use anyhow::Result;

use crate::gram::WordGram;
use crate::loader::{GramsLoader, GramsTextLoader};
use crate::trie_array::TrieArray;
use crate::vocabulary::Vocabulary;

pub use builder::TrieProbLmBuilder;

pub const DEFAULT_UNK_PROB: f32 = -100.0;

/// Elias-Fano trie for indexing *N*-grams with their frequency counts.
#[derive(Default, Debug)]
#[allow(dead_code)]
pub struct TrieProbLm<T, V> {
    vocab: V,
    arrays: Vec<T>,
    probs: Vec<Vec<f32>>,    // TODO: Quantize
    backoffs: Vec<Vec<f32>>, // TODO: Quantize
}

impl<'a, T, V> TrieProbLm<T, V>
where
    T: TrieArray,
    V: Vocabulary<GramType = WordGram>,
{
    /// Builds the index from *N*-gram models of raw texts (for debug).
    #[doc(hidden)]
    pub fn from_texts(texts: Vec<&'static str>) -> Result<Self> {
        let mut loaders = Vec::with_capacity(texts.len());
        for text in texts {
            let loader: Box<dyn GramsLoader<_>> = Box::new(GramsTextLoader::new(text.as_bytes()));
            loaders.push(loader);
        }
        TrieProbLmBuilder::new(loaders)?.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimpleTrieProbLm;

    use float_cmp::ApproxEq;

    const GRAMS_1: &'static str = "4
A\t-1.83\t-0.74
B\t-2.01\t-0.69
C\t-2.22\t-0.55
D\t-1.91\t-0.62
";

    const GRAMS_2: &'static str = "9
A A\t-1.43\t-0.33
C A\t-1.39\t-0.49
B B\t-1.23\t-0.41
C B\t-1.11\t-0.29
D B\t-0.96\t-0.20
A C\t-1.02\t-0.34
D C\t-0.81\t-0.37
B D\t-0.45\t-0.41
D D\t-0.60\t-0.22
";

    const GRAMS_3: &'static str = "7
C A A\t1.01
C B B\t0.81
D B B\t0.91
D C B\t0.71
B B D\t0.52
C B D\t0.45
D D D\t0.34
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

    fn test_unigrams(probs: &[f32], backoffs: &[f32]) {
        for (i, &(p, b)) in [
            (-1.83, -0.74),
            (-2.01, -0.69),
            (-2.22, -0.55),
            (-1.91, -0.62),
        ]
        .iter()
        .enumerate()
        {
            probs[i].approx_eq(p, (0.0, 2));
            backoffs[i].approx_eq(b, (0.0, 2));
        }
    }

    fn test_bigrams<T: TrieArray>(ta: &T, probs: &[f32], backoffs: &[f32]) {
        for (i, &token_id) in [A, C, B, C, D, A, D, B, D].iter().enumerate() {
            assert_eq!(ta.token_id(i), token_id);
        }
        for (i, &range) in [(0, 2), (2, 5), (5, 7), (7, 9)].iter().enumerate() {
            assert_eq!(ta.range(i), range);
        }
        for (i, &(p, b)) in [
            (-1.43, -0.33),
            (-1.39, -0.49),
            (-1.23, -0.41),
            (-1.11, -0.29),
            (-0.96, -0.20),
            (-1.02, -0.34),
            (-0.81, -0.37),
            (-0.45, -0.41),
            (-0.60, -0.22),
        ]
        .iter()
        .enumerate()
        {
            probs[i].approx_eq(p, (0.0, 2));
            backoffs[i].approx_eq(b, (0.0, 2));
        }
    }

    fn test_trigrams<T: TrieArray>(ta: &T, probs: &[f32]) {
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
        for (i, &p) in [1.01, 0.81, 0.91, 0.71, 0.52, 0.45, 0.34]
            .iter()
            .enumerate()
        {
            probs[i].approx_eq(p, (0.0, 2));
        }
    }

    #[test]
    fn test_simple_components() {
        let lm = SimpleTrieProbLm::from_texts(vec![GRAMS_1, GRAMS_2, GRAMS_3]).unwrap();
        test_vocabulary(&lm.vocab);
        test_unigrams(&lm.probs[0], &lm.backoffs[0]);
        test_bigrams(&lm.arrays[0], &lm.probs[1], &lm.backoffs[1]);
        test_trigrams(&lm.arrays[1], &lm.probs[2]);
    }
}
