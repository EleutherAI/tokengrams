use crate::vocabulary::Vocabulary;
use crate::Gram;

use crate::MAX_ORDER;

#[derive(Default)]
pub struct SortedArrayMapper {
    mapped: [usize; MAX_ORDER],
    len: usize,
}

impl SortedArrayMapper {
    #[inline(always)]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_gram<V: Vocabulary>(&mut self, gram: V::GramType, vocab: &V) -> bool {
        let unigrams = gram.to_unigrams();
        if MAX_ORDER < unigrams.len() {
            return false;
        }
        self.len = unigrams.len();

        for (i, w) in unigrams.into_iter().enumerate() {
            if let Some(mapped_id) = vocab.get(w) {
                self.mapped[i] = mapped_id;
            } else {
                return false;
            }
        }
        true
    }

    #[inline(always)]
    pub fn get(&self) -> &[usize] {
        &self.mapped[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocabulary::DoubleArrayVocabulary;
    use crate::WordGram;

    #[test]
    fn test_basic() {
        let grams = [
            WordGram::from_str("A"),
            WordGram::from_str("D"),
            WordGram::from_str("B"),
        ];
        let vocab = DoubleArrayVocabulary::build(grams).unwrap();
        let mut mapper = SortedArrayMapper::default();

        assert_eq!(mapper.from_gram(WordGram::from_str("A B D"), &vocab), true);
        assert_eq!(mapper.get(), &[0, 2, 1][..]);
        assert_eq!(mapper.from_gram(WordGram::from_str("E B"), &vocab), false);
    }
}
