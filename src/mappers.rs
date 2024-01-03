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
    pub fn from_gram<V>(&mut self, gram: Gram<u8>, vocab: &V) -> bool
    where
        V: Vocabulary,
    {
        let tokens = gram.split_to_tokens();
        if MAX_ORDER < tokens.len() {
            return false;
        }
        for (i, &w) in tokens.iter().enumerate() {
            if let Some(mapped_id) = vocab.get(w) {
                self.mapped[i] = mapped_id;
            } else {
                return false;
            }
        }
        self.len = tokens.len();
        true
    }

    #[inline(always)]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_tokens<V>(&mut self, tokens: &[&str], vocab: &V) -> bool
    where
        V: Vocabulary,
    {
        if MAX_ORDER < tokens.len() {
            return false;
        }
        for (i, &w) in tokens.iter().enumerate() {
            if let Some(mapped_id) = vocab.get(Gram::from_str(w)) {
                self.mapped[i] = mapped_id;
            } else {
                return false;
            }
        }
        self.len = tokens.len();
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
    use crate::vocabulary::SimpleVocabulary;

    #[test]
    fn test_basic() {
        let grams = vec![
            Gram::from_str("A"),
            Gram::from_str("D"),
            Gram::from_str("B"),
        ];
        let vocab = SimpleVocabulary::build(&grams).unwrap();
        let mut mapper = SortedArrayMapper::default();

        assert_eq!(mapper.from_gram(Gram::from_str("A B D"), &vocab), true);
        assert_eq!(mapper.get(), &[0, 2, 1][..]);
        assert_eq!(mapper.from_gram(Gram::from_str("E B"), &vocab), false);

        assert_eq!(mapper.from_tokens(&["A", "B", "D"], &vocab), true);
        assert_eq!(mapper.get(), &[0, 2, 1][..]);
        assert_eq!(mapper.from_tokens(&["E", "B"], &vocab), false);
    }
}
