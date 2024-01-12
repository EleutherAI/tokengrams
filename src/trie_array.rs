mod ef;
mod simple;

pub use crate::trie_array::ef::EliasFanoTrieArray;
pub use crate::trie_array::simple::SimpleTrieArray;

/// Trait for a data structure for sorted arrays of each trie level.
pub trait TrieArray {
    /// Builds a [`TrieArray`] from sequences of token ids and pointers.
    fn build(token_ids: Vec<usize>, pointers: Vec<usize>) -> Self;

    /// Gets the `i`-th token id.
    fn token_id(&self, i: usize) -> Option<usize>;

    /// Gets the range `pointers[pos]..pointers[pos+1]`.
    fn range(&self, pos: usize) -> Option<(usize, usize)>;

    /// Finds the position `i` such that `token_id(i) = id` and `i in range(pos)`.
    fn find_token(&self, pos: usize, id: usize) -> Option<usize>;

    /// Gets the number of tokens stored.
    fn num_tokens(&self) -> usize;

    /// Gets the number of pointers stored.
    fn num_pointers(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_basic_1<T: TrieArray>() {
        let token_ids = vec![0, 2, 1, 2, 3, 0, 3, 1, 3];
        let pointers = vec![0, 2, 5, 7, 9];
        let ta = T::build(token_ids.clone(), pointers.clone());

        for (i, &x) in token_ids.iter().enumerate() {
            assert_eq!(ta.token_id(i).unwrap(), x);
        }
        for i in 0..pointers.len() - 1 {
            assert_eq!(ta.range(i).unwrap(), (pointers[i], pointers[i + 1]));
        }

        assert_eq!(ta.find_token(1, 3), Some(4));
        assert_eq!(ta.find_token(1, 1), Some(2));
        assert_eq!(ta.find_token(1, 4), None);

        assert_eq!(ta.num_tokens(), 9);
        assert_eq!(ta.num_pointers(), 5);
    }

    fn test_basic_2<T: TrieArray>() {
        let token_ids = vec![2, 2, 3, 3, 1, 2, 3];
        let pointers = vec![0, 1, 1, 3, 4, 4, 4, 4, 6, 7];
        let ta = T::build(token_ids.clone(), pointers.clone());

        for (i, &x) in token_ids.iter().enumerate() {
            assert_eq!(ta.token_id(i).unwrap(), x);
        }
        for i in 0..pointers.len() - 1 {
            assert_eq!(ta.range(i).unwrap(), (pointers[i], pointers[i + 1]));
        }

        assert_eq!(ta.find_token(2, 2), Some(1));
        assert_eq!(ta.find_token(2, 3), Some(2));
        assert_eq!(ta.find_token(2, 4), None);

        assert_eq!(ta.num_tokens(), 7);
        assert_eq!(ta.num_pointers(), 10);
    }

    #[test]
    fn test_basic_simple() {
        test_basic_1::<SimpleTrieArray>();
        test_basic_2::<SimpleTrieArray>();
    }

    #[test]
    fn test_basic_ef() {
        test_basic_1::<EliasFanoTrieArray>();
        test_basic_2::<EliasFanoTrieArray>();
    }
}
