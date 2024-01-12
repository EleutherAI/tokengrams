use crate::trie_array::TrieArray;

/// Simple implementation of [`TrieArray`] with `Vec<usize>`.
#[derive(Default, Debug)]
pub struct SimpleTrieArray {
    token_ids: Vec<usize>,
    pointers: Vec<usize>,
}

impl TrieArray for SimpleTrieArray {
    fn build(token_ids: Vec<usize>, pointers: Vec<usize>) -> Self {
        Self {
            token_ids,
            pointers,
        }
    }

    /// Gets the token id with a given index.
    fn token_id(&self, i: usize) -> Option<usize> {
        self.token_ids.get(i).copied()
    }

    fn range(&self, pos: usize) -> Option<(usize, usize)> {
        Some((
            self.pointers.get(pos).copied()?,
            self.pointers.get(pos + 1).copied()?,
        ))
    }

    fn find_token(&self, pos: usize, id: usize) -> Option<usize> {
        let (b, e) = self.range(pos)?;
        self.token_ids[b..e]
            .iter()
            .position(|&x| x == id)
            .map(|i| i + b)
    }

    fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }

    fn num_pointers(&self) -> usize {
        self.pointers.len()
    }
}
