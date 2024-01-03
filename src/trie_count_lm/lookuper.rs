use crate::mappers::SortedArrayMapper;
use crate::rank_array::RankArray;
use crate::trie_array::TrieArray;
use crate::trie_count_lm::TrieCountLm;
use crate::vocabulary::Vocabulary;
use crate::Gram;

/// Lookuper for [`TrieCountLm`].
pub struct TrieCountLmLookuper<'a, T, V, A>
where
    T: TrieArray,
    V: Vocabulary,
    A: RankArray,
{
    trie: &'a TrieCountLm<T, V, A>,
    mapper: SortedArrayMapper,
}

impl<'a, T, V, A> TrieCountLmLookuper<'a, T, V, A>
where
    T: TrieArray,
    V: Vocabulary,
    A: RankArray,
{
    /// Creates [`TrieCountLmLookuper`] from [`TrieCountLm`].
    pub fn new(trie: &'a TrieCountLm<T, V, A>) -> TrieCountLmLookuper<'a, T, V, A> {
        TrieCountLmLookuper {
            trie,
            mapper: SortedArrayMapper::default(),
        }
    }

    /// Looks up a gram, returning the count.
    #[inline(always)]
    pub fn with_gram(&mut self, gram: Gram<u8>) -> Option<usize> {
        if self.mapper.from_gram(gram, &self.trie.vocab) {
            self.find()
        } else {
            None
        }
    }

    /// Looks up a gram in which tokens are sparated by a space, (e.g., `"the same time"`)
    /// returning the count.
    #[inline(always)]
    pub fn with_str(&mut self, gram: &str) -> Option<usize> {
        self.with_gram(Gram::from_str(gram))
    }

    /// Looks up a gram formed by a token list, (e.g., `&["the", "same", "time"]`)
    /// returning the count.
    #[inline(always)]
    pub fn with_tokens(&mut self, tokens: &[&str]) -> Option<usize> {
        if self.mapper.from_tokens(tokens, &self.trie.vocab) {
            self.find()
        } else {
            None
        }
    }

    #[inline(always)]
    fn find(&self) -> Option<usize> {
        let token_ids = self.mapper.get();
        let order = token_ids.len() - 1;
        let mut pos = token_ids[0];
        for (&token_id, array) in token_ids[1..].iter().zip(self.trie.arrays.iter()) {
            if let Some(next_pos) = array.find_token(pos, token_id) {
                pos = next_pos;
            } else {
                return None;
            }
        }
        let count_rank = self.trie.count_ranks[order].get(pos);
        Some(self.trie.counts[order].get(count_rank))
    }
}
