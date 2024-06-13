extern crate utf16_literal;

use crate::par_quicksort::par_sort_unstable_by_key;
use anyhow::Result;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{u64, fmt, ops::Deref, ops::Div, ops::Mul};

/// A suffix table is a sequence of lexicographically sorted suffixes.
/// The table supports n-gram statistics computation and language modeling over text corpora.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
pub struct SuffixTable<T = Box<[u16]>, U = Box<[u64]>> {
    text: T,
    table: U,
}

/// Methods for vanilla in-memory suffix tables
impl SuffixTable<Box<[u16]>, Box<[u64]>> {
    /// Creates a new suffix table for `text` in `O(n log n)` time and `O(n)`
    /// space.
    pub fn new<S>(src: S, verbose: bool) -> Self
    where
        S: Into<Box<[u16]>>,
    {
        let text = src.into();

        // Implicitly store the suffixes using indices into the corpus,
        // and sort the suffixes in parallel. Unstable sorting ensures we
        // use no extra memory during this operation.
        //
        // Rayon's implementation falls back to a sequential algorithm for
        // sufficiently small inputs, so we don't need to worry about
        // parallelism overhead here.
        let mut table: Vec<_> = (0..text.len() as u64).collect();
        par_sort_unstable_by_key(&mut table[..], |&i| &text[i as usize..], verbose);

        SuffixTable {
            text,
            table: table.into(),
        }
    }
}

impl<T, U> SuffixTable<T, U>
where
    T: Deref<Target = [u16]> + Sync,
    U: Deref<Target = [u64]> + Sync,
{
    pub fn from_parts(text: T, table: U) -> Self {
        SuffixTable { text, table }
    }

    /// Consumes the suffix table and returns the underlying text and table.
    pub fn into_parts(self) -> (T, U) {
        (self.text, self.table)
    }

    /// Returns the number of suffixes in the table.
    ///
    /// Alternatively, this is the number of *bytes* in the text.
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns `true` iff `self.len() == 0`.
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the suffix at index `i`.
    #[inline]
    #[allow(dead_code)]
    pub fn suffix(&self, i: usize) -> &[u16] {
        &self.text[self.table[i] as usize..]
    }

    /// Returns true if and only if `query` is in text.
    ///
    /// This runs in `O(mlogn)` time, where `m == query.len()` and
    /// `n == self.len()`. (As far as this author knows, this is the best known
    /// bound for a plain suffix table.)
    ///
    /// You should prefer this over `positions` when you only need to test
    /// existence (because it is faster).
    ///
    /// # Example
    ///
    /// Build a suffix array of some text and test existence of a substring:
    ///
    /// ```rust
    /// use tokengrams::SuffixTable;
    /// use utf16_literal::utf16;
    ///
    /// let sa = SuffixTable::new(utf16!("The quick brown fox.").to_vec(), false);
    /// assert!(sa.contains(utf16!("quick")));
    /// ```
    #[allow(dead_code)]
    pub fn contains(&self, query: &[u16]) -> bool {
        !query.is_empty()
            && self
                .table
                .binary_search_by(|&sufi| {
                    self.text[sufi as usize..]
                        .iter()
                        .take(query.len())
                        .cmp(query.iter())
                })
                .is_ok()
    }

    /// Returns an unordered list of positions where `query` starts in `text`.
    ///
    /// This runs in `O(mlogn)` time, where `m == query.len()` and
    /// `n == self.len()`. (As far as this author knows, this is the best known
    /// bound for a plain suffix table.)
    ///
    /// Positions are byte indices into `text`.
    ///
    /// If you just need to test existence, then use `contains` since it is
    /// faster.
    ///
    /// # Example
    ///
    /// Build a suffix array of some text and find all occurrences of a
    /// substring:
    ///
    /// ```rust
    /// use tokengrams::SuffixTable;
    /// use utf16_literal::utf16;
    ///
    /// let sa = SuffixTable::new(utf16!("The quick brown fox was very quick.").to_vec(), false);
    /// assert_eq!(sa.positions(utf16!("quick")), &[4, 29]);
    /// ```
    #[allow(dead_code)]
    pub fn positions(&self, query: &[u16]) -> &[u64] {
        // We can quickly decide whether the query won't match at all if
        // it's outside the range of suffixes.
        if self.text.is_empty()
            || query.is_empty()
            || (query < self.suffix(0) && !self.suffix(0).starts_with(query))
            || query > self.suffix(self.len() - 1)
        {
            return &[];
        }

        // The below is pretty close to the algorithm on Wikipedia:
        //
        //     http://en.wikipedia.org/wiki/Suffix_array#Applications
        //
        // The key difference is that after we find the start index, we look
        // for the end by finding the first occurrence that doesn't start
        // with `query`. That becomes our upper bound.
        let start = binary_search(&self.table, |&sufi| query <= &self.text[sufi as usize..]);
        let end = start
            + binary_search(&self.table[start..], |&sufi| {
                !self.text[sufi as usize..].starts_with(query)
            });

        // Whoops. If start is somehow greater than end, then we've got
        // nothing.
        if start > end {
            &[]
        } else {
            &self.table[start..end]
        }
    }

    /// Returns the start and end `table` indices of suffixes starting with `query`
    fn boundaries(&self, query: &[u16]) -> (usize, usize) {
        if self.text.is_empty()
            || query.is_empty()
            || (query < self.suffix(0) && !self.suffix(0).starts_with(query))
            || query > self.suffix(self.len() - 1)
        {
            return (0, self.table.len());
        }
        // Should return (0, 0) if nothing in table starts with query
        if (query < self.suffix(0) && !self.suffix(0).starts_with(query)) || query > self.suffix(self.len() - 1) {
            return (0, 0);
        }

        let start = binary_search(&self.table, |&sufi| query <= &self.text[sufi as usize..]);
        let end = start
            + binary_search(&self.table[start..], |&sufi| {
                !self.text[sufi as usize..].starts_with(query)
            });

        (start, end)
    }

    /// Determine start and end indices of items that start with `query` in the `table` range.
    fn range_positions(&self, query: &[u16], range_start: usize, range_end: usize) -> (usize, usize) {
        if self.text.is_empty()
            || query.is_empty()
            || range_start.eq(&range_end)
            || (query < self.suffix(range_start) && !self.suffix(range_start).starts_with(query))
            || query > self.suffix(std::cmp::max(0, range_end - 1))
        {
            return (0, 0)
        }

        let start = binary_search(&self.table[range_start..range_end], |&sufi| {
            query <= &self.text[sufi as usize..]
        });
        let end = start
            + binary_search(&self.table[range_start + start..range_end], |&sufi| {
                !self.text[sufi as usize..].starts_with(query)
            });

        if start > end {
            (0, 0)
        } else {
            (range_start + start, range_start + end)
        }
    }

    /// Count how often each token succeeds `query`.
    pub fn count_next(&self, query: &[u16], vocab: Option<u16>) -> Vec<usize> {
        let vocab_size: usize = vocab.unwrap_or(u16::MAX) as usize + 1;
        let mut counts: Vec<usize> = vec![0usize; vocab_size];
        let mut query_vec = query.to_vec();

        let (range_start, range_end) = self.boundaries(query);
        let mut stack = vec![(range_start, range_end)];

        while let Some((search_start, search_end)) = stack.pop() {
            if search_start == search_end { continue; }
            
            // Find median index of suffixes starting with `query` and ending with at least one additional token
            let mut idx = search_start + (search_end - search_start) / 2;
            while self.suffix(idx).eq(query) {
                idx = idx + (search_end - idx) / 2 + 1;
            }
            if idx >= search_end {
                continue;
            }

            // Count query completion
            let token = self.suffix(idx)[query.len()];
            query_vec.push(token);
            let (start, end) = self.range_positions(&query_vec, search_start, search_end);
            counts[token as usize] = end - start;
            query_vec.pop();

            // Count other completions
            if search_start < start { stack.push((search_start, start)); }
            if end < search_end { stack.push((end, search_end)); }
        }
        counts
    }

    /// Count the occurrences of each token that directly follows each query sequence.
    pub fn batch_count_next(&self, queries: &[Vec<u16>], vocab: Option<u16>) -> Vec<Vec<usize>> {
        queries
            .into_par_iter()
            .map(|query| self.count_next(query, vocab))
            .collect()
    }

    pub fn kneser_ney_sample(&self, query: &[u16], n: usize, k: usize) -> Result<Vec<u16>> {
        let mut rng = thread_rng();
        // TODO Use cross validation to estimate these
        let delta = 1e-1;
        let lambda = 1e-1;
        let mut sequence = Vec::from(query);
        
        let unigram_counts = self.count_next(&[], None);

        // we have a token and we need the bigram probs for the next token
        // for one possible next token the probs are ~count(token, next_token) / count(token, all tokens)
        // count(token, all tokens) = self.boundaries(token)
        // so we need to do this for every item in the vocabulary - very expensive! but can pre-cache
        // as a first approximation can just use the unigram counts. then we need to filter out the instances 
        // that don't have a continuation

        for _ in 0..k {
            // look at the previous (n - 1) characters to predict the n-gram completion
            let start = sequence.len().saturating_sub(n - 1);
            let prev = &sequence[start..];
            let mut counts: Vec<usize> = self.count_next(prev, None);

            // Apply smoothing
            if prev.len() != 0 {
                // Update this to divide by the corresponding element in unigram_counts
                // Ingredients:
                // Number of times previous appears in the corpus after any word / number of distinct (prev, next) pairs in corpus
                // 
                // let smoothed_counts: Vec<f64> = counts.iter().enumerate().map(|(i, &count)| {
                //     let unigram_count = unigram_counts[i] as f64;
                //     return ((count as f64 - delta) / unigram_counts[i] as f64).max(0.0) + lambda * unigram_count;
                // }).collect();

                // Try using https://medium.com/@dennyc/a-simple-numerical-example-for-kneser-ney-smoothing-nlp-4600addf38b8
                // Both the wikipedia page and this article use frequency and count interchangeably. I'll use count for simplicity
                // TODO check that everything works out the same and delta is still correct in range
                let (prefix_start, prefix_end) = self.boundaries(prev);
                let prefix_count = (prefix_end - prefix_start) as f64;
                let mut prev_vec = prev.to_vec();
                let smoothed_counts: Vec<f64> = counts.iter().enumerate().map(|(i, &count)| {                    
                    let suffix_count = {
                        prev_vec.push(i as u16);
                        let (start, end) = self.boundaries(prev_vec);
                        prev_vec.pop();
                        (end - start) as f64
                    };
                    let first_term = suffix_count.div(prefix_count);
                    
                    let num_unique_suffixes = counts.iter().filter(|&&c| c > 0).count() as f64;
                    let lambda = delta.div(prefix_count).mul(num_unique_suffixes);
                    
                    let unigram_count = unigram_counts[i] as f64;
                    let num_unique_suffix_prefixes; // as a fn of (i, n, table), need to find every n-gram that ends with i 
                    // if there are no n-grams that end with i, we look for (n-1)-grams and so on until we find a match
                    // (convert to f using) the number of n-grams in the table, which is the table len - count(suffixes of len < n)
                    // Ensure this is the same as wikipedia version because it's going to be a real pain to implement
                    
                    let first_term = {
                        if n == 2 {
                            // first_term is different in the special bigram case
                            unimplemented!("Bigram case not implemented");
                        }
                        (count as f64 - delta).max(0.0).div(prefix_count)
                    };
                    let p_continuation = num_unique_suffix_prefixes.div(unigram_count);

                    return first_term + lambda.mul(p_continuation);
                }).collect();
            }
            
            let dist = WeightedIndex::new(&counts)?;
            let sampled_index = dist.sample(&mut rng);

            sequence.push(sampled_index as u16);
        }

        Ok(sequence)
    }

    /// Autoregressively sample k characters from a conditional distribution based
    /// on the previous (n - 1) characters (n-gram prefix) in the sequence.
    pub fn sample(&self, query: &[u16], n: usize, k: usize) -> Result<Vec<u16>> {
        let mut rng = thread_rng();
        let mut sequence = Vec::from(query);

        for _ in 0..k {
            // look at the previous (n - 1) characters to predict the n-gram completion
            let start = sequence.len().saturating_sub(n - 1);
            let prev = &sequence[start..];

            let counts: Vec<usize> = self.count_next(prev, None);
            let dist = WeightedIndex::new(&counts)?;
            let sampled_index = dist.sample(&mut rng);

            sequence.push(sampled_index as u16);
        }

        Ok(sequence)
    }

    /// Autoregressively samples num_samples of k characters each from conditional distributions based
    /// on the previous (n - 1) characters (n-gram prefix) in the sequence."""
    pub fn batch_sample(
        &self,
        query: &[u16],
        n: usize,
        k: usize,
        num_samples: usize,
    ) -> Result<Vec<Vec<u16>>> {
        (0..num_samples)
            .into_par_iter()
            .map(|_| self.sample(query, n, k))
            .collect()
    }

    /// Checks if the suffix table is lexicographically sorted. This is always true for valid suffix tables.
    pub fn is_sorted(&self) -> bool {
        self.table
            .windows(2)
            .all(|pair| self.text[pair[0] as usize..] <= self.text[pair[1] as usize..])
    }
}

impl fmt::Debug for SuffixTable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "\n-----------------------------------------")?;
        writeln!(f, "SUFFIX TABLE")?;
        for (rank, &sufstart) in self.table.iter().enumerate() {
            writeln!(f, "suffix[{}] {}", rank, sufstart,)?;
        }
        writeln!(f, "-----------------------------------------")
    }
}

/// Binary search to find first element such that `pred(T) == true`.
///
/// Assumes that if `pred(xs[i]) == true` then `pred(xs[i+1]) == true`.
///
/// If all elements yield `pred(T) == false`, then `xs.len()` is returned.
#[allow(dead_code)]
fn binary_search<T, F>(xs: &[T], mut pred: F) -> usize
where
    F: FnMut(&T) -> bool,
{
    let (mut left, mut right) = (0, xs.len());
    while left < right {
        let mid = (left + right) / 2;
        if pred(&xs[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use utf16_literal::utf16;

    fn sais(text: &str) -> SuffixTable {
        SuffixTable::new(text.encode_utf16().collect::<Vec<_>>(), false)
    }

    #[test]
    fn count_next_exists() {
        let sa = sais("aaab");

        let query = utf16!("a");
        let a_index = utf16!("a")[0] as usize;
        let b_index = utf16!("b")[0] as usize;

        assert_eq!(2, sa.count_next(query, Option::None)[a_index]);
        assert_eq!(1, sa.count_next(query, Option::None)[b_index]);
    }

    #[test]
    fn count_next_empty_query() {
        let sa = sais("aaab");

        let query = utf16!("");
        let a_index = utf16!("a")[0] as usize;
        let b_index = utf16!("b")[0] as usize;

        assert_eq!(3, sa.count_next(query, Option::None)[a_index]);
        assert_eq!(1, sa.count_next(query, Option::None)[b_index]);
    }

    #[test]
    fn batch_count_next_exists() {
        let sa = sais("aaab");

        let queries: Vec<Vec<u16>> = vec![vec![utf16!("a")[0]; 1]; 10_000];
        let a_index = utf16!("a")[0] as usize;
        let b_index = utf16!("b")[0] as usize;

        assert_eq!(2, sa.batch_count_next(&queries, Option::None)[0][a_index]);
        assert_eq!(1, sa.batch_count_next(&queries, Option::None)[0][b_index]);
    }
}
