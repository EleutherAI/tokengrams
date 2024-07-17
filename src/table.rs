extern crate utf16_literal;

use crate::par_quicksort::par_sort_unstable_by_key;
use anyhow::Result;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{fmt, ops::Deref, ops::Mul, u64};

/// A suffix table is a sequence of lexicographically sorted suffixes.
/// The table supports n-gram statistics computation and language modeling over text corpora.
#[derive(Clone, Deserialize, Serialize)]
pub struct SuffixTable<T = Box<[u16]>, U = Box<[u64]>> {
    text: T,
    table: U,
    cache: KneserNeyCache,
}

impl<T: PartialEq, U: PartialEq> PartialEq for SuffixTable<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text && self.table == other.table
    }
}

impl<T: Eq, U: Eq> Eq for SuffixTable<T, U> {}

#[derive(Clone, Deserialize, Serialize)]
struct KneserNeyCache {
    unigram_probs: Option<Vec<f64>>,
    n_delta: HashMap<usize, f64>,
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
            cache: KneserNeyCache {
                unigram_probs: None,
                n_delta: HashMap::new(),
            },
        }
    }
}

impl<T, U> SuffixTable<T, U>
where
    T: Deref<Target = [u16]> + Sync,
    U: Deref<Target = [u64]> + Sync,
{
    pub fn from_parts(text: T, table: U) -> Self {
        SuffixTable {
            text,
            table,
            cache: KneserNeyCache {
                unigram_probs: None,
                n_delta: HashMap::new(),
            },
        }
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

    /// Checks if the suffix table is lexicographically sorted. This is always true for valid suffix tables.
    pub fn is_sorted(&self) -> bool {
        self.table
            .par_windows(2)
            .all(|pair| self.text[pair[0] as usize..] <= self.text[pair[1] as usize..])
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

    /// Determine start and end `table` indices of items that start with `query`.
    fn boundaries(&self, query: &[u16]) -> (usize, usize) {
        if self.text.is_empty() || query.is_empty() {
            return (0, self.table.len());
        }
        if (query < self.suffix(0) && !self.suffix(0).starts_with(query))
            || query > self.suffix(self.len() - 1)
        {
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
    fn range_boundaries(
        &self,
        query: &[u16],
        range_start: usize,
        range_end: usize,
    ) -> (usize, usize) {
        if self.text.is_empty()
            || query.is_empty()
            || range_start.eq(&range_end)
            || (query < self.suffix(range_start) && !self.suffix(range_start).starts_with(query))
            || query > self.suffix(std::cmp::max(0, range_end - 1))
        {
            return (0, 0);
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

    // Count occurrences of each token directly following the query sequence.
    pub fn count_next(&self, query: &[u16], vocab: Option<u16>) -> Vec<usize> {
        let vocab_size: usize = match vocab {
            Some(size) => size as usize,
            None => u16::MAX as usize + 1,
        };
        let mut counts: Vec<usize> = vec![0; vocab_size];

        let (range_start, range_end) = self.boundaries(query);
        self.recurse_count_next(&mut counts, query, range_start, range_end);
        counts
    }

    // Count occurrences of each token directly following the query sequence.
    pub fn batch_count_next(&self, queries: &[Vec<u16>], vocab: Option<u16>) -> Vec<Vec<usize>> {
        queries
            .into_par_iter()
            .map(|query| self.count_next(query, vocab))
            .collect()
    }
       
    // count_next helper method.
    fn recurse_count_next(
        &self,
        counts: &mut Vec<usize>,
        query: &[u16],
        search_start: usize,
        search_end: usize,
    ) {
        if search_start >= search_end {
            return;
        }

        let mid = (search_start + search_end) / 2;
        let mut suffix = self.suffix(mid);
        if suffix == query {
            if mid + 1 == search_end {
                return;
            }
            suffix = self.suffix(mid + 1);
        }

        let (token_start, token_end) = self.range_boundaries(&suffix[..query.len() + 1], search_start, search_end);
        counts[suffix[query.len()] as usize] = token_end - token_start;

        if search_start < token_start {
            self.recurse_count_next(counts, query, search_start, token_start);
        }
        if token_end < search_end {
            self.recurse_count_next(counts, query, token_end, search_end);
        }
    }

    /// Autoregressively sample num_samples of k characters from an unsmoothed n-gram model."""
    pub fn sample_unsmoothed(
        &self,
        query: &[u16],
        n: usize,
        k: usize,
        num_samples: usize,
        vocab: Option<u16>,
    ) -> Result<Vec<Vec<u16>>> {
        (0..num_samples)
            .into_par_iter()
            .map(|_| self.sample(query, n, k, vocab))
            .collect()
    }

    //// Autoregressively sample a sequence of k characters from an unsmoothed n-gram model."""
    fn sample(&self, query: &[u16], n: usize, k: usize, vocab: Option<u16>) -> Result<Vec<u16>> {
        let mut rng = thread_rng();
        let mut sequence = Vec::from(query);

        for _ in 0..k {
            // look at the previous (n - 1) characters to predict the n-gram completion
            let start = sequence.len().saturating_sub(n - 1);
            let prev = &sequence[start..];

            let counts = self.count_next(prev, vocab);
            let dist = WeightedIndex::new(&counts)?;
            let sampled_index = dist.sample(&mut rng);

            sequence.push(sampled_index as u16);
        }

        Ok(sequence)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    pub fn get_smoothed_probs(&mut self, query: &[u16], vocab: Option<u16>) -> Vec<f64> {
        self.estimate_deltas(1);
        self.compute_smoothed_unigram_probs(vocab);
        self.smoothed_probs(query, vocab)
    }

    /// Returns interpolated Kneser-Ney smoothed token probability distribution using all previous
    /// tokens in the query.
    pub fn batch_get_smoothed_probs(&mut self, queries: &[Vec<u16>], vocab: Option<u16>) -> Vec<Vec<f64>> {
        self.estimate_deltas(1);
        self.compute_smoothed_unigram_probs(vocab);

        queries
            .into_par_iter()
            .map(|query| self.smoothed_probs(query, vocab))
            .collect()
    }

    /// Autoregressively sample num_samples of k characters from a Kneser-Ney smoothed n-gram model.
    pub fn sample_smoothed(
        &mut self,
        query: &[u16],
        n: usize,
        k: usize,
        num_samples: usize,
        vocab: Option<u16>,
    ) -> Result<Vec<Vec<u16>>> {
        self.estimate_deltas(1);
        self.compute_smoothed_unigram_probs(vocab);

        (0..num_samples)
            .into_par_iter()
            .map(|_| self.kn_sample(query, n, k, vocab))
            .collect()
    }
        
    /// Returns the Kneser-Ney smoothed token probability distribution for a query 
    /// continuation using absolute discounting as described in
    /// "On structuring probabilistic dependences in stochastic language modelling", page 25,
    /// doi:10.1006/csla.1994.1001
    fn smoothed_probs(&self, query: &[u16], vocab: Option<u16>) -> Vec<f64> {
        let p_continuations = if query.is_empty() {
            self.get_cached_smoothed_unigram_probs().to_vec()
        } else {
            self.smoothed_probs(&query[1..], vocab)
        };
        
        let counts = self.count_next(query, vocab);
        let suffix_count_recip = {
            let suffix_count: usize = counts.iter().sum();
            if suffix_count == 0 {
                return p_continuations;
            }
            1.0 / suffix_count as f64
        };

        let (gt_zero_count, eq_one_count) = self.get_occurrence_counts(&counts);
        let used_suffix_count = gt_zero_count as f64;
        let used_once_suffix_count = eq_one_count as f64;

        // Interpolation budget to be distributed according to lower order n-gram distribution
        let delta = self.get_cached_delta(query.len() + 1);
        let lambda = if delta < 1.0 {
            delta.mul(used_suffix_count).mul(suffix_count_recip)
        } else {
            used_once_suffix_count
                + delta
                    .mul(used_suffix_count - used_once_suffix_count)
                    .mul(suffix_count_recip)
        };
        
        let mut probs = Vec::with_capacity(counts.len());
        counts
            .iter()
            .zip(p_continuations.iter())
            .for_each(|(&count, &p_continuation)| {
                let prob =
                    (count as f64 - delta).max(0.0).mul(suffix_count_recip) + lambda.mul(p_continuation);
                probs.push(prob);
            });
        probs
    }

    /// Returns tuple of the number of elements that are greater than 0 and the number of elements that equal 1.
    fn get_occurrence_counts(&self, slice: &[usize]) -> (usize, usize) {
        slice
            .iter()
            .fold((0, 0), |(gt_zero_count, eq_one_count), &c| {
                let gt_zero_count = gt_zero_count + (c > 0) as usize;
                let eq_one_count = eq_one_count + (c == 1) as usize;
                (gt_zero_count, eq_one_count)
            })
    }

    /// Autoregressively sample k characters from a Kneser-Ney smoothed n-gram model.
    fn kn_sample(&self, query: &[u16], n: usize, k: usize, vocab: Option<u16>) -> Result<Vec<u16>> {
        let mut rng = thread_rng();
        let mut sequence = Vec::from(query);

        for _ in 0..k {
            let start = sequence.len().saturating_sub(n - 1);
            let prev = &sequence[start..];
            let probs = self.smoothed_probs(prev, vocab);
            let dist = WeightedIndex::new(&probs)?;
            let sampled_index = dist.sample(&mut rng);

            sequence.push(sampled_index as u16);
        }

        Ok(sequence)
    }

    // For a given n, produce a map from an occurrence count in (1, 2) to the number of unique n-grams with that occurrence count.
    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
        let mut count_map = HashMap::from([(1, 0), (2, 0)]);
        let query = Vec::new();
        let (range_start, range_end) = self.boundaries(&query);
        self.recurse_count_ngrams(range_start, range_end, 1, &query, n, &mut count_map);
        count_map
    }

    // count_ngrams helper method.
    fn recurse_count_ngrams(
        &self,
        search_start: usize,
        search_end: usize,
        n: usize,
        query: &[u16],
        target_n: usize,
        count_map: &mut HashMap<usize, usize>,
    ) {
        if search_start == search_end {
            return;
        }

        // Find median of indices ending with at least one additional token
        let mid = (search_start + search_end) / 2;
        let mut suffix = self.suffix(mid);
        if suffix == query {
            if mid + 1 == search_end {
                return;
            }
            suffix = self.suffix(mid + 1);
        }

        let (start, end) = self.range_boundaries(&suffix[..query.len() + 1], search_start, search_end);
        if n < target_n {
            self.recurse_count_ngrams(start, end, n + 1, &suffix[..query.len() + 1], target_n, count_map);
        } else {
            *count_map.entry(end - start).or_insert(0) += 1;
        }

        if search_start < start {
            self.recurse_count_ngrams(search_start, start, n, query, target_n, count_map);
        }
        if end < search_end {
            self.recurse_count_ngrams(end, search_end, n, query, target_n, count_map);
        }
    }
    
    /// Warning: O(k**n) where k is vocabulary size, use with caution.
    /// Improve smoothed model quality by replacing the default delta hyperparameters
    /// for models of order n and below with improved estimates over the entire index.
    /// https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf, page 16."""
    pub fn estimate_deltas(&mut self, n: usize) {
        for i in 1..n + 1 {
            if self.cache.n_delta.contains_key(&i) {
                continue;
            }
            
            let count_map = self.count_ngrams(i);
            let n1 = *count_map.get(&1).unwrap_or(&0) as f64;
            let n2 = *count_map.get(&2).unwrap_or(&0) as f64;

            // n1 and n2 are greater than 0 for non-trivial datasets
            let delta = if n1 == 0. || n2 == 0. {
                1.
            } else {
                n1 / (n1 + n2.mul(2.))
            };

            self.cache.n_delta.insert(i, delta);
        }
    }

    fn get_cached_delta(&self, n: usize) -> f64 {
        *self.cache.n_delta.get(&n).unwrap_or(&0.5)
    }

    /// Returns unigram probabilities with additive smoothing applied.
    fn compute_smoothed_unigram_probs(&mut self, vocab: Option<u16>) {
        if let Some(_) = &self.cache.unigram_probs {
            return;
        }

        let eps = 1e-9;
        let max_vocab = u16::MAX as usize + 1;
        let vocab_size = match vocab {
            Some(size) => size as usize,
            None => max_vocab,
        };

        // Count the number of unique bigrams that end with each token
        let counts = self.count_next(&[], vocab);
        let total_count: usize = counts.iter().sum();
        let adjusted_total_count = total_count as f64 + eps.mul(vocab_size as f64);
        let unigram_probs: Vec<f64> = counts
            .iter()
            .map(|&count| {
                (count as f64 + eps) / adjusted_total_count
            })
            .collect();

        self.cache.unigram_probs = Some(unigram_probs);
    }

    fn get_cached_smoothed_unigram_probs(&self) -> &Vec<f64> {
        self.cache.unigram_probs.as_ref().unwrap()
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

    #[test]
    fn unigram_probs_exists() {
        let mut sa = sais("aaab");
        sa.compute_smoothed_unigram_probs(Some(100));
        let sa = sa;

        let unigram_probs = sa.get_cached_smoothed_unigram_probs();
        let residual = (unigram_probs.iter().sum::<f64>() - 1.0).abs();
        assert!(residual < 1e-4);

        // Additive smoothing results in non-zero probability on unused vocabulary
        assert!(unigram_probs[0] > 0.0);
    }

    #[test]
    fn compute_ngram_counts_exists() {
        let sa = sais("aaabbbaaa");
        let count_map = sa.count_ngrams(3);

        // Every 3-gram except aaa occurs once
        let mut expected_map = std::collections::HashMap::new();
        expected_map.insert(1, 5);
        expected_map.insert(2, 1);

        assert_eq!(count_map, expected_map);
    }

    #[test]
    fn sample_query_exists() {
        let sa = sais("aaa");
        let a = utf16!("a");
        let tokens = sa.sample(a, 3, 10, None).unwrap();

        assert_eq!(*tokens.last().unwrap(), a[0]);
    }

    #[test]
    fn sample_empty_query_exists() {
        let sa = sais("aaa");
        let a = utf16!("a");
        let empty_query = utf16!("");
        let tokens = sa.sample(empty_query, 3, 10, None).unwrap();

        assert_eq!(*tokens.last().unwrap(), a[0]);
    }
}
