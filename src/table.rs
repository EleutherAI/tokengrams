extern crate utf16_literal;

use crate::par_quicksort::par_sort_unstable_by_key;
use anyhow::Result;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};
use std::{fmt, ops::Deref, ops::Div, ops::Mul, u64};

/// A suffix table is a sequence of lexicographically sorted suffixes.
/// The table supports n-gram statistics computation and language modeling over text corpora.
#[derive(Deserialize, Serialize, Clone)]
pub struct SuffixTable<T = Box<[u16]>, U = Box<[u64]>> {
    text: T,
    table: U,
    kn_cache: Option<Vec<f64>>,
}

impl<T: PartialEq, U: PartialEq> PartialEq for SuffixTable<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text && self.table == other.table
    }
}

impl<T: Eq, U: Eq> Eq for SuffixTable<T, U> {}

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
            kn_cache: None
        }
    }
}

impl<T, U> SuffixTable<T, U>
where
    T: Deref<Target = [u16]> + Sync,
    U: Deref<Target = [u64]> + Sync,
{
    pub fn from_parts(text: T, table: U) -> Self {
        SuffixTable { text, table, kn_cache: None }
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
            return &[]
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
            return (0, self.table.len())
        }
        if (query < self.suffix(0) && !self.suffix(0).starts_with(query))
            || query > self.suffix(self.len() - 1)
        {
            return (0, 0)
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
        let vocab_size: usize = match vocab {
            Some(size) => size.into(),
            None => u16::MAX as usize + 1,
        };
        let mut counts: Vec<usize> = vec![0; vocab_size];
        let mut query_vec = query.to_vec();

        let (range_start, range_end) = self.boundaries(query);
        let mut stack = vec![(range_start, range_end)];

        while let Some((search_start, search_end)) = stack.pop() {
            if search_start == search_end {
                continue;
            }

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
            let (start, end) = self.range_boundaries(&query_vec, search_start, search_end);
            counts[token as usize] = end - start;
            query_vec.pop();

            // Count other completions
            if search_start < start {
                stack.push((search_start, start));
            }
            if end < search_end {
                stack.push((end, search_end));
            }
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

    pub fn kneser_ney_probs(
        &self,
        query: &[u16],
        vocab: Option<u16>,
    ) -> Vec<f64> {
        // TODO Use leave-one-out cross-validation to estimate for each n
        let delta = 0.75;
        let kn_unigram_probs = self.get_cached_kn_unigram_probs();
        let p_continuations = if query.is_empty() {
            kn_unigram_probs.to_vec()
        } else {
            self.kneser_ney_probs(&query[1..], vocab)
        };
        let residual = (p_continuations.iter().sum::<f64>() - 1.0).abs();
        assert!(residual < 1e-3, "p_continuations: {:?}", p_continuations);
        
        let counts = self.count_next(query, vocab);
        let suffix_count = counts.iter().sum::<usize>() as f64;
        let unique_suffix_count = count_gt_zero(&counts) as f64;
        let uncommon_suffix_count = count_eq_one(&counts) as f64;

        if suffix_count == 0.0 {
            return p_continuations
        }

        // Interpolation budget to be distributed according to lower order n-gram distribution
        let lambda = if delta < 1.0 {
            delta.mul(unique_suffix_count).div(suffix_count)
        } else {
            uncommon_suffix_count + delta.mul(unique_suffix_count - uncommon_suffix_count).div(suffix_count)
        };
        
        counts
            .iter()
            .zip(p_continuations.iter())
            .map(|(&count, &p_continuation)| {
                (count as f64 - delta).max(0.0).div(suffix_count) + lambda.mul(p_continuation)
            })
            .collect()
    }

    /// Autoregressively sample k characters from a smoothed n-gram model
    pub fn kneser_ney_batch_sample(
        &self,
        query: &[u16],
        n: usize,
        k: usize,
        n_samples: usize,
        vocab: Option<u16>
    ) -> Result<Vec<Vec<u16>>> {
        (0..n_samples)
            .into_par_iter()
            .map(|_| self.kneser_ney_sample(query, n, k, vocab))
            .collect()
    }

    /// Autoregressively sample k characters from a smoothed n-gram model
    pub fn kneser_ney_sample(
        &self,
        query: &[u16],
        n: usize,
        k: usize,
        vocab: Option<u16>,
    ) -> Result<Vec<u16>> {
        let mut rng = thread_rng();
        let mut sequence = Vec::from(query);

        for _ in 0..k {
            let start = sequence.len().saturating_sub(n - 1);
            let prev = &sequence[start..];
            let probs = self.kneser_ney_probs(prev, vocab);

            let dist = WeightedIndex::new(&probs)?;
            let sampled_index = dist.sample(&mut rng);

            sequence.push(sampled_index as u16);
        }

        Ok(sequence)
    }

    /// Determine the kneser-ney unigram probabilities of each token, defined as the number of unique bigrams
    /// in text that end with a token divided by the number of unique bigrams.
    pub fn compute_kn_unigram_probs(&mut self, vocab: Option<u16>) {
        if let Some(_) = &self.kn_cache {
            return
        }

        let eps = 1e-9;
        let max_vocab = u16::MAX as usize + 1;
        let vocab = match vocab {
            Some(size) => size as usize,
            None => max_vocab,
        };

        let counts = if self.text.len() < 1000 {
            let mut counts = vec![0usize; max_vocab * max_vocab];
            self.text
                .windows(2)
                .map(|w| (u32::from(w[0]) << 16) | u32::from(w[1]))
                .for_each(|bigram| {
                    counts[bigram as usize] += 1;
                });
            counts

        } else {
            let counts: Vec<AtomicU32> = (0..max_vocab * max_vocab).map(|_| AtomicU32::new(0)).collect();
            self.text.par_windows(2).for_each(|w| {
                let bigram = (u32::from(w[0]) << 16) | u32::from(w[1]);
                counts[bigram as usize].fetch_add(1, Ordering::Relaxed);
            });
            counts.iter().map(|count| count.load(Ordering::Relaxed) as usize).collect()
        };

        let vocab_counts: Vec<usize> = (0..vocab).flat_map(|i| {
            let start = i * max_vocab;
            counts[start..start + vocab].iter().cloned()
        }).collect();
        let unique_bigram_count = vocab_counts.iter().filter(|&&count| count != 0).count() as f64;
        
        let unigram_probs: Vec<f64> = vocab_counts.chunks(vocab).map(|bigram_counts| {
            let suffix_count = bigram_counts.iter().filter(|&&count| count > 0).count() as f64;
            (suffix_count + eps) / (unique_bigram_count + eps * vocab as f64)
        }).collect();

        self.kn_cache = Some(unigram_probs);
    }

    fn get_cached_kn_unigram_probs(&self) -> &Vec<f64> {
        self.kn_cache.as_ref().unwrap()
    }
    /// Autoregressively sample k characters from a conditional distribution based
    /// on the previous (n - 1) characters (n-gram prefix) in the sequence.
    pub fn sample(&self, query: &[u16], n: usize, k: usize, vocab: Option<u16>) -> Result<Vec<u16>> {
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

    /// Autoregressively samples n_samples of k characters each from conditional distributions based
    /// on the previous (n - 1) characters (n-gram prefix) in the sequence."""
    pub fn batch_sample(
        &self,
        query: &[u16],
        n: usize,
        k: usize,
        n_samples: usize,
        vocab: Option<u16>
    ) -> Result<Vec<Vec<u16>>> {
        (0..n_samples)
            .into_par_iter()
            .map(|_| self.sample(query, n, k, vocab))
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

/// Count the number of elements in a slice that are greater than zero.
fn count_gt_zero(slice: &[usize]) -> usize {
    slice.iter().filter(|&&c| c > 0).count()
}

/// Count the number of elements in a slice that equal one.
fn count_eq_one(slice: &[usize]) -> usize {
    slice.iter().filter(|&&c| c == 1).count()
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
    fn kn_unigram_probs_exists() {
        let mut sa = sais("aaab");
        sa.compute_kn_unigram_probs(Some(100));
        let sa = sa;

        let kn_unigram_prob = sa.get_cached_kn_unigram_probs();
        let residual = (kn_unigram_prob.iter().sum::<f64>() - 1.0).abs();
        assert!(residual < 1e-4);

        // Additive smoothing results in non-zero probability on unused vocabulary
        assert!(kn_unigram_prob[0] > 0.0);
    }
}
