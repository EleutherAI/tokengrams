extern crate utf16_literal;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fmt, ops::Deref, u64};
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use anyhow::Result;


/// A suffix table is a sequence of lexicographically sorted suffixes.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
pub struct SuffixTable<T = Box<[u16]>, U = Box<[u64]>> {
    text: T,
    table: U,
}

/// Methods for vanilla in-memory suffix tables
impl SuffixTable<Box<[u16]>, Box<[u64]>> {
    /// Creates a new suffix table for `text` in `O(n log n)` time and `O(n)`
    /// space.
    pub fn new<S>(src: S) -> Self
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
        table.par_sort_unstable_by_key(|&i| &text[i as usize..]);

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
    /// let sa = SuffixTable::new(utf16!("The quick brown fox.").to_vec());
    /// assert!(sa.contains(utf16!("quick")));
    /// ```
    #[allow(dead_code)]
    pub fn contains(&self, query: &[u16]) -> bool {
        query.len() > 0
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
    /// let sa = SuffixTable::new(utf16!("The quick brown fox was very quick.").to_vec());
    /// assert_eq!(sa.positions(utf16!("quick")), &[4, 29]);
    /// ```
    #[allow(dead_code)]
    pub fn positions(&self, query: &[u16]) -> &[u64] {
        // We can quickly decide whether the query won't match at all if
        // it's outside the range of suffixes.
        if self.text.len() == 0
            || query.len() == 0
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

    pub fn boundaries(&self, query: &[u16]) -> (usize, usize) { // a, aa
        if self.text.len() == 0
            || query.len() == 0
            || (query < self.suffix(0) && !self.suffix(0).starts_with(query))
            || query > self.suffix(self.len() - 1)
        {
            return (0, self.table.len());
        }

        let start = binary_search(&self.table, |&sufi| query <= &self.text[sufi as usize..]);
        let end = start
            + binary_search(&self.table[start..], |&sufi| {
                !self.text[sufi as usize..].starts_with(query)
            });

        (start, end)
    }

    fn range_positions(&self, query: &[u16], range_start: usize, range_end: usize) -> &[u64] {
        if self.text.len() == 0
            || query.len() == 0
            || (query < self.suffix(0 + range_start) && !self.suffix(0 + range_start).starts_with(query))
            || query > self.suffix(std::cmp::max(0, range_end - 1))
        {
            return &[];
        }

        let start = binary_search(&self.table[range_start..range_end], |&sufi| query <= &self.text[sufi as usize..]);
        let end = start
            + binary_search(&self.table[range_start + start..range_end], |&sufi| {
                !self.text[sufi as usize..].starts_with(query)
            });
    
        if start > end {
            &[]
        } else {
            &self.table[range_start + start..range_start + end]
        }
    }

    /// Returns an unordered list of counts of token values that succeed `query`.
    /// Counts all tokens if query is empty.
    fn bincount_next_tokens(&self, query: &[u16], vocab: Option<u16>) -> Vec<usize> {
        let mut counts: Vec<usize> = vec![0usize; vocab.unwrap_or(u16::MAX) as usize + 1];
        let mut suffixed_query = query.to_vec();
        let (range_start, range_end) = self.boundaries(query);

        for i in 0..counts.len() {
            suffixed_query.push(i as u16);
            
            let positions = self.range_positions(&suffixed_query, range_start, range_end);
            counts[i] = positions.len();
            suffixed_query.pop();
        }
        counts
    }

    /// Sample a character with probability proportional to its frequency succeeding the query.
    pub fn sample(&self, query: &[u16], n: usize, k: usize) -> Result<Vec<u16>> {
        let mut rng = thread_rng();
        let mut sequence = Vec::from(query);

        for _ in 0..k {
            // look at the previous (n - 1) characters to predict the n-gram completion
            let start = sequence.len().saturating_sub(n as usize - 1);
            let prev = &sequence[start..];
            
            let counts: Vec<usize> = self.bincount_next_tokens(prev, Option::None);
            let dist = WeightedIndex::new(&counts)?;
            let sampled_index = dist.sample(&mut rng);

            sequence.push(sampled_index as u16);
        }

        Ok(sequence)
    }

    pub fn batch_sample(&self, query: &[u16], n: usize, k: usize, num_samples: usize) -> Result<Vec<Vec<u16>>> {
        (0..num_samples).into_par_iter()
            .map(|_| {
                self.sample(query, n, k)
            })
            .collect()
    }

    // Checks if the suffix table is lexicographically sorted. This is always true for valid suffix tables.
    // pub fn is_sorted(&self) -> bool {
    //     self.table.windows(2).all(|pair| {
    //         self.suffix(pair[0] as usize) < self.suffix(pair[1] as usize)
    //     })
    // }

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
        SuffixTable::new(text.encode_utf16().collect::<Vec<_>>())
    }

    #[test]
    fn bincount_next_tokens_exists() {
        let sa = sais("aaab");
        
        let query = utf16!("a");
        let a_index = utf16!("a")[0] as usize;
        let b_index = utf16!("b")[0] as usize;

        assert_eq!(2, sa.bincount_next_tokens(query, Option::None)[a_index]);
        assert_eq!(1, sa.bincount_next_tokens(query, Option::None)[b_index]);
    }

    #[test]
    fn bincount_next_tokens_empty_query() {
        let sa = sais("aaab");
        
        let query = utf16!("");
        let a_index = utf16!("a")[0] as usize;
        let b_index = utf16!("b")[0] as usize;

        assert_eq!(3, sa.bincount_next_tokens(query, Option::None)[a_index]);
        assert_eq!(1, sa.bincount_next_tokens(query, Option::None)[b_index]);
    }
}