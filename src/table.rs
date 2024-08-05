extern crate utf16_literal;

use crate::par_quicksort::par_sort_unstable_by_key;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::{fmt, ops::Deref, u64};
use funty::Unsigned;
use std::fmt::Debug;

pub trait Token: Unsigned + Copy + Sync + Debug + TryInto<usize> + TryFrom<usize> + 'static {}
impl Token for u16 {}
impl Token for u32 {}

#[typetag::serde(tag = "type")]
pub trait Table: Send {
    /// Checks if the suffix table is lexicographically sorted. This is always true for valid suffix tables.
    fn is_sorted(&self) -> bool;

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
    fn contains(&self, query: &[usize]) -> bool;

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
    fn positions(&self, query: &[usize]) -> &[u64];

    // Count occurrences of each token directly following the query sequence.
    fn count_next(&self, query: &[usize], vocab: Option<usize>) -> Vec<usize>;

    fn batch_count_next(&self, queries: &[Vec<usize>], vocab: Option<usize>) -> Vec<Vec<usize>>;

    // For a given n, produce a map from an occurrence count to the number of unique n-grams with that occurrence count.
    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize>;
}

macro_rules! impl_table_for_suffix_table {
    ($t:ty) => {
        #[typetag::serde]
        impl Table for SuffixTable<Box<[$t]>, Box<[u64]>> {
            fn is_sorted(&self) -> bool {
                self.table
                    .par_windows(2)
                    .all(|pair| self.text[pair[0] as usize..] <= self.text[pair[1] as usize..])
            }

            fn contains(&self, query: &[usize]) -> bool {
                let query = query.iter().map(|&x| x as $t).collect::<Vec<$t>>();
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

            fn positions(&self, query: &[usize]) -> &[u64] {
                let query = query.iter().map(|&x| x as $t).collect::<Vec<$t>>();
                if self.text.is_empty()
                    || query.is_empty()
                    || (query.as_slice() < self.suffix(0) && !self.suffix(0).starts_with(&query))
                    || query.as_slice() > self.suffix(self.len() - 1)
                {
                    return &[];
                }

                let start = binary_search(&self.table, |&sufi| query.as_slice() <= &self.text[sufi as usize..]);
                let end = start
                    + binary_search(&self.table[start..], |&sufi| {
                        !self.text[sufi as usize..].starts_with(&query)
                    });

                if start > end {
                    &[]
                } else {
                    // This cast is not ideal, but necessary to match the trait signature
                    unsafe { std::mem::transmute(&self.table[start..end]) }
                }
            }

            fn count_next(&self, query: &[usize], vocab: Option<usize>) -> Vec<usize> {
                let query = query.iter().map(|&x| x as $t).collect::<Vec<$t>>();
                let vocab_size: usize = match vocab {
                    Some(size) => size,
                    None => <$t>::MAX as usize + 1,
                };
                let mut counts: Vec<usize> = vec![0; vocab_size];

                let (range_start, range_end) = self.boundaries(&query);
                self.recurse_count_next(&mut counts, &query, range_start, range_end);
                counts
            }

            // Count occurrences of each token directly following the query sequence.
            fn batch_count_next(&self, queries: &[Vec<usize>], vocab: Option<usize>) -> Vec<Vec<usize>> {
                // let queries = queries.iter().map(|&x| x as $t).collect::<Vec<$t>>();

                queries
                    .into_par_iter()
                    .map(|query| self.count_next(query.as_slice(), vocab))
                    .collect()
            }

            fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
                let mut count_map = HashMap::new();
                let (range_start, range_end) = self.boundaries(&[]);
                self.recurse_count_ngrams(range_start, range_end, 1, &[], n, &mut count_map);
                count_map
            }
        }
    };
}

// Use the macro to implement Table for u16 and u32 variants
impl_table_for_suffix_table!(u16);
impl_table_for_suffix_table!(u32);

// impl Table for SuffixTable<Box<[u32]>> {
//     fn count_next(&self, query: &[usize], vocab: Option<usize>) -> Vec<usize> {
//         let query = query.iter().map(|&x| x as u32).collect::<Vec<u32>>();
//         let vocab_size: usize = match vocab {
//             Some(size) => size,
//             None => u16::MAX as usize + 1,
//         };
//         let mut counts: Vec<usize> = vec![0; vocab_size];

//         let (range_start, range_end) = self.boundaries(&query);
//         self.recurse_count_next(&mut counts, &query, range_start, range_end);
//         counts
//     }
// }

// impl Table for SuffixTable<Box<[u16]>> {
//     fn count_next(&self, query: &[usize], vocab: Option<usize>) -> Vec<usize> {
//         let query = query.iter().map(|&x| x as u16).collect::<Vec<u16>>();
//         let vocab_size: usize = match vocab {
//             Some(size) => size,
//             None => u16::MAX as usize + 1,
//         };
//         let mut counts: Vec<usize> = vec![0; vocab_size];

//         let (range_start, range_end) = self.boundaries(&query);
//         self.recurse_count_next(&mut counts, &query, range_start, range_end);
//         counts
//     }
// }

/// A suffix table is a sequence of lexicographically sorted suffixes.
/// The table supports n-gram statistics computation and language modeling over text corpora.
#[derive(Clone, Serialize, Deserialize)]
pub struct SuffixTable<T = Box<[u16]>, U = Box<[u64]>> {
    text: T,
    table: U,
}

/// Methods for vanilla in-memory suffix tables
impl<T: Unsigned> SuffixTable<Box<[T]>, Box<[u64]>> {
    /// Creates a new suffix table for `text` in `O(n log n)` time and `O(n)`
    /// space.
    pub fn new<S>(src: S, verbose: bool) -> Self
    where
        S: Into<Box<[T]>>,
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

impl<T, U, E> SuffixTable<T, U>
where
    E: Token,
    T: Deref<Target = [E]> + Sync,
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

    /// Checks if the suffix table is lexicographically sorted. This is always true for valid suffix tables.
    pub fn is_sorted(&self) -> bool {
        self.table
            .par_windows(2)
            .all(|pair| self.text[pair[0] as usize..] <= self.text[pair[1] as usize..])
    }

    /// Returns the suffix at index `i`.
    #[inline]
    #[allow(dead_code)]
    pub fn suffix(&self, i: usize) -> &[E] {
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
    pub fn contains(&self, query: &[E]) -> bool {
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
    // #[allow(dead_code)]
    // pub fn positions(&self, query: &[E]) -> &[u64] {
    //     // We can quickly decide whether the query won't match at all if
    //     // it's outside the range of suffixes.
    //     if self.text.is_empty()
    //         || query.is_empty()
    //         || (query < self.suffix(0) && !self.suffix(0).starts_with(query))
    //         || query > self.suffix(self.len() - 1)
    //     {
    //         return &[];
    //     }

    //     // The below is pretty close to the algorithm on Wikipedia:
    //     //
    //     //     http://en.wikipedia.org/wiki/Suffix_array#Applications
    //     //
    //     // The key difference is that after we find the start index, we look
    //     // for the end by finding the first occurrence that doesn't start
    //     // with `query`. That becomes our upper bound.
    //     let start = binary_search(&self.table, |&sufi| query <= &self.text[sufi as usize..]);
    //     let end = start
    //         + binary_search(&self.table[start..], |&sufi| {
    //             !self.text[sufi as usize..].starts_with(query)
    //         });

    //     // Whoops. If start is somehow greater than end, then we've got
    //     // nothing.
    //     if start > end {
    //         &[]
    //     } else {
    //         &self.table[start..end]
    //     }
    // }

    /// Determine start and end `table` indices of items that start with `query`.
    fn boundaries(&self, query: &[E]) -> (usize, usize) {
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
        query: &[E],
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
    // pub fn count_next(&self, query: &[E], vocab: Option<usize>) -> Vec<usize> {
    //     // TODO u32 max
    //     // maybe better to provide max vocab then truncate and log if max vocab is greater than u32::max
    //     // could do this in python class and then we can take an unchecked usize everywhere
    //     let vocab_size: usize = match vocab {
    //         Some(size) => size,
    //         None => u16::MAX as usize + 1,
    //     };
    //     let mut counts: Vec<usize> = vec![0; vocab_size];

    //     let (range_start, range_end) = self.boundaries(query);
    //     self.recurse_count_next(&mut counts, query, range_start, range_end);
    //     counts
    // }

    // count_next helper method.
    fn recurse_count_next(
        &self,
        counts: &mut Vec<usize>,
        query: &[E],
        search_start: usize,
        search_end: usize,
    ) {
        if search_start >= search_end {
            return;
        }

        let mid = (search_start + search_end) / 2;
        let mut suffix = self.suffix(mid);
        // The search range may include the query itself, so we need to skip over it.
        if suffix == query {
            if mid + 1 == search_end {
                return;
            }
            suffix = self.suffix(mid + 1);
        }

        let (token_start, token_end) =
            self.range_boundaries(&suffix[..query.len() + 1], search_start, search_end);
        
        let index: usize = suffix[query.len()].try_into().unwrap_or_else(|_| panic!("Token > usize::MAX"));
        counts[index] = token_end - token_start;

        if search_start < token_start {
            self.recurse_count_next(counts, query, search_start, token_start);
        }
        if token_end < search_end {
            self.recurse_count_next(counts, query, token_end, search_end);
        }
    }

    // count_ngrams helper method.
    fn recurse_count_ngrams(
        &self,
        search_start: usize,
        search_end: usize,
        n: usize,
        query: &[E],
        target_n: usize,
        count_map: &mut HashMap<usize, usize>,
    ) {
        if search_start == search_end {
            return;
        }

        let mid = (search_start + search_end) / 2;
        let mut suffix = self.suffix(mid);
        // The search range may include the query itself, so we need to skip over it.
        if suffix == query {
            if mid + 1 == search_end {
                return;
            }
            suffix = self.suffix(mid + 1);
        }

        let (start, end) =
            self.range_boundaries(&suffix[..query.len() + 1], search_start, search_end);
        if n < target_n {
            self.recurse_count_ngrams(
                start,
                end,
                n + 1,
                &suffix[..query.len() + 1],
                target_n,
                count_map,
            );
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

    // For a given n, produce a map from an occurrence count to the number of unique n-grams with that occurrence count.
    // pub fn count_ngrams(&self, n: usize) -> HashMap<usize, usize> {
    //     let mut count_map = HashMap::new();
    //     let (range_start, range_end) = self.boundaries(&[]);
    //     self.recurse_count_ngrams(range_start, range_end, 1, &[], n, &mut count_map);
    //     count_map
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
