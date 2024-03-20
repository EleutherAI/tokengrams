/* Copyright 2021 Google LLC
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* This code is almost entirely based on suffix from BurntSushi. The original
 * program was licensed under the MIT license. We have modified it for
 * for two reasons:
 *
 * 1. The original implementation used u32 indices to point into the
 *    suffix array. This is smaller and fairly cache efficient, but here
 *    in the Real World we have to work with Big Data and our datasets
 *    are bigger than 2^32 bytes. So we have to work with u64 instead.
 *
 * 2. The original implementation had a utf8 interface. This is very
 *    convenient if you're working with strings, but we are working with
 *    byte arrays almost exclusively, and so just cut out the strings.
 *
 * When the comments below contradict these two statements, that's why.
 */
 extern crate utf16_literal;

 use rayon::prelude::*;
 use serde::{Deserialize, Serialize};
 use std::{fmt, u64};
 
 /// A suffix table is a sequence of lexicographically sorted suffixes.
 ///
 /// This is distinct from a suffix array in that it *only* contains
 /// suffix indices. It has no "enhanced" information like the inverse suffix
 /// table or least-common-prefix lengths (LCP array). This representation
 /// limits what you can do (and how fast), but it uses very little memory
 /// (4 bytes per character in the text).
 ///
 /// # Construction
 ///
 /// Suffix array construction is done in `O(n)` time and in `O(kn)` space,
 /// where `k` is the number of unique characters in the text. (More details
 /// below.) The specific algorithm implemented is from
 /// [(Nong et al., 2009)](https://local.ugene.unipro.ru/tracker/secure/attachment/12144/Linear%20Suffix%20Array%20Construction%20by%20Almost%20Pure%20Induced-Sorting.pdf),
 /// but I actually used the description found in
 /// [(Shrestha et al., 2014)](http://bib.oxfordjournals.org/content/15/2/138.full.pdf),
 /// because it is much more accessible to someone who is not used to reading
 /// algorithms papers.
 ///
 /// The main thrust of the algorithm is that of "reduce and conquer." Namely,
 /// it reduces the problem of finding lexicographically sorted suffixes to a
 /// smaller subproblem, and solves it recursively. The subproblem is to find
 /// the suffix array of a smaller string, where that string is composed by
 /// naming contiguous regions of the original text. If there are any duplicate
 /// names, then the algorithm proceeds recursively. If there are no duplicate
 /// names (base case), then the suffix array of the subproblem is already
 /// computed. In essence, this "inductively sorts" suffixes of the original
 /// text with several linear scans over the text. Because of the number of
 /// linear scans, the performance of construction is heavily tied to cache
 /// performance.
 ///
 /// The space usage is roughly `6` bytes per character. (The optimal bound is
 /// `5` bytes per character, although that may be for a small constant
 /// alphabet.) `4` bytes comes from the suffix array itself. The extra `2`
 /// bytes comes from storing the suffix type of each character (`1` byte) and
 /// information about bin boundaries, where the number of bins is equal to
 /// the number of unique characters in the text. This doesn't formally imply
 /// another byte of overhead, but in practice, the alphabet can get quite large
 /// when solving the subproblems mentioned above (even if the alphabet of the
 /// original text is very small).
 
 #[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
 pub struct SuffixTable {
     text: Box<[u16]>,
     table: Box<[u64]>,
 }
 
 impl SuffixTable {
     /// Creates a new suffix table for `text` in `O(n log n)` time and `O(n)`
     /// space.
     ///
     /// The table stores either `S` or a `&S` and a lexicographically sorted
     /// list of suffixes. Each suffix is represented by a 64 bit integer and
     /// is a **token index** into `text`.
     pub fn new<S>(text: S) -> SuffixTable
     where
         S: Into<Box<[u16]>>,
     {
        let text = text.into();

        // Simply construct a vector containing all suffix indices, and sort it
        // in parallel using the suffixes they point to as keys.
        let mut table: Vec<_> = (0..text.len() as u64).collect();
        table.sort_by_key(|&i| &text[i as usize..]);

        SuffixTable {
            text: text,
            table: table.into(),
        }
     }
 
     pub fn par_new<S>(src: S) -> SuffixTable
     where
         S: Into<Box<[u16]>>,
     {
         let text = src.into();

         // Simply construct a vector containing all suffix indices, and sort it
         // in parallel using the suffixes they point to as keys.
         let mut table: Vec<_> = (0..text.len() as u64).collect();
         table.par_sort_by_key(|&i| &text[i as usize..]);

         SuffixTable {
             text: text,
             table: table.into(),
         }
     }
 
     /// Creates a new suffix table from an existing list of lexicographically
     /// sorted suffix indices.
     ///
     /// Note that the invariant that `table` must be a suffix table of `text`
     /// is not checked! If it isn't, this will cause other operations on a
     /// suffix table to fail in weird ways.
     ///
     /// This fails if the number of characters in `text` does not equal the
     /// number of suffixes in `table`.
     #[allow(dead_code)]
     pub fn from_parts<S, T>(text: S, table: T) -> SuffixTable
     where
         S: Into<Box<[u16]>>,
         T: Into<Box<[u64]>>,
     {
         let (text, table) = (text.into(), table.into());
         assert_eq!(text.len(), table.len());
         SuffixTable {
             text: text,
             table: table,
         }
     }

     /// Merge several suffix tables into one.
     pub fn from_tables(tables: Vec<SuffixTable>) -> SuffixTable {
         let mut offset = 0;
         let num_tokens = tables.iter().map(|t| t.len()).sum();

         let mut text = Vec::with_capacity(num_tokens);
         let mut table = Vec::with_capacity(num_tokens);

         for t in tables {
             text.extend(t.text.into_iter());
             table.extend(t.table.into_iter().map(|i| i + offset));

             offset += t.len() as u64;
         }

         // TODO: Use a more efficient way to merge the sorted tables.
         table.sort_unstable_by_key(|&i| &text[i as usize..]);

         SuffixTable {
             text: text.into_boxed_slice(),
             table: table.into_boxed_slice(),
         }
     }
 
     /// Extract the parts of a suffix table.
     ///
     /// This is useful to avoid copying when the suffix table is part of an
     /// intermediate computation.
     pub fn into_parts(self) -> (Box<[u16]>, Box<[u64]>) {
         (self.text, self.table)
     }

     /// Return the suffix table.
     #[inline]
     pub fn table(&self) -> &[u64] {
         &self.table
     }
 
     /// Return the text.
     #[inline]
     pub fn text(&self) -> &[u16] {
         &self.text
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
 