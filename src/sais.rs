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
use std::u64;

use self::SuffixType::{Ascending, Descending, Valley};


pub fn sais_table(text: &[u16]) -> Vec<u64> {
    let mut sa = vec![0u64; text.len()];
    let mut stypes = SuffixTypes::new(text.len() as u64);
    let mut bins = Bins::new();
    sais(&mut sa, &mut stypes, &mut bins, text);
    sa
}


fn sais<T: Text + ?Sized>(sa: &mut [u64], stypes: &mut SuffixTypes, bins: &mut Bins, text: &T) {
    match text.len() {
        0 => return,
        1 => {
            sa[0] = 0;
            return;
        }
        _ => {}
    }
    sa.fill(0);

    stypes.compute(text);
    bins.find_sizes((0..text.len()).map(|i| text.char_at(i)));
    bins.find_tail_pointers();

    // Insert the valley suffixes.
    for i in 0..text.len() {
        if stypes.is_valley(i) {
            bins.tail_insert(sa, i, text.char_at(i));
        }
    }
    bins.find_head_pointers();

    // Insert the descending suffixes.
    let (lasti, lastc) = text.prev(text.len());
    if stypes.is_desc(lasti) {
        bins.head_insert(sa, lasti, lastc);
    }
    for i in 0..sa.len() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_desc(lasti) {
                bins.head_insert(sa, lasti, lastc);
            }
        }
    }
    // ... and the find the end of each bin.
    bins.find_tail_pointers();

    // Insert the ascending suffixes.
    for i in (0..sa.len()).rev() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_asc(lasti) {
                bins.tail_insert(sa, lasti, lastc);
            }
        }
    }

    // Find and move all wstrings to the beginning of `sa`.
    let mut num_wstrs = 0u64;
    for i in 0..sa.len() {
        let sufi = sa[i];
        if stypes.is_valley(sufi) {
            sa[num_wstrs as usize] = sufi;
            num_wstrs += 1;
        }
    }
    // This check is necessary because we don't have a sentinel, which would
    // normally guarantee at least one wstring.
    if num_wstrs == 0 {
        num_wstrs = 1;
    }

    let mut prev_sufi = 0u64; // the first suffix can never be a valley
    let mut name = 0u64;
    // We set our "name buffer" to be max u64 values. Since there are at
    // most n/2 wstrings, a name can never be greater than n/2.
    sa[num_wstrs as usize..].fill(u64::MAX);

    for i in 0..num_wstrs {
        let cur_sufi = sa[i as usize];
        if prev_sufi == 0 || !text.wstring_equal(stypes, cur_sufi, prev_sufi) {
            name += 1;
            prev_sufi = cur_sufi;
        }
        // This divide-by-2 trick only works because it's impossible to have
        // two wstrings start at adjacent locations (they must at least be
        // separated by a single descending character).
        sa[(num_wstrs + (cur_sufi / 2)) as usize] = name - 1;
    }

    // We've inserted the lexical names into the latter half of the suffix
    // array, but it's sparse. so let's smush them all up to the end.
    let mut j = sa.len() as u64 - 1;
    for i in (num_wstrs..(sa.len() as u64)).rev() {
        if sa[i as usize] != u64::MAX {
            sa[j as usize] = sa[i as usize];
            j -= 1;
        }
    }

    // If we have fewer names than wstrings, then there are at least 2
    // equivalent wstrings, which means we need to recurse and sort them.
    if name < num_wstrs {
        let split_at = sa.len() - (num_wstrs as usize);
        let (r_sa, r_text) = sa.split_at_mut(split_at);
        sais(
            &mut r_sa[..num_wstrs as usize],
            stypes,
            bins,
            &LexNames(r_text),
        );
        stypes.compute(text);
    } else {
        for i in 0..num_wstrs {
            let reducedi = sa[((sa.len() as u64) - num_wstrs + i) as usize];
            sa[reducedi as usize] = i;
        }
    }

    bins.find_sizes((0..text.len()).map(|i| text.char_at(i)));
    bins.find_tail_pointers();

    let mut j = sa.len() - (num_wstrs as usize);
    for i in 0..text.len() {
        if stypes.is_valley(i) {
            sa[j] = i;
            j += 1;
        }
    }
    for i in 0..num_wstrs {
        let sufi = sa[i as usize];
        sa[i as usize] = sa[(sa.len() as u64 - num_wstrs + sufi) as usize];
    }
    sa[num_wstrs as usize..].fill(0);

    for i in (0..num_wstrs).rev() {
        let sufi = sa[i as usize];
        sa[i as usize] = 0;
        bins.tail_insert(sa, sufi, text.char_at(sufi));
    }
    bins.find_head_pointers();

    let (lasti, lastc) = text.prev(text.len());
    if stypes.is_desc(lasti) {
        bins.head_insert(sa, lasti, lastc);
    }
    for i in 0..sa.len() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_desc(lasti) {
                bins.head_insert(sa, lasti, lastc);
            }
        }
    }
    bins.find_tail_pointers();

    for i in (0..sa.len()).rev() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_asc(lasti) {
                bins.tail_insert(sa, lasti, lastc);
            }
        }
    }
}

 
 struct SuffixTypes {
    types: Vec<SuffixType>,
 }
 
 #[derive(Clone, Copy, Debug, Eq)]
 enum SuffixType {
    Ascending,
    Descending,
    Valley,
 }


impl SuffixTypes {
    fn new(num_bytes: u64) -> SuffixTypes {
        SuffixTypes {
            types: vec![SuffixType::Ascending; num_bytes as usize],
        }
    }

    fn compute<T: Text + ?Sized>(&mut self, text: &T) {
        if text.len() == 0 {
            return;
        }

        let mut lasti = text.len() - 1;
        let mut lastc = text.char_at(lasti);
        self.types[lasti as usize] = SuffixType::Descending;

        for i in (0..text.len() - 1).rev() {
            let c = text.char_at(i);
            if c < lastc {
                self.types[i as usize] = SuffixType::Ascending;
            } else if c > lastc {
                self.types[i as usize] = SuffixType::Descending;
            } else {
                self.types[i as usize] = self.types[lasti as usize].inherit();
            }
            if self.types[i as usize].is_desc() && self.types[lasti as usize].is_asc() {
                self.types[lasti as usize] = SuffixType::Valley;
            }
            lastc = c;
            lasti = i;
        }
    }

    #[inline]
    fn ty(&self, i: u64) -> SuffixType {
        self.types[i as usize]
    }

    #[inline]
    fn is_asc(&self, i: u64) -> bool {
        self.ty(i).is_asc()
    }

    #[inline]
    fn is_desc(&self, i: u64) -> bool {
        self.ty(i).is_desc()
    }

    #[inline]
    fn is_valley(&self, i: u64) -> bool {
        self.ty(i).is_valley()
    }

    #[inline]
    fn equal(&self, i: u64, j: u64) -> bool {
        self.ty(i) == self.ty(j)
    }

 }
 
 impl SuffixType {
     #[inline]
     fn is_asc(&self) -> bool {
         match *self {
             Ascending | Valley => true,
             _ => false,
         }
     }
 
     #[inline]
     fn is_desc(&self) -> bool {
         if let Descending = *self {
             true
         } else {
             false
         }
     }
 
     #[inline]
     fn is_valley(&self) -> bool {
         if let Valley = *self {
             true
         } else {
             false
         }
     }
 
     fn inherit(&self) -> SuffixType {
         match *self {
             Valley => Ascending,
             _ => *self,
         }
     }
 }
 
impl PartialEq for SuffixType {
    #[inline]
    fn eq(&self, other: &SuffixType) -> bool {
        (self.is_asc() && other.is_asc()) || (self.is_desc() && other.is_desc())
    }
}
 
struct Bins {
    alphas: Vec<u64>,
    sizes: Vec<u64>,
    ptrs: Vec<u64>,
}
 
impl Bins {
    fn new() -> Bins {
        Bins {
            alphas: Vec::with_capacity(10_000),
            sizes: Vec::with_capacity(10_000),
            ptrs: Vec::new(), // re-allocated later, no worries
        }
    }

    fn find_sizes<I>(&mut self, chars: I)
    where
        I: Iterator<Item = u64>,
    {
        self.alphas.clear();
        self.sizes.fill(0);

        for c in chars {
            self.inc_size(c);
            if self.size(c) == 1 {
                self.alphas.push(c);
            }
        }
        self.alphas.par_sort();

        let ptrs_len = self.alphas[self.alphas.len() - 1] + 1;
        self.ptrs = vec![0u64; ptrs_len as usize];
    }

    fn find_head_pointers(&mut self) {
        let mut sum = 0u64;
        for &c in self.alphas.iter() {
            self.ptrs[c as usize] = sum;
            sum += self.size(c);
        }
    }

    fn find_tail_pointers(&mut self) {
        let mut sum = 0u64;
        for &c in self.alphas.iter() {
            sum += self.size(c);
            self.ptrs[c as usize] = sum - 1;
        }
    }

    #[inline]
    fn head_insert(&mut self, sa: &mut [u64], i: u64, c: u64) {
        let ptr = &mut self.ptrs[c as usize];
        sa[*ptr as usize] = i;
        *ptr += 1;
    }

    #[inline]
    fn tail_insert(&mut self, sa: &mut [u64], i: u64, c: u64) {
        let ptr = &mut self.ptrs[c as usize];
        sa[*ptr as usize] = i;
        if *ptr > 0 {
            *ptr -= 1;
        }
    }

    #[inline]
    fn inc_size(&mut self, c: u64) {
        if c as usize >= self.sizes.len() {
            self.sizes.resize(1 + (c as usize), 0);
        }
        self.sizes[c as usize] += 1;
    }

    #[inline]
    fn size(&self, c: u64) -> u64 {
        self.sizes[c as usize]
    }
}
 
/// Encapsulates iteration and indexing over text.
///
/// This enables us to expose a common interface between a `Vec<u16>` (the token values) and
/// a `Vec<u64>`. Specifically, a `Vec<u64>` is used for lexical renaming.
trait Text {

    fn len(&self) -> u64;

    fn char_at(&self, i: u64) -> u64;

    fn prev(&self, i: u64) -> (u64, u64);

    fn wstring_equal(&self, stypes: &SuffixTypes, w1: u64, w2: u64) -> bool;
}
 

impl Text for [u16] {
    fn len(&self) -> u64 {
        self.len() as u64
    }

    fn char_at(&self, i: u64) -> u64 {
        self[i as usize] as u64
    }

    fn prev(&self, i: u64) -> (u64, u64) {
        (i - 1, self[i as usize - 1] as u64)
    }

    fn wstring_equal(&self, stypes: &SuffixTypes, w1: u64, w2: u64) -> bool {
        let w1chars = self[w1 as usize..].iter().enumerate();
        let w2chars = self[w2 as usize..].iter().enumerate();
        for ((i1, c1), (i2, c2)) in w1chars.zip(w2chars) {
            let (i1, i2) = (w1 + i1 as u64, w2 + i2 as u64);
            if c1 != c2 || !stypes.equal(i1, i2) {
                return false;
            }
            if i1 > w1 && (stypes.is_valley(i1) || stypes.is_valley(i2)) {
                return true;
            }
        }
        // At this point, we've exhausted either `w1` or `w2`, which means the
        // next character for one of them should be the sentinel. Since
        // `w1 != w2`, only one string can be exhausted. The sentinel is never
        // equal to another character, so we can conclude that the wstrings
        // are not equal.
        false
    }
}

struct LexNames<'s>(&'s [u64]);

impl<'s> Text for LexNames<'s> {

    #[inline]
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    #[inline]
    fn prev(&self, i: u64) -> (u64, u64) {
        (i - 1, self.0[i as usize - 1])
    }

    #[inline]
    fn char_at(&self, i: u64) -> u64 {
        self.0[i as usize]
    }

    fn wstring_equal(&self, stypes: &SuffixTypes, w1: u64, w2: u64) -> bool {
        let w1chars = self.0[w1 as usize..].iter().enumerate();
        let w2chars = self.0[w2 as usize..].iter().enumerate();
        for ((i1, c1), (i2, c2)) in w1chars.zip(w2chars) {
            let (i1, i2) = (w1 + i1 as u64, w2 + i2 as u64);
            if c1 != c2 || !stypes.equal(i1, i2) {
                return false;
            }
            if i1 > w1 && (stypes.is_valley(i1) || stypes.is_valley(i2)) {
                return true;
            }
        }
        // At this point, we've exhausted either `w1` or `w2`, which means the
        // next character for one of them should be the sentinel. Since
        // `w1 != w2`, only one string can be exhausted. The sentinel is never
        // equal to another character, so we can conclude that the wstrings
        // are not equal.
        false
    }
}