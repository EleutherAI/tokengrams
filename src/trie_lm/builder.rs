use ahash::AHashMap as HashMap;
use anyhow::{anyhow, Result};
use sucds::int_vectors::{CompactVector, PrefixSummedEliasFano};

use crate::ef_trie_array::EliasFanoTrieArray;
use crate::loader::GramSource;
use crate::vocabulary::Vocabulary;
use crate::Gram;
use crate::TrieLm;
use crate::MAX_ORDER;

/// Builder for [`TrieLm`].
pub struct TrieLmBuilder<L, V> {
    loaders: Vec<L>,
    vocab: V,
    arrays: Vec<EliasFanoTrieArray>,
    count_ranks: Vec<PrefixSummedEliasFano>,
    counts_builder: CountsBuilder,
}

impl<'a, L, V> TrieLmBuilder<L, V>
where
    L: GramSource,
    V: Vocabulary<GramType = L::GramType>,
{
    /// Creates [`TrieLmBuilder`] from loaders.
    pub fn new(loaders: Vec<L>) -> Result<Self> {
        if MAX_ORDER < loaders.len() {
            return Err(anyhow!("loaders.len() must be no more than {}", MAX_ORDER));
        }
        Ok(Self {
            loaders,
            vocab: V::new(),
            arrays: vec![],
            count_ranks: vec![],
            counts_builder: CountsBuilder::default(),
        })
    }

    /// Builds [`TrieLm`].
    pub fn build(mut self) -> Result<TrieLm<V>> {
        self.build_counts()?;
        self.build_vocabulary()?;

        let max_order = self.loaders.len() - 1;
        for order in 1..=max_order {
            self.build_sorted_array(order)?;
        }

        Ok(TrieLm {
            vocab: self.vocab,
            arrays: self.arrays,
            count_ranks: self.count_ranks,
            counts: self.counts_builder.release(),
        })
    }

    fn build_counts(&mut self) -> Result<()> {
        for loader in &self.loaders {
            let gp = loader.iter()?;
            for rec in gp {
                self.counts_builder.eat_value(rec?.count);
            }
            self.counts_builder.build_sequence()?;
        }
        Ok(())
    }

    fn build_vocabulary(&mut self) -> Result<()> {
        let records = {
            let gp = self.loaders[0].iter()?;
            let mut records = Vec::new();
            for rec in gp {
                let rec = rec?;
                records.push(rec);
            }
            records
        };

        let mut count_ranks = Vec::with_capacity(records.len());
        for rec in &records {
            let count_rank = self.counts_builder.rank(0, rec.count).unwrap();
            count_ranks.push(count_rank);
        }
        self.count_ranks.push(PrefixSummedEliasFano::from_slice(&count_ranks)?);

        let grams: Vec<_> = records.into_iter().map(|r| r.gram).collect();
        self.vocab = V::build(grams)?;
        Ok(())
    }

    /// Builds the sorted array of `order`.
    fn build_sorted_array(&mut self, order: usize) -> Result<()> {
        let mut prev_gp = self.loaders[order - 1].iter()?;
        let curr_gp = self.loaders[order].iter()?;

        let mut count_ranks = Vec::new();
        let mut token_ids = Vec::new();
        if let (_, Some(hi)) = curr_gp.size_hint() {
            count_ranks.reserve_exact(hi);
            token_ids.reserve_exact(hi);
        }

        let mut pointers = Vec::new();
        if let (_, Some(hi)) = prev_gp.size_hint() {
            pointers.reserve_exact(hi + 1);
        }
        pointers.push(0);

        let mut pointer = 0;
        let mut prev_rec = prev_gp.next().unwrap()?;

        for curr_rec in curr_gp {
            // NOTE:
            // in a FORWARD trie, 'pattern' is the predecessor of 'gram'
            // and 'token' is the last token of 'gram'
            let curr_rec = curr_rec?;
            let (pattern, token) = curr_rec.gram.pop_token().unwrap(); // TODO: Error handling

            while pattern != prev_rec.gram {
                // NOTE:
                // this test is here only to
                // guarantee termination in
                // case of wrong data:
                // 'pattern' should ALWAYS
                // be found within previous order grams
                pointers.push(pointer);
                if let Some(rec) = prev_gp.next() {
                    prev_rec = rec?;
                } else {
                    return Err(anyhow!("{}-grams data is incomplete.", order + 1));
                }
            }
            pointer += 1;

            let token_id = self.vocab.get(token).unwrap();
            let count_rank = self.counts_builder.rank(order, curr_rec.count).unwrap();
            token_ids.push(token_id);
            count_ranks.push(count_rank);
        }

        for _ in prev_gp {
            pointers.push(pointer);
        }
        pointers.push(pointer);

        self.arrays.push(EliasFanoTrieArray::build(token_ids, pointers)?);
        self.count_ranks.push(PrefixSummedEliasFano::from_slice(&count_ranks)?);
        Ok(())
    }
}

#[derive(Default)]
pub struct CountsBuilder {
    // Mapping from eaten values to their frequencies
    v2f_map: HashMap<usize, usize>,
    // Mappings from eaten values to their ranks
    v2r_maps: Vec<HashMap<usize, usize>>,
    // In which values are sorted in decreasing order of their frequencies
    sorted_sequences: Vec<CompactVector>,
}

impl CountsBuilder {
    #[allow(clippy::missing_const_for_fn)]
    pub fn release(self) -> Vec<CompactVector> {
        self.sorted_sequences
    }

    pub fn eat_value(&mut self, x: usize) {
        if let Some(e) = self.v2f_map.get_mut(&x) {
            *e += 1;
        } else {
            self.v2f_map.insert(x, 1);
        }
    }

    /// Builds the sequence of the current order.
    pub fn build_sequence(&mut self) -> Result<()> {
        if self.v2f_map.is_empty() {
            self.v2r_maps.push(HashMap::new());
            self.sorted_sequences.push(CompactVector::default());
            return Ok(());
        }

        let mut sorted = vec![];
        let mut max_value = 0;

        for (&value, &freq) in &self.v2f_map {
            sorted.push((value, freq));
            max_value = std::cmp::max(max_value, value);
        }
        self.v2f_map.clear();

        // `then_with` is needed to stably sort
        sorted.sort_by(|(v1, f1), (v2, f2)| f2.cmp(f1).then_with(|| v1.cmp(v2)));

        // TODO: maybe don't unwrap here?
        let mut values =
            CompactVector::with_capacity(sorted.len(), sucds::utils::needed_bits(max_value))?;
        sorted
            .iter()
            .try_for_each(|&(v, _)| values.push_int(v))?;
        self.sorted_sequences.push(values);

        let mut v2r_map = HashMap::new();
        for (i, &(v, _)) in sorted.iter().enumerate() {
            v2r_map.insert(v, i);
        }
        self.v2r_maps.push(v2r_map);
        Ok(())
    }

    pub fn rank(&self, order: usize, value: usize) -> Option<usize> {
        self.v2r_maps[order].get(&value).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let seqs = vec![vec![2, 2, 1, 2, 4, 2, 1, 2, 1], vec![2, 1, 2, 1, 1, 1]];

        let mut scb = CountsBuilder::default();
        for seq in &seqs {
            for &x in seq {
                scb.eat_value(x);
            }
            scb.build_sequence().unwrap();
        }

        assert_eq!(scb.rank(0, 1), Some(1));
        assert_eq!(scb.rank(0, 2), Some(0));
        assert_eq!(scb.rank(0, 3), None);
        assert_eq!(scb.rank(0, 4), Some(2));
        assert_eq!(scb.rank(1, 1), Some(0));
        assert_eq!(scb.rank(1, 2), Some(1));

        let counts = scb.release();
        assert_eq!(counts[0].get_int(0), Some(2));
        assert_eq!(counts[0].get_int(1), Some(1));
        assert_eq!(counts[0].get_int(2), Some(4));
        assert_eq!(counts[1].get_int(0), Some(1));
        assert_eq!(counts[1].get_int(1), Some(2));
    }
}
