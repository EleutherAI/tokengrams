use std::io::Read;

use anyhow::{anyhow, Result};

use super::TrieProbLm;
use crate::loader::GramsLoader;
use crate::trie_array::TrieArray;
use crate::vocabulary::Vocabulary;
use crate::{Gram, WordGram};
use crate::MAX_ORDER;

pub struct TrieProbLmBuilder<R, T, V> {
    loaders: Vec<Box<dyn GramsLoader<R>>>,
    vocab: V,
    arrays: Vec<T>,
    probs: Vec<Vec<f32>>,    // TODO: Quantize
    backoffs: Vec<Vec<f32>>, // TODO: Quantize
}

impl<R, T, V> TrieProbLmBuilder<R, T, V>
where
    R: Read,
    T: TrieArray,
    V: Vocabulary<GramType = WordGram>,
{
    pub fn new(loaders: Vec<Box<dyn GramsLoader<R>>>) -> Result<Self> {
        if MAX_ORDER < loaders.len() {
            return Err(anyhow!("loaders.len() must be no more than {}", MAX_ORDER));
        }
        Ok(Self {
            loaders,
            vocab: V::new(),
            arrays: vec![],
            probs: vec![],
            backoffs: vec![],
        })
    }

    pub fn build(mut self) -> Result<TrieProbLm<T, V>> {
        self.build_vocabulary()?;

        let max_order = self.loaders.len() - 1;
        for order in 1..=max_order {
            self.build_sorted_array(order)?;
        }

        Ok(TrieProbLm {
            vocab: self.vocab,
            arrays: self.arrays,
            probs: self.probs,
            backoffs: self.backoffs,
        })
    }

    fn build_vocabulary(&mut self) -> Result<()> {
        let records = {
            let mut gp = self.loaders[0].parser()?;
            let mut records = Vec::new();
            while let Some(rec) = gp.next_prob_record() {
                let rec = rec?;
                records.push(rec);
            }
            records
        };

        let mut probs = Vec::with_capacity(records.len());
        let mut backoffs = Vec::with_capacity(records.len());

        for rec in &records {
            probs.push(rec.prob);
            backoffs.push(rec.backoff);
        }
        self.probs.push(probs);
        self.backoffs.push(backoffs);

        let grams: Vec<WordGram> = records.into_iter().map(|r| r.gram).collect();
        self.vocab = V::build(grams)?;

        Ok(())
    }

    /// Builds the sorted array of `order`.
    fn build_sorted_array(&mut self, order: usize) -> Result<()> {
        let mut prev_gp = self.loaders[order - 1].parser()?;
        let mut curr_gp = self.loaders[order].parser()?;

        let mut token_ids = Vec::with_capacity(curr_gp.num_grams());
        let mut probs = vec![];
        let mut backoffs = vec![];

        let num_pointers = prev_gp.num_grams() + 1;
        let mut pointers = Vec::with_capacity(num_pointers);
        pointers.push(0);

        let mut pointer = 0;
        let mut prev_rec = prev_gp.next_prob_record().unwrap()?;

        while let Some(curr_rec) = curr_gp.next_prob_record() {
            // NOTE:
            // in a BACKWARD trie, 'pattern' is the suffix of 'gram'
            // and 'token' is the first token of 'gram'
            let curr_rec = curr_rec?;
            let (token, pattern) = curr_rec.gram.pop_front_token().unwrap(); // TODO: Error handling

            while pattern != prev_rec.gram {
                // NOTE:
                // this test is here only to
                // guarantee termination in
                // case of wrong data:
                // 'pattern' should ALWAYS
                // be found within previous order grams
                pointers.push(pointer);
                if let Some(rec) = prev_gp.next_prob_record() {
                    prev_rec = rec?;
                } else {
                    return Err(anyhow!("{}-grams data is incomplete.", order + 1));
                }
            }

            pointer += 1;

            let token_id = self.vocab.get(token).unwrap();
            token_ids.push(token_id);
            probs.push(curr_rec.prob);
            if order < self.max_order() {
                backoffs.push(curr_rec.backoff);
            }
        }

        while prev_gp.next_count_record().is_some() {
            pointers.push(pointer);
        }
        pointers.push(pointer);

        self.arrays.push(T::build(token_ids, pointers));
        self.probs.push(probs);
        if order < self.max_order() {
            self.backoffs.push(backoffs);
        }

        Ok(())
    }

    fn max_order(&self) -> usize {
        self.loaders.len() - 1
    }
}
