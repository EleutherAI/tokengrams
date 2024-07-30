use std::collections::HashMap;

pub trait Countable: Send + Sync {
    fn count_next_slice(&self, query: &[u16], vocab: Option<u16>) -> Vec<usize>;
    
    /// Generate a frequency map from occurrence frequency to the number of 
    /// unique n-grams in the corpus with that frequency.
    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize>;
}