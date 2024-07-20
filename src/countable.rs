use std::collections::HashMap;

pub trait Countable: Send + Sync {
    fn count_next_slice(&self, query: &[u16], vocab: Option<u16>) -> Vec<usize>;
    
    /// Generate a frequency map from an occurrence frequency 
    /// to the number of n-grams in the data structure with that 
    /// frequency.
    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize>;
}