use std::collections::HashMap;

pub trait CountableIndex: Send + Sync {
    fn count_next(&self, query: Vec<u16>, vocab: Option<u16>) -> Vec<usize>;
    
    /// Generate a frequency map from an occurrence frequency 
    /// to the number of n-grams in the data structure with that 
    /// frequency.
    fn count_ngrams(&self, n: usize) -> HashMap<usize, usize>;
}
