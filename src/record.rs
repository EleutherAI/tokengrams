use crate::Gram;

/// Handler of a pair of a gram and its count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountRecord<G: Gram> {
    pub gram: G,
    pub count: usize,
}