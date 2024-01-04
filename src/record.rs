use float_cmp::ApproxEq;

use crate::Gram;

/// Handler of a pair of a gram and its count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountRecord<G: Gram> {
    pub gram: G,
    pub count: usize,
}

/// Handler of a tuple of a gram, its probability, and its backoff weight.
#[derive(Clone, Debug)]
pub struct ProbRecord<G: Gram> {
    pub gram: G,
    pub prob: f32,
    pub backoff: f32,
}

impl<G: Gram> PartialEq for ProbRecord<G> {
    fn eq(&self, other: &Self) -> bool {
        self.gram == other.gram
            && self.prob.approx_eq(other.prob, (0.0, 2))
            && self.backoff.approx_eq(other.backoff, (0.0, 2))
    }
}

impl<G: Gram> Eq for ProbRecord<G> {}
