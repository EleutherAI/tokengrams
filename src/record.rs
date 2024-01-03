use float_cmp::ApproxEq;

use crate::Gram;

/// Handler of a pair of a gram and its count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountRecord {
    gram: String, // TODO: Store as a byte slice to another buffer
    count: usize,
}

impl CountRecord {
    /// Creates a [`CountRecord`].
    #[inline]
    pub const fn new(gram: String, count: usize) -> Self {
        Self { gram, count }
    }

    /// Gets the [`Gram`].
    #[inline]
    pub fn gram(&self) -> Gram<u8> {
        Gram::new(self.gram.as_bytes())
    }

    /// Gets the count.
    #[inline]
    pub const fn count(&self) -> usize {
        self.count
    }
}

/// Handler of a tuple of a gram, its probability, and its backoff weight.
#[derive(Clone, Debug)]
pub struct ProbRecord {
    gram: String, // TODO: Store as a byte slice to another buffer
    prob: f32,
    backoff: f32,
}

impl ProbRecord {
    /// Creates a [`ProbRecord`].
    #[inline]
    pub const fn new(gram: String, prob: f32, backoff: f32) -> Self {
        Self {
            gram,
            prob,
            backoff,
        }
    }

    /// Gets the [`Gram`].
    #[inline]
    pub fn gram(&self) -> Gram<u8> {
        Gram::new(self.gram.as_bytes())
    }

    /// Gets the probability.
    #[inline]
    pub const fn prob(&self) -> f32 {
        self.prob
    }

    /// Gets the backoff weight.
    #[inline]
    pub const fn backoff(&self) -> f32 {
        self.backoff
    }
}

impl PartialEq for ProbRecord {
    fn eq(&self, other: &Self) -> bool {
        self.gram == other.gram
            && self.prob.approx_eq(other.prob, (0.0, 2))
            && self.backoff.approx_eq(other.backoff, (0.0, 2))
    }
}

impl Eq for ProbRecord {}
