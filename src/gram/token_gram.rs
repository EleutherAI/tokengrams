use crate::gram::Gram;

/// A token n-gram, where tokens are constant-sized primitive types.
/// This obviates the need for a token separator.
#[derive(Clone, PartialEq, Eq)]
pub struct TokenGram<C = u16> {
    data: Box<[C]>,
}

impl<C: Clone + Copy + Eq> Gram for TokenGram<C> {
    type CharType = C;

    #[inline]
    fn new<T: Into<Box<[Self::CharType]>>>(data: T) -> Self {
        Self { data: data.into() }
    }

    #[inline]
    fn into_boxed_slice(self) -> Box<[Self::CharType]> {
        self.data
    }

    /// Pops the last token.
    ///
    /// ```
    /// use tongrams::WordGram;
    ///
    /// let mut gram = WordGram { data: &[0, 1, 2] };
    ///
    /// let (gram, last) = gram.pop_token().unwrap();
    /// assert_eq!(gram.into_boxed_slice(), [0, 1]);
    /// assert_eq!(last.into_boxed_slice(), [2]);
    ///
    /// let (gram, last) = gram.pop_token().unwrap();
    /// assert_eq!(gram.into_boxed_slice(), [0]);
    /// assert_eq!(last.into_boxed_slice(), [1]);
    ///
    /// assert_eq!(gram.pop_token(), None);
    /// ```
    #[inline(always)]
    fn pop_token(&self) -> Option<(Self, Self)> {
        let idx = self.data.len() - 1;
        if idx > 0 {
            Some((
                Self {
                    data: self.data[..idx].into(),
                },
                Self {
                    data: self.data[idx..].into(),
                },
            ))
        } else {
            None
        }
    }

    /// Pops the first token.
    ///
    /// ```
    /// use tongrams::WordGram;
    ///
    /// let mut gram = WordGram { data: &[0, 1, 2] };
    ///
    /// let (front, gram) = gram.pop_front_token().unwrap();
    /// assert_eq!(front.into_boxed_slice(), [0]);
    /// assert_eq!(gram.into_boxed_slice(), [1, 2]);
    ///
    /// let (front, gram) = gram.pop_front_token().unwrap();
    /// assert_eq!(front.into_boxed_slice(), [1]);
    /// assert_eq!(gram.into_boxed_slice(), [2]);
    ///
    /// assert_eq!(gram.pop_front_token(), None);
    /// ```
    #[inline(always)]
    fn pop_front_token(&self) -> Option<(Self, Self)> {
        let (left, right) = self.data.split_at(1);

        if !right.is_empty() {
            Some((Self { data: left.into() }, Self { data: right.into() }))
        } else {
            None
        }
    }

    /// Splits the gram into tokens.
    ///
    /// ```
    /// use tongrams::WordGram;
    ///
    /// let mut gram = WordGram { data: &[0, 1, 2] };
    ///
    /// let tokens = gram.to_unigrams();
    /// assert_eq!(tokens.len(), 3);
    /// assert_eq!(tokens[0].into_boxed_slice(), [0]);
    /// assert_eq!(tokens[1].into_boxed_slice(), [1]);
    /// assert_eq!(tokens[2].into_boxed_slice(), [2]);
    /// ```
    #[inline(always)]
    fn to_unigrams(&self) -> Vec<Self> {
        self.data
            .chunks(1)
            .map(|data| Self { data: data.into() })
            .collect()
    }
}
