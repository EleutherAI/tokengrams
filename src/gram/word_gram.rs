use crate::gram::Gram;
use std::boxed::Box;
use std::fmt;

use crate::TOKEN_SEPARATOR;

/// A word n-gram, where tokens are u8 slices.
#[derive(Clone, Eq, PartialEq)]
pub struct WordGram {
    data: Box<[u8]>,
}

impl Gram for WordGram {
    type CharType = u8;

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
    /// let tokens = "abc de f";
    /// let mut gram = WordGram::from_str(tokens);
    ///
    /// let (gram, last) = gram.pop_token().unwrap();
    /// assert_eq!(gram.into_boxed_slice(), "abc de".as_bytes());
    /// assert_eq!(last.into_boxed_slice(), "f".as_bytes());
    ///
    /// let (gram, last) = gram.pop_token().unwrap();
    /// assert_eq!(gram.into_boxed_slice(), "abc".as_bytes());
    /// assert_eq!(last.into_boxed_slice(), "de".as_bytes());
    ///
    /// assert_eq!(gram.pop_token(), None);
    /// ```
    #[inline(always)]
    fn pop_token(&self) -> Option<(Self, Self)> {
        let data = &self.data;
        data.iter()
            .rev()
            .position(|&x| x == TOKEN_SEPARATOR)
            .map(|i| {
                let pos = data.len() - i;
                let pfx = &data[..pos - 1];
                let sfx = &data[pos..];
                (Self { data: pfx.into() }, Self { data: sfx.into() })
            })
    }

    /// Pops the first token.
    ///
    /// ```
    /// use tongrams::WordGram;
    ///
    /// let tokens = "abc de f";
    /// let mut gram = WordGram::from_str(tokens);
    ///
    /// let (front, gram) = gram.pop_front_token().unwrap();
    /// assert_eq!(front.into_boxed_slice(), "abc".as_bytes());
    /// assert_eq!(gram.into_boxed_slice(), "de f".as_bytes());
    ///
    /// let (front, gram) = gram.pop_front_token().unwrap();
    /// assert_eq!(front.into_boxed_slice(), "de".as_bytes());
    /// assert_eq!(gram.into_boxed_slice(), "f".as_bytes());
    ///
    /// assert_eq!(gram.pop_front_token(), None);
    /// ```
    #[inline(always)]
    fn pop_front_token(&self) -> Option<(Self, Self)> {
        let data = &self.data;
        data.iter().position(|&x| x == TOKEN_SEPARATOR).map(|i| {
            let pfx = &data[..i];
            let sfx = &data[i + 1..];
            (Self { data: pfx.into() }, Self { data: sfx.into() })
        })
    }

    /// Splits the gram into tokens.
    ///
    /// ```
    /// use tongrams::WordGram;
    ///
    /// let tokens = "abc de f";
    /// let mut gram = WordGram::from_str(tokens);
    ///
    /// let tokens = gram.to_unigrams();
    /// assert_eq!(tokens.len(), 3);
    /// assert_eq!(tokens[0].into_boxed_slice(), "abc".as_bytes());
    /// assert_eq!(tokens[1].into_boxed_slice(), "de".as_bytes());
    /// assert_eq!(tokens[2].into_boxed_slice(), "f".as_bytes());
    /// ```
    #[inline(always)]
    fn to_unigrams(&self) -> Vec<Self> {
        self.data
            .split(|&b| b == TOKEN_SEPARATOR)
            .map(|data| Self { data: data.into() })
            .collect()
    }
}

impl WordGram {
    /// Creates a [`WordGram`] from a string slice.
    #[inline]
    pub fn from_str(s: &str) -> Self {
        Self { data: s.as_bytes().into() }
    }
}

impl fmt::Display for WordGram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", String::from_utf8(self.data.to_vec()).unwrap())
    }
}

impl fmt::Debug for WordGram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = String::from_utf8(self.data.to_vec()).unwrap();
        f.debug_struct("WordGram").field("data", &data).finish()
    }
}