use std::fmt;

use crate::TOKEN_SEPARATOR;

/// Handler of a gram.
#[derive(Clone, Copy, Eq)]
pub struct Gram<'a, T: Clone> {
    data: &'a [T],
}

impl<'a, T: Clone> Gram<'a, T> {
    /// Creates a [`Gram`] from a byte slice.
    #[inline]
    pub const fn new(data: &'a [T]) -> Self {
        Self { data }
    }

    /// Copies `self` into a new `Vec`.
    #[inline]
    pub fn to_vec(self) -> Vec<T> {
        self.data.to_vec()
    }

    /// Gets the reference to the byte slice.
    #[inline]
    pub const fn raw(&self) -> &[T] {
        self.data
    }
}

impl<'a, T: Clone + PartialEq> PartialEq for Gram<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

/// The Debug and Display traits are only implemented for `Gram<u8>`.

impl<'a> Gram<'a, u8> {
    /// Creates a [`Gram<u8>`] from a string.
    #[inline]
    pub const fn from_str(data: &'a str) -> Self {
        Self {
            data: data.as_bytes(),
        }
    }

    /// Pops the last token.
    ///
    /// ```
    /// use tongrams::Gram;
    ///
    /// let tokens = "abc de f";
    /// let mut gram = Gram::from_str(tokens);
    ///
    /// let (gram, last) = gram.pop_token().unwrap();
    /// assert_eq!(gram.raw(), "abc de".as_bytes());
    /// assert_eq!(last.raw(), "f".as_bytes());
    ///
    /// let (gram, last) = gram.pop_token().unwrap();
    /// assert_eq!(gram.raw(), "abc".as_bytes());
    /// assert_eq!(last.raw(), "de".as_bytes());
    ///
    /// assert_eq!(gram.pop_token(), None);
    /// ```
    #[inline(always)]
    pub fn pop_token(&self) -> Option<(Self, Self)> {
        let data = self.data;
        data.iter()
            .rev()
            .position(|&x| x == TOKEN_SEPARATOR)
            .map(|i| {
                let pos = data.len() - i;
                let pfx = &data[..pos - 1];
                let sfx = &data[pos..];
                (Self { data: pfx }, Self { data: sfx })
            })
    }

    /// Pops the first token.
    ///
    /// ```
    /// use tongrams::Gram;
    ///
    /// let tokens = "abc de f";
    /// let mut gram = Gram::from_str(tokens);
    ///
    /// let (front, gram) = gram.pop_front_token().unwrap();
    /// assert_eq!(front.raw(), "abc".as_bytes());
    /// assert_eq!(gram.raw(), "de f".as_bytes());
    ///
    /// let (front, gram) = gram.pop_front_token().unwrap();
    /// assert_eq!(front.raw(), "de".as_bytes());
    /// assert_eq!(gram.raw(), "f".as_bytes());
    ///
    /// assert_eq!(gram.pop_front_token(), None);
    /// ```
    #[inline(always)]
    pub fn pop_front_token(&self) -> Option<(Self, Self)> {
        let data = self.data;
        data.iter().position(|&x| x == TOKEN_SEPARATOR).map(|i| {
            let pfx = &data[..i];
            let sfx = &data[i + 1..];
            (Self { data: pfx }, Self { data: sfx })
        })
    }

    /// Splits the gram into tokens.
    ///
    /// ```
    /// use tongrams::Gram;
    ///
    /// let tokens = "abc de f";
    /// let mut gram = Gram::from_str(tokens);
    ///
    /// let tokens = gram.split_to_tokens();
    /// assert_eq!(tokens.len(), 3);
    /// assert_eq!(tokens[0].raw(), "abc".as_bytes());
    /// assert_eq!(tokens[1].raw(), "de".as_bytes());
    /// assert_eq!(tokens[2].raw(), "f".as_bytes());
    /// ```
    #[inline(always)]
    pub fn split_to_tokens(&self) -> Vec<Self> {
        self.data
            .split(|&b| b == TOKEN_SEPARATOR)
            .map(|data| Self { data })
            .collect()
    }
}

impl<'a> fmt::Display for Gram<'a, u8> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", String::from_utf8(self.to_vec()).unwrap())
    }
}

impl<'a> fmt::Debug for Gram<'a, u8> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = String::from_utf8(self.data.to_vec()).unwrap();
        f.debug_struct("Gram").field("data", &data).finish()
    }
}
