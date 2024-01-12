use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::vocabulary::Vocabulary;
use crate::WordGram;

/// Simple implementation of [`Vocabulary`] with `HashMap`.
#[derive(Default, Debug)]
pub struct SimpleVocabulary {
    map: HashMap<String, usize>,
}

impl Vocabulary for SimpleVocabulary {
    type GramType = WordGram;

    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn build<T: IntoIterator<Item = Self::GramType>>(tokens: T) -> Result<Self> {
        let mut map = HashMap::new();
        for (id, token) in tokens.into_iter().enumerate() {
            if let Some(v) = map.insert(token.to_string(), id) {
                return Err(anyhow!("Duplicated key: {:?} => {}", token, v));
            }
        }
        Ok(Self { map })
    }

    fn get(&self, token: Self::GramType) -> Option<usize> {
        self.map.get(&token.to_string()).copied()
    }
}
