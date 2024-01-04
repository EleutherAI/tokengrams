use std::collections::HashMap;
use std::io::{Read, Write};

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

    fn serialize_into<W>(&self, writer: W) -> Result<usize>
    where
        W: Write,
    {
        bincode::serialize_into(writer, &self.map).map_err(handle_bincode_error)?;
        Ok(self.size_in_bytes())
    }

    fn deserialize_from<R>(reader: R) -> Result<Self>
    where
        R: Read,
    {
        let map = bincode::deserialize_from(reader).map_err(handle_bincode_error)?;
        Ok(Self { map })
    }

    fn size_in_bytes(&self) -> usize {
        bincode::serialize(&self.map)
            .map_err(handle_bincode_error)
            .unwrap()
            .len()
    }

    fn memory_statistics(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    fn get(&self, token: Self::GramType) -> Option<usize> {
        self.map.get(&token.to_string()).copied()
    }
}

fn handle_bincode_error(e: std::boxed::Box<bincode::ErrorKind>) -> anyhow::Error {
    anyhow::anyhow!("{:?}", e)
}
