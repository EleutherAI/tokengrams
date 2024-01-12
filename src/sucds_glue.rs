//! Glue code for serializing and deserializing `sucds` types with `serde`.
use std::{fmt, io::Cursor};

use serde::{de, ser, Deserializer, Serializer};
use sucds::Serializable;

struct SucdsVisitor;

impl<'de> de::Visitor<'de> for SucdsVisitor {
    type Value = &'de [u8];

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a byte array encoding a sucds type")
    }

    fn visit_borrowed_bytes<E>(self, v: &'de [u8]) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(v)
    }
}

pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Serializable,
{
    let buf = Cursor::new(deserializer.deserialize_bytes(SucdsVisitor)?);
    T::deserialize_from(buf).map_err(de::Error::custom)
}

pub fn serialize<T, S>(val: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serializable,
{
    let mut buf = Vec::with_capacity(val.size_in_bytes());

    match val.serialize_into(&mut buf) {
        Ok(_) => serializer.serialize_bytes(&buf),
        Err(e) => Err(ser::Error::custom(e)),
    }
}
