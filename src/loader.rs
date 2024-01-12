mod flate2;
mod plain;

use std::str::FromStr;

use anyhow::Result;

use crate::gram::Gram;
use crate::record::CountRecord;

pub use crate::loader::flate2::GramsGzFileLoader;
pub use crate::loader::plain::{GramsFileLoader, GramsTextLoader};

/// Loader for a *N*-gram counts file.
pub trait GramSource {
    type GramType: Gram;
    type Iter: Iterator<Item = Result<CountRecord<Self::GramType>>>;

    /// Returns an iterator over fallible CountRecords for the GramType.
    fn iter(&self) -> Result<Self::Iter>;
}

/// File formats supported.
#[derive(Clone, Copy, Debug)]
pub enum GramsFileFormats {
    Plain,
    Gzip,
}

impl FromStr for GramsFileFormats {
    type Err = &'static str;

    fn from_str(fmt: &str) -> Result<Self, Self::Err> {
        match fmt {
            "plain" => Ok(Self::Plain),
            "gzip" => Ok(Self::Gzip),
            _ => Err("Invalid format"),
        }
    }
}
