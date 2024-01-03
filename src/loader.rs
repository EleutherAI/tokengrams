mod flate2;
mod plain;

use std::io::Read;
use std::str::FromStr;

use anyhow::Result;

use crate::parser::GramsParser;

pub use crate::loader::flate2::GramsGzFileLoader;
pub use crate::loader::plain::{GramsFileLoader, GramsTextLoader};

/// Loader for a *N*-gram counts file.
pub trait GramsLoader<R>
where
    R: Read,
{
    /// Creates [`GramsParser`].
    fn parser(&self) -> Result<GramsParser<R>>;
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
