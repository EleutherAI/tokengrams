use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::gram::WordGram;
use crate::loader::GramSource;
use crate::parser::GramsParser;

pub struct GramsFileLoader {
    filepath: PathBuf,
}

impl GramsFileLoader {
    pub fn new<P>(filepath: P) -> Self
    where
        P: AsRef<Path>,
    {
        Self {
            filepath: PathBuf::from(filepath.as_ref()),
        }
    }
}

impl GramSource for GramsFileLoader {
    type GramType = WordGram;
    type Iter = GramsParser<File>;

    fn iter(&self) -> Result<GramsParser<File>> {
        let reader = BufReader::new(File::open(&self.filepath)?);
        GramsParser::new(reader)
    }
}

pub struct GramsTextLoader<'a> {
    text: &'a [u8],
}

impl<'a> GramsTextLoader<'a> {
    pub const fn new(text: &'a [u8]) -> Self {
        Self { text }
    }
}

impl<'a> GramSource for GramsTextLoader<'a> {
    type GramType = WordGram;
    type Iter = GramsParser<&'a [u8]>;

    fn iter(&self) -> Result<GramsParser<&'a [u8]>> {
        let reader = BufReader::new(self.text);
        GramsParser::new(reader)
    }
}
