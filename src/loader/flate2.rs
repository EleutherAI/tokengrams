use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Result;
use flate2::read::GzDecoder;

use crate::gram::WordGram;
use crate::loader::GramSource;
use crate::parser::GramsParser;

pub struct GramsGzFileLoader {
    filepath: PathBuf,
}

impl GramsGzFileLoader {
    pub fn new<P>(filepath: P) -> Self
    where
        P: AsRef<Path>,
    {
        Self {
            filepath: PathBuf::from(filepath.as_ref()),
        }
    }
}

impl GramSource for GramsGzFileLoader {
    type GramType = WordGram;
    type Iter = GramsParser<GzDecoder<File>>;

    fn iter(&self) -> Result<GramsParser<GzDecoder<File>>> {
        let reader = GzDecoder::new(File::open(&self.filepath)?);
        GramsParser::new(BufReader::new(reader))
    }
}
