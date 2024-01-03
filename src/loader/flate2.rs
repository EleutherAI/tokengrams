use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::Result;
use flate2::read::GzDecoder;

use crate::loader::GramsLoader;
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

impl GramsLoader<GzDecoder<File>> for GramsGzFileLoader {
    fn parser(&self) -> Result<GramsParser<GzDecoder<File>>> {
        let reader = GzDecoder::new(File::open(&self.filepath)?);
        GramsParser::new(BufReader::new(reader))
    }
}
