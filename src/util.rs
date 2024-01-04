use std::io::Read;
use std::path::Path;

use anyhow::Result;

use crate::loader::{GramsFileLoader, GramsGzFileLoader, GramsLoader};
use crate::vocabulary::{DoubleArrayVocabulary, Vocabulary};
use crate::{CountRecord, WordGram, GramsFileFormats};

/// Loads all of [`CountRecord`] from a file.
///
/// # Arguments
///
///  - `filepath`: *N*-gram counts file.
///  - `fmt`: File format.
pub fn load_records_from_file<P>(filepath: P, fmt: GramsFileFormats) -> Result<Vec<CountRecord<WordGram>>>
where
    P: AsRef<Path>,
{
    match fmt {
        GramsFileFormats::Plain => {
            let loader: Box<dyn GramsLoader<_>> = Box::new(GramsFileLoader::new(filepath));
            load_records(loader)
        }
        GramsFileFormats::Gzip => {
            let loader: Box<dyn GramsLoader<_>> = Box::new(GramsGzFileLoader::new(filepath));
            load_records(loader)
        }
    }
}

/// Loads all of [`CountRecord`] from a gram-count file.
fn load_records<R: Read>(loader: Box<dyn GramsLoader<R>>) -> Result<Vec<CountRecord<WordGram>>>
where
    R: Read,
{
    let mut gp = loader.parser()?;
    let mut records = Vec::new();
    while let Some(rec) = gp.next_count_record() {
        let rec = rec?;
        records.push(rec);
    }
    Ok(records)
}

/// Builds [`DoubleArrayVocabulary`] from a file.
///
/// # Arguments
///
///  - `filepath`: *N*-gram counts file.
///  - `fmt`: File format.
pub fn build_vocabulary_from_file<P>(
    filepath: P,
    fmt: GramsFileFormats,
) -> Result<DoubleArrayVocabulary>
where
    P: AsRef<Path>,
{
    let records = load_records_from_file(filepath, fmt)?;
    let grams: Vec<_> = records.into_iter().map(|r| r.gram).collect();
    let vocab = DoubleArrayVocabulary::build(grams)?;
    Ok(vocab)
}

/// Gets the file extension of *N*-gram counts file.
pub fn get_format_extension(fmt: GramsFileFormats) -> Option<String> {
    match fmt {
        GramsFileFormats::Plain => None,
        GramsFileFormats::Gzip => Some("gz".to_string()),
    }
}
