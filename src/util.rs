use std::path::Path;

use anyhow::Result;

use crate::loader::{GramSource, GramsFileLoader, GramsGzFileLoader};
use crate::vocabulary::{DoubleArrayVocabulary, Vocabulary};
use crate::{CountRecord, GramsFileFormats, WordGram};

/// Loads all of [`CountRecord`] from a file.
///
/// # Arguments
///
///  - `filepath`: *N*-gram counts file.
///  - `fmt`: File format.
pub fn load_records_from_file<P>(
    filepath: P,
    fmt: GramsFileFormats,
) -> Result<Vec<CountRecord<WordGram>>>
where
    P: AsRef<Path>,
{
    match fmt {
        GramsFileFormats::Plain => {
            let loader = GramsFileLoader::new(filepath);
            load_records(loader)
        }
        GramsFileFormats::Gzip => {
            let loader = GramsGzFileLoader::new(filepath);
            load_records(loader)
        }
    }
}

/// Loads all of [`CountRecord`] from a gram-count file.
fn load_records<L: GramSource>(loader: L) -> Result<Vec<CountRecord<L::GramType>>> {
    let gp = loader.iter()?;
    let mut records = Vec::new();
    for rec in gp {
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
