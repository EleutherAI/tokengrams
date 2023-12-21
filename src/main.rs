use clap::Parser;
use rayon::prelude::*;
use serde_json::Value;
use std::{
    boxed::Box,
    error::Error,
    fs::{self, File},
    io::{BufReader, BufRead},
    path::PathBuf,
};
use tokenizers::tokenizer::Tokenizer;
use zstd::stream::read::Decoder;
//use tongrams::EliasFanoTrieCountLm;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of HuggingFace tokenizer to use
    #[arg(short, long)]
    model: String,

    /// Path to the directory of zstandard files
    #[arg(short, long)]
    path: PathBuf,
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = Args::parse();

    // Collect a vector of all *.zst paths in the directory.
    let paths = fs::read_dir(args.path)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("zst"))
        .collect::<Vec<_>>();

    // Load the tokenizer.
    let tokenizer = Tokenizer::from_pretrained(&args.model, None)?;

    // Process each file in parallel.
    paths.into_par_iter().try_for_each(|path| {
        let reader = BufReader::new(Decoder::new(File::open(path)?)?);

        // For each line in the file, parse the JSON and print the text.
        for line in reader.lines().take(10) {
            let row: Value = serde_json::from_str(&line?)?;

            if let Some(text) = row["text"].as_str() {
                let encoding = tokenizer.encode(text, false)?;
                println!("{}", encoding.get_tokens().join(" "));
            }
        }
        Ok(())
    })
}
