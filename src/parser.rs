use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader, Read};

use crate::CountRecord;
use crate::{Gram, WordGram, GRAM_COUNT_SEPARATOR};

/// Parser for a *N*-gram file of counts.
/// TODO: Add example of the format.
pub struct GramsParser<R> {
    reader: BufReader<R>,
    num_grams: usize,
    num_parsed: usize,
}

impl<R> GramsParser<R>
where
    R: Read,
{
    /// Creates a new [`GramsParser`] from `BufReader` of a *N*-gram file.
    pub fn new(mut reader: BufReader<R>) -> Result<Self> {
        let num_grams = {
            let mut header = String::new();
            reader.read_line(&mut header)?;
            header.trim().parse()?
        };
        Ok(Self {
            reader,
            num_grams,
            num_parsed: 0,
        })
    }

    /// Gets the number of input grams.
    #[allow(clippy::missing_const_for_fn)]
    pub fn num_grams(&self) -> usize {
        self.num_grams
    }

    /// Parses a next [`CountRecord`].
    pub fn next_record(&mut self) -> Option<Result<CountRecord<WordGram>>> {
        self.num_parsed += 1;
        if self.num_parsed > self.num_grams {
            return None;
        }

        let mut buffer = String::new();
        self.reader.read_line(&mut buffer).ok()?;

        let items: Vec<&str> = buffer
            .trim_end()
            .split(GRAM_COUNT_SEPARATOR as char)
            .collect();
        if items.len() != 2 {
            return Some(Err(anyhow!("Invalid line, {:?}", items)));
        }

        let gram = WordGram::new(items[0].as_bytes());
        items[1].parse().map_or_else(
            |_| Some(Err(anyhow!("Parse error, {:?}", items))),
            |count| Some(Ok(CountRecord { gram, count })),
        )
    }
}

impl<R: Read> Iterator for GramsParser<R> {
    type Item = Result<CountRecord<WordGram>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_record()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_grams - self.num_parsed;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COUNT_GRAMS_1: &'static str = "4
A\t10
B\t7
C\t4
D\t1
";

    const COUNT_GRAMS_2: &'static str = "4
A A\t1
A C\t2
B B\t3
D C\t1
";

    const COUNT_GRAMS_3: &'static str = "3
A A C\t2
B B C\t1
D D D\t1
";

    #[test]
    fn test_count_grams_1() {
        let mut gp = GramsParser::new(BufReader::new(COUNT_GRAMS_1.as_bytes())).unwrap();
        assert_eq!(gp.num_grams(), 4);
        for (gram, count) in [("A", 10), ("B", 7), ("C", 4), ("D", 1)] {
            let gram = WordGram::new(gram.as_bytes());
            assert_eq!(
                gp.next_record().unwrap().unwrap(),
                CountRecord { gram, count }
            );
        }
        assert!(gp.next_record().is_none());
    }

    #[test]
    fn test_count_grams_2() {
        let mut gp = GramsParser::new(BufReader::new(COUNT_GRAMS_2.as_bytes())).unwrap();
        assert_eq!(gp.num_grams(), 4);
        for (gram, count) in [("A A", 1), ("A C", 2), ("B B", 3), ("D C", 1)] {
            let gram = WordGram::new(gram.as_bytes());
            assert_eq!(
                gp.next_record().unwrap().unwrap(),
                CountRecord { gram, count }
            );
        }
        assert!(gp.next_record().is_none());
    }

    #[test]
    fn test_count_grams_3() {
        let mut gp = GramsParser::new(BufReader::new(COUNT_GRAMS_3.as_bytes())).unwrap();
        assert_eq!(gp.num_grams(), 3);
        for (gram, count) in [("A A C", 2), ("B B C", 1), ("D D D", 1)] {
            let gram = WordGram::new(gram.as_bytes());
            assert_eq!(
                gp.next_record().unwrap().unwrap(),
                CountRecord { gram, count }
            );
        }
        assert!(gp.next_record().is_none());
    }
}
