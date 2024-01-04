use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader, Read};

use crate::{GRAM_COUNT_SEPARATOR, Gram, WordGram};
use crate::{CountRecord, ProbRecord};

/// Parser for a *N*-gram file of counts or probs/backoff-weights.
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
    pub fn next_count_record(&mut self) -> Option<Result<CountRecord<WordGram>>> {
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

    /// Parses a next [`ProbRecord`].
    pub fn next_prob_record(&mut self) -> Option<Result<ProbRecord<WordGram>>> {
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
        if items.len() != 2 && items.len() != 3 {
            return Some(Err(anyhow!("Invalid line, {:?}", items)));
        }

        let gram = WordGram::new(items[0].as_bytes());
        items[1].parse().map_or_else(
            |_| Some(Err(anyhow!("Parse error, {:?}", items))),
            |prob| {
                let prob = if prob > 0.0 {
                    eprintln!(
                        "Warning: positive log10 probability detected. This will be mapped to 0."
                    );
                    0.0
                } else {
                    prob
                };
                if let Some(x) = items.get(2) {
                    x.parse().map_or_else(
                        |_| Some(Err(anyhow!("Parse error, {:?}", items))),
                        |backoff| Some(Ok(ProbRecord { gram, prob, backoff })),
                    )
                } else {
                    Some(Ok(ProbRecord { gram, prob, backoff: 0.0 }))
                }
            },
        )
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
                gp.next_count_record().unwrap().unwrap(),
                CountRecord { gram, count }
            );
        }
        assert!(gp.next_count_record().is_none());
    }

    #[test]
    fn test_count_grams_2() {
        let mut gp = GramsParser::new(BufReader::new(COUNT_GRAMS_2.as_bytes())).unwrap();
        assert_eq!(gp.num_grams(), 4);
        for (gram, count) in [("A A", 1), ("A C", 2), ("B B", 3), ("D C", 1)] {
            let gram = WordGram::new(gram.as_bytes());
            assert_eq!(
                gp.next_count_record().unwrap().unwrap(),
                CountRecord { gram, count }
            );
        }
        assert!(gp.next_count_record().is_none());
    }

    #[test]
    fn test_count_grams_3() {
        let mut gp = GramsParser::new(BufReader::new(COUNT_GRAMS_3.as_bytes())).unwrap();
        assert_eq!(gp.num_grams(), 3);
        for (gram, count) in [("A A C", 2), ("B B C", 1), ("D D D", 1)] {
            let gram = WordGram::new(gram.as_bytes());
            assert_eq!(
                gp.next_count_record().unwrap().unwrap(),
                CountRecord { gram, count }
            );
        }
        assert!(gp.next_count_record().is_none());
    }

    const PROB_GRAMS_1: &'static str = "4
A\t-1.83\t-0.74
B\t-2.01\t-0.69
C\t-2.22
D\t-1.91\t-0.62
";

    const PROB_GRAMS_2: &'static str = "4
A A\t-1.43\t-0.33
A C\t-0.59\t-0.43
B B\t-1.03\t-0.32
D C\t-1.08
";

    const PROB_GRAMS_3: &'static str = "3
A A C\t-1.12
B B C\t-0.53
D D D\t-0.98
";

    #[test]
    fn test_prob_grams_1() {
        let mut gp = GramsParser::new(BufReader::new(PROB_GRAMS_1.as_bytes())).unwrap();
        assert_eq!(gp.num_grams(), 4);
        for (gram, prob, backoff) in [
            ("A", -1.83, -0.74),
            ("B", -2.01, -0.69),
            ("C", -2.22, 0.0),
            ("D", -1.91, -0.62),
        ] {
            let gram = WordGram::new(gram.as_bytes());
            assert_eq!(
                gp.next_prob_record().unwrap().unwrap(),
                ProbRecord { gram, prob, backoff }
            );
        }
        assert!(gp.next_prob_record().is_none());
    }

    #[test]
    fn test_prob_grams_2() {
        let mut gp = GramsParser::new(BufReader::new(PROB_GRAMS_2.as_bytes())).unwrap();
        assert_eq!(gp.num_grams(), 4);
        for (gram, prob, backoff) in [
            ("A A", -1.43, -0.33),
            ("A C", -0.59, -0.43),
            ("B B", -1.03, -0.32),
            ("D C", -1.08, 0.0),
        ] {
            let gram = WordGram::new(gram.as_bytes());
            assert_eq!(
                gp.next_prob_record().unwrap().unwrap(),
                ProbRecord { gram, prob, backoff }
            );
        }
        assert!(gp.next_prob_record().is_none());
    }

    #[test]
    fn test_prob_grams_3() {
        let mut gp = GramsParser::new(BufReader::new(PROB_GRAMS_3.as_bytes())).unwrap();
        assert_eq!(gp.num_grams(), 3);
        for (gram, prob, backoff) in [
            ("A A C", -1.12, 0.0),
            ("B B C", -0.53, 0.0),
            ("D D D", -0.98, 0.0),
        ] {
            let gram = WordGram::new(gram.as_bytes());
            assert_eq!(
                gp.next_prob_record().unwrap().unwrap(),
                ProbRecord { gram, prob, backoff }
            );
        }
        assert!(gp.next_prob_record().is_none());
    }
}
