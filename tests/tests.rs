extern crate quickcheck;
extern crate utf16_literal;

use quickcheck::{QuickCheck, Testable};
use tokengrams::{SuffixTable, InMemoryIndex};
use utf16_literal::utf16;

fn sais(text: &str) -> SuffixTable {
    SuffixTable::new(text.encode_utf16().collect::<Vec<_>>(), None, false)
}

fn qc<T: Testable>(f: T) {
    QuickCheck::new().tests(1000).max_tests(10000).quickcheck(f);
}

fn utf16_as_usize(s: &str) -> Vec<usize> {
    s.encode_utf16().map(|x| x as usize).collect()
}

// Do some testing on substring search.

#[test]
fn empty_find_empty() {
    let sa = sais("");
    assert_eq!(sa.positions(&[]), &[]);
    assert!(!sa.contains(&[]));
}

#[test]
fn empty_find_one() {
    let sa = sais("");
    assert_eq!(sa.positions(utf16!("a")), &[]);
    assert!(!sa.contains(utf16!("a")));
}

#[test]
fn empty_find_two() {
    let sa = sais("");
    assert_eq!(sa.positions(utf16!("ab")), &[]);
    assert!(!sa.contains(utf16!("ab")));
}

#[test]
fn one_find_empty() {
    let sa = sais("a");
    assert_eq!(sa.positions(utf16!("")), &[]);
    assert!(!sa.contains(utf16!("")));
}

#[test]
fn one_find_one_notexists() {
    let sa = sais("a");
    assert_eq!(sa.positions(utf16!("b")), &[]);
    assert!(!sa.contains(utf16!("b")));
}

#[test]
fn one_find_one_exists() {
    let sa = sais("a");
    assert_eq!(sa.positions(utf16!("a")), &[0]);
    assert!(sa.contains(utf16!("a")));
}

#[test]
fn two_find_one_exists() {
    let sa = sais("ab");
    assert_eq!(sa.positions(utf16!("b")), &[1]);
    assert!(sa.contains(utf16!("b")));
}

#[test]
fn two_find_two_exists() {
    let sa = sais("aa");
    assert_eq!(vec![1, 0], sa.positions(utf16!("a")));
    assert!(sa.contains(utf16!("a")));
}

#[test]
fn many_exists() {
    let sa = sais("zzzzzaazzzzz");
    assert_eq!(vec![5, 6], sa.positions(utf16!("a")));
    assert!(sa.contains(utf16!("a")));
}

#[test]
fn many_exists_long() {
    let sa = sais("zzzzabczzzzzabczzzzzz");
    assert_eq!(sa.positions(utf16!("abc")), &[4, 12]);
    assert!(sa.contains(utf16!("abc")));
}

#[test]
fn query_longer() {
    let sa = sais("az");
    assert_eq!(sa.positions(utf16!("mnomnomnomnomnomnomno")), &[]);
    assert!(!sa.contains(utf16!("mnomnomnomnomnomnomno")));
}

#[test]
fn query_longer_less() {
    let sa = sais("zz");
    assert_eq!(sa.positions(utf16!("mnomnomnomnomnomnomno")), &[]);
    assert!(!sa.contains(utf16!("mnomnomnomnomnomnomno")));
}

#[test]
fn query_longer_greater() {
    let sa = sais("aa");
    assert_eq!(sa.positions(utf16!("mnomnomnomnomnomnomno")), &[]);
    assert!(!sa.contains(utf16!("mnomnomnomnomnomnomno")));
}

#[test]
fn query_spaces() {
    let sa = sais("The quick brown fox was very quick.");
    assert_eq!(sa.positions(utf16!("quick")), &[4, 29]);
}

#[test]
fn prop_length() {
    fn prop(s: String) -> bool {
        s.encode_utf16().count() == sais(&s).len()
    }
    qc(prop as fn(String) -> bool);
}

#[test]
fn prop_contains() {
    fn prop(s: String, c: u8) -> bool {
        let c = (c as char).to_string();
        let c16 = c.encode_utf16().collect::<Vec<_>>();
        s.contains(&c) == sais(&s).contains(c16.as_slice())
    }
    qc(prop as fn(String, u8) -> bool);
}

#[test]
fn prop_positions() {
    fn prop(s: String, c: u16) -> bool {
        let s = s.encode_utf16().collect::<Vec<_>>();
        let table = SuffixTable::new(s.clone(), None, false);

        let got = table.positions(&[c]);
        for (i, c_) in s.into_iter().enumerate() {
            if (c_ == c) != got.contains(&(i as u64)) {
                return false;
            }
        }
        true
    }
    qc(prop as fn(String, u16) -> bool);
}

#[test]
fn sample_unsmoothed_exists() {
    let s = utf16_as_usize("aaaa");
    let a = &s[0..1];

    let index = InMemoryIndex::new(s.clone(), None, false);
    let seqs = index
        .sample_unsmoothed(a.to_vec(), 3, 10, 20)
        .unwrap();

    assert_eq!(*seqs[0].last().unwrap(), a[0]);
    assert_eq!(*seqs[19].last().unwrap(), a[0]);
}

#[test]
fn sample_unsmoothed_empty_query_exists() {
    let s = utf16_as_usize("aaa");
    let a = s[0];
    let index = InMemoryIndex::new(s.clone(), None, false);
    let seqs = index
        .sample_unsmoothed(Vec::new(), 3, 10, 20)
        .unwrap();

    assert_eq!(*seqs[0].last().unwrap(), a);
    assert_eq!(*seqs[19].last().unwrap(), a);
}

#[test]
fn sample_smoothed_exists() {
    let s = utf16_as_usize("aabbccabccba");
    let mut index = InMemoryIndex::new(s.clone(), None, false);

    let tokens = &index
        .sample_smoothed(s[0..1].to_vec(), 3, 10, 1)
        .unwrap()[0];

    assert_eq!(tokens.len(), 11);
}

#[test]
fn sample_smoothed_unigrams_exists() {
    let s = utf16_as_usize("aabbccabccba");
    let mut index = InMemoryIndex::new(s.clone(), None, false);

    let tokens = &index
        .sample_smoothed(s[0..1].to_vec(), 1, 10, 10)
        .unwrap()[0];

    assert_eq!(tokens.len(), 11);
}

#[test]
fn prop_sample() {
    fn prop(s: String) -> bool {
        let s = utf16_as_usize(&s);
        if s.len() < 2 {
            return true;
        }

        let query = match s.get(0..1) {
            Some(slice) => slice,
            None => &[],
        };
        let index = InMemoryIndex::new(s.clone(), None, false);

        let got = &index
            .sample_unsmoothed(query.to_vec(), 2, 1, 1)
            .unwrap()[0];
        s.contains(got.first().unwrap())
    }

    qc(prop as fn(String) -> bool);
}

#[test]
fn smoothed_probs_exists() {
    let tokens = "aaaaaaaabc".to_string();

    let sa: SuffixTable = sais(&tokens);
    let query = vec![utf16!("b")[0]];
    let vocab = utf16!("c")[0] + 1;
    let a = utf16!("a")[0] as usize;
    let c = utf16!("c")[0] as usize;

    let bigram_counts = sa.count_next(&query);
    let unsmoothed_probs = bigram_counts
        .iter()
        .map(|&x| x as f64 / bigram_counts.iter().sum::<usize>() as f64)
        .collect::<Vec<f64>>();

    let s = utf16_as_usize(&tokens);
    let query = utf16_as_usize("b");
    let mut index = InMemoryIndex::new(s.clone(), Some(vocab as usize), false);
    let smoothed_probs = index.get_smoothed_probs(query);

    // The naive bigram probability for query 'b' is p(c) = 1.0.
    assert!(unsmoothed_probs[a] == 0.0);
    assert!(unsmoothed_probs[c] == 1.0);

    // The smoothed bigram probabilities interpolate with the lower-order unigram
    // probabilities where p(a) is high, lowering p(c)
    assert!(smoothed_probs[a] > 0.1);
    assert!(smoothed_probs[c] < 1.0);
}

#[test]
fn smoothed_probs_empty_query_exists() {
    let s = utf16_as_usize("aaa");
    let vocab = s[0] + 1;

    let mut index = InMemoryIndex::new(s, Some(vocab), false);

    let probs = index.get_smoothed_probs(Vec::new());
    let residual = (probs.iter().sum::<f64>() - 1.0).abs();

    assert!(residual < 1e-4);
}