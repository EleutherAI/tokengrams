extern crate quickcheck;
extern crate utf16_literal;

use quickcheck::{QuickCheck, Testable};
use tokengrams::SuffixTable;
use utf16_literal::utf16;

fn sais(text: &str) -> SuffixTable {
    SuffixTable::new(text.encode_utf16().collect::<Vec<_>>())
}

fn qc<T: Testable>(f: T) {
    QuickCheck::new().tests(1000).max_tests(10000).quickcheck(f);
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
        let table = SuffixTable::new(s.clone());

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
fn sample_query_exists() {
    let sa = sais("aaa");
    let a = utf16!("a");
    let tokens = sa.sample(a, 3, 10).unwrap();
    assert_eq!(*tokens.last().unwrap(), a[0]);
}

#[test]
fn sample_empty_query_exists() {
    let sa = sais("aaa");
    let empty_query = utf16!("");
    let tokens = sa.sample(empty_query, 3, 10).unwrap();
    assert_eq!(*tokens.last().unwrap(), utf16!("a")[0]);
}

#[test]
fn batch_sample_query_exists() {
    let sa = sais("aaa");
    let a = utf16!("a");
    let seqs = sa.batch_sample(a, 3, 10, 20).unwrap();
    assert_eq!(*seqs[0].last().unwrap(), a[0]);
    assert_eq!(*seqs[19].last().unwrap(), a[0]);
}

#[test]
fn batch_sample_empty_query_exists() {
    let sa = sais("aaa");
    let empty_query = utf16!("");
    let seqs = sa.batch_sample(empty_query, 3, 10, 20).unwrap();
    assert_eq!(*seqs[0].last().unwrap(), utf16!("a")[0]);
    assert_eq!(*seqs[19].last().unwrap(), utf16!("a")[0]);
}

#[test]
fn prop_sample() {
    fn prop(s: String) -> bool {
        let s = s.encode_utf16().collect::<Vec<_>>();
        if s.len() < 2 {
            return true;
        }
        
        let table = SuffixTable::new(s.clone());

        let query = match s.get(0..1) {
            Some(slice) => slice,
            None => &[],
        };
        let got = table.sample(query, 2, 1).unwrap();
        s.contains(got.first().unwrap())
    }

    qc(prop as fn(String) -> bool);
}

#[test]
fn prop_sais() {
    fn prop(s: String) -> bool {
        let s = s.encode_utf16().collect::<Vec<_>>();
        let table = SuffixTable::new(s.clone());
        let sais_table = SuffixTable::new_sais(s.clone());

        table == sais_table
    }
    qc(prop as fn(String) -> bool);
}