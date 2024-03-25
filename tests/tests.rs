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
fn bincount_next_tokens_exists() {
    let sa = sais("aaab");
    
    let query = utf16!("a");
    let a_index = utf16!("a")[0] as usize;
    let b_index = utf16!("b")[0] as usize;

    assert_eq!(2, sa.bincount_next_tokens(query, Option::None)[a_index]);
    assert_eq!(1, sa.bincount_next_tokens(query, Option::None)[b_index]);
}

#[test]
fn bincount_next_tokens_empty_query() {
    let sa = sais("aaab");
    
    let query = utf16!("");
    let a_index = utf16!("a")[0] as usize;
    let b_index = utf16!("b")[0] as usize;

    assert_eq!(3, sa.bincount_next_tokens(query, Option::None)[a_index]);
    assert_eq!(1, sa.bincount_next_tokens(query, Option::None)[b_index]);
}

#[test]
fn sample_exists() {
    let sa = sais("ab");

    let query = utf16!("a");
    let b_index = utf16!("b")[0];

    let token = sa.sample(query, 2, 1);
    assert_eq!(*token.unwrap().last().unwrap(), b_index);
}
