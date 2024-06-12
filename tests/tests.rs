extern crate quickcheck;
extern crate utf16_literal;

use quickcheck::{QuickCheck, Testable};
use tokengrams::SuffixTable;
use tokengrams::memmap_index::MemmapIndex;
use utf16_literal::utf16;

fn sais(text: &str) -> SuffixTable {
    SuffixTable::new(text.encode_utf16().collect::<Vec<_>>(), false)
}

fn qc<T: Testable>(f: T) {
    QuickCheck::new().tests(1000).max_tests(10000).quickcheck(f);
}


// #[test]
// fn bus_err() {
//     let tokens_path = "/mnt/ssd-1/pile_preshuffled/deduped/document.bin".to_string();
//     let output_path = "/mnt/ssd-1/lucia/table.bin".to_string();
//     let _ = MemmapIndex::build(tokens_path, output_path, true);
// }

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
        let table = SuffixTable::new(s.clone(), false);

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
        
        let table = SuffixTable::new(s.clone(), false);

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
fn sample_benchmark() {
    let text_path = "/mnt/ssd-1/nora/pile-10G.bin".to_string();
    let index_path = "/mnt/ssd-1/nora/pile-10G.idx".to_string();
    // let text_path = "/mnt/ssd-1/nora/pile-40B.bin".to_string();
    // let index_path = "/mnt/ssd-1/nora/pile-40B.idx".to_string();

    let mmap_index = MemmapIndex::new(text_path, index_path).unwrap();
    let start = std::time::Instant::now();
    let _sample = mmap_index.batch_sample(
        vec![], 3, 2049, 100
    );
    println!("Time elapsed: {:?}", start.elapsed());
}


// extern crate memmap;
// extern stored_array crate ndarray;

// use memmap::MmapOptions;
// use ndarray::{ArrayView2, s};
// use std::fs::File;


// fn load_pile_trigram_prefixes() -> ndarray::ArrayView2<'static, u32> {
//     let file_path = "/mnt/ssd-1/pile-deduped/document.bin";
//     let file = File::open(file_path).expect("File not found");
//     let mmap = unsafe { MmapOptions::new().map(&file).expect("Error during the memory map creation") };

//     // Interpret the mmap as an array of u32, given a file of uint32
//     let data = unsafe {
//         ndarray::ArrayView2::<u32>::from_shape_ptr((2049, usize::MAX), mmap.as_ptr() as *const u32)
//     };

//     // Slice the array to get the first 1024 x 2049 block
//     let slice = data.slice(s![..1024, ..2049]);
//     let trigram_prefixes = slice.outer_iter().map(|row| {
//         row.windows(2)
//            .map(|win| win.to_vec())
//            .collect::<Vec<Vec<u80>>>()
//     }).collect::<Vec<Vec<Vec<u32>>>>();

//     trigram_prefixes
// }

// #[test]
// fn count_next_benchmark() {
//     // let text_path = "/mnt/ssd-1/nora/pile-10G.bin".to_string();
//     // let index_path = "/mnt/ssd-1/nora/pile-10G.idx".to_string();
//     let text_path = "/mnt/ssd-1/nora/pile-40B.bin".to_string();
//     let index_path = "/mnt/ssd-1/nora/pile-40B.idx".to_string();
    
//     let trigram_prefixes = load_pile_trigram_prefixes();

//     let start = std::time::Instant::now();
//     let mmap_index = MemmapIndex::new(text_path, index_path).unwrap();
//     println!("Loaded! {:?}", start.elapsed());
//     let _sample = mmap_index.batch_count_next(
//         pile, None
//     );
//     println!("Time elapsed: {:?}", start.elapsed());
// }