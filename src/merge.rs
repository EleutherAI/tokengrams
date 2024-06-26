use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

struct SuffixArrayMergeIterator<'a> {
    arrays: Vec<(&'a [u16], Box<dyn Iterator<Item = u64> + 'a>)>,
    heap: BinaryHeap<Reverse<HeapItem>>,
    next_index: u64,
}

#[derive(Eq, PartialEq)]
struct HeapItem {
    suffix_start: u64,
    array_index: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        let (self_array, self_iter) = &self.arrays[self.array_index];
        let (other_array, other_iter) = &other.arrays[other.array_index];
        
        let self_suffix = &self_array[self.suffix_start as usize..];
        let other_suffix = &other_array[other.suffix_start as usize..];
        
        self_suffix.cmp(other_suffix).reverse()
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> SuffixArrayMergeIterator<'a> {
    fn new<I>(inputs: I) -> Self
    where
        I: IntoIterator<Item = (&'a [u16], &'a [u64])>,
    {
        let mut arrays = Vec::new();
        let mut heap = BinaryHeap::new();
        let mut next_index = 0;

        for (i, (array, sa)) in inputs.into_iter().enumerate() {
            let iter = Box::new(sa.iter().cloned());
            arrays.push((array, iter));
            if let Some(first) = arrays[i].1.next() {
                heap.push(Reverse(HeapItem {
                    suffix_start: first,
                    array_index: i,
                }));
            }
            next_index += array.len() as u64;
        }

        SuffixArrayMergeIterator {
            arrays,
            heap,
            next_index,
        }
    }
}

impl<'a> Iterator for SuffixArrayMergeIterator<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.heap.pop().map(|Reverse(item)| {
            let result = item.suffix_start + 
                (0..item.array_index).map(|i| self.arrays[i].0.len() as u64).sum::<u64>();

            if let Some(next_suffix) = self.arrays[item.array_index].1.next() {
                self.heap.push(Reverse(HeapItem {
                    suffix_start: next_suffix,
                    array_index: item.array_index,
                }));
            }

            result
        })
    }
}

fn merge_suffix_arrays<'a, I>(inputs: I) -> SuffixArrayMergeIterator<'a>
where
    I: IntoIterator<Item = (&'a [u16], &'a [u64])>,
{
    SuffixArrayMergeIterator::new(inputs)
}

fn main() {
    let x = vec![1, 2, 3];
    let y = vec![2, 3, 4];
    let z = vec![3, 4, 5];
    let sa_x = vec![2, 1, 0];
    let sa_y = vec![2, 1, 0];
    let sa_z = vec![2, 1, 0];

    let inputs = vec![(&x[..], &sa_x[..]), (&y[..], &sa_y[..]), (&z[..], &sa_z[..])];
    let merged = merge_suffix_arrays(inputs);
    let result: Vec<u64> = merged.collect();
    println!("Merged suffix array: {:?}", result);
}