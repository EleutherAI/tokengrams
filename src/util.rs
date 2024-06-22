use funty::Unsigned;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Essentially np.bincount(data) in parallel.
pub fn par_bincount<'a, I, T>(data: &'a I) -> Vec<usize>
where
    I: IntoParallelRefIterator<'a, Item = T>,
    T: Unsigned,
{
    // Find the maximum value in the data
    let max = match data.par_iter().max() {
        Some(m) => m,
        None => return Vec::new(),
    };

    // Create a vector of atomic counters
    let mut counts = Vec::with_capacity(max.as_usize() + 1);
    for _ in 0..=max.as_usize() {
        counts.push(AtomicUsize::new(0));
    }

    // Increment the counters in parallel
    data.par_iter().for_each(|x| {
        counts[x.as_usize()].fetch_add(1, Ordering::Relaxed);
    });
    counts.into_iter().map(|c| c.into_inner()).collect()
}

/// Return a zero-copy view of the given slice with the given type.
/// The resulting view has the same lifetime as the provided slice.
#[inline]
pub fn transmute_slice<'a, T, U>(slice: &'a [T]) -> &'a [U] {
    // SAFETY: We use floor division to ensure that we can't read past the end of the slice.
    let new_len = (slice.len() * std::mem::size_of::<T>()) / std::mem::size_of::<U>();
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const U, new_len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    macro_rules! test_transmute {
        ($bytes:ident, $type:ty) => {
            let num_bytes = std::mem::size_of::<$type>();
            let transmuted = transmute_slice::<u8, $type>(&$bytes);
            assert_eq!(transmuted.len(), $bytes.len() / num_bytes);

            for (cis, trans) in $bytes.chunks(num_bytes).zip(transmuted) {
                assert_eq!(<$type>::from_le_bytes(cis.try_into().unwrap()), *trans);
            }
        };
    }

    #[test]
    fn test_transmute_slice() {
        let mut rng = rand::thread_rng();
        let mut bytes = vec![0u8; 100];
        rng.fill_bytes(&mut bytes);

        test_transmute!(bytes, u8);
        test_transmute!(bytes, u16);
        test_transmute!(bytes, u32);
        test_transmute!(bytes, u64);
        test_transmute!(bytes, u128);
    }
}
