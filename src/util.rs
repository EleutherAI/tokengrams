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
    use rand::RngCore;
    use super::*;

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