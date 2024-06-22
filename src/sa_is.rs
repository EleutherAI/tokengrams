// SA-IS algorithm for suffix array construction
fn sa_is(s: &[u16]) -> Vec<usize> {
    let n = s.len();
    let mut sa = vec![0; n];
    if n <= 1 {
        return sa; // Base case: for empty or single-character strings
    }

    // Step 1: Classify each character as S-type or L-type
    let mut t = vec![false; n]; // false: L-type, true: S-type
    t[n - 1] = true; // Last character is always S-type
    for i in (0..n - 1).rev() {
        t[i] = if s[i] < s[i + 1] {
            true // S-type
        } else if s[i] > s[i + 1] {
            false // L-type
        } else {
            t[i + 1] // Same as next if equal
        };
    }

    // Step 2: Bucket counting
    let mut bkt = vec![0; 65536]; // Bucket for each possible u16 value
    for &c in s {
        bkt[c as usize] += 1;
    }

    // Calculate bucket heads
    let mut sum = 0;
    for i in 0..65536 {
        sum += bkt[i];
        bkt[i] = sum;
    }

    // Step 3: Place LMS suffixes
    sa.fill(0);
    for i in (0..n - 1).rev() {
        if t[i] && !t[i + 1] { // If current is S and next is L, it's an LMS character
            bkt[s[i] as usize] -= 1;
            sa[bkt[s[i] as usize]] = i + 1;
        }
    }

    // Step 4: Induce L-type suffixes
    induce_l(&mut sa, s, &mut bkt, &t);

    // Step 5: Induce S-type suffixes
    induce_s(&mut sa, s, &mut bkt, &t);

    sa
}

// Helper function to induce L-type suffixes
fn induce_l(sa: &mut [usize], s: &[u16], bkt: &mut [usize], t: &[bool]) {
    let n = s.len();
    // Reset bucket heads
    for i in 0..65536 {
        bkt[i] = if i == 0 { 0 } else { bkt[i - 1] };
    }
    for i in 0..n {
        if sa[i] > 0 && !t[sa[i] - 1] {
            let c = s[sa[i] - 1] as usize;
            sa[bkt[c]] = sa[i] - 1;
            bkt[c] += 1;
        }
    }
}

// Helper function to induce S-type suffixes
fn induce_s(sa: &mut [usize], s: &[u16], bkt: &mut [usize], t: &[bool]) {
    let n = s.len();
    // Reset bucket tails
    for i in (1..65536).rev() {
        bkt[i] = bkt[i - 1];
    }
    bkt[0] = 0;
    for i in (0..n).rev() {
        if sa[i] > 0 && t[sa[i] - 1] {
            let c = s[sa[i] - 1] as usize;
            bkt[c] -= 1;
            sa[bkt[c]] = sa[i] - 1;
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to verify if the suffix array is correct
    fn is_suffix_array_correct(s: &[u16], sa: &[usize]) -> bool {
        let n = s.len();
        if sa.len() != n {
            return false;
        }

        let mut used = vec![false; n];
        for &pos in sa {
            if pos >= n || used[pos] {
                return false;
            }
            used[pos] = true;
        }

        for i in 1..n {
            let suf1 = &s[sa[i - 1]..];
            let suf2 = &s[sa[i]..];
            if suf1 <= suf2 {
                continue;
            }
            return false;
        }

        true
    }

    #[test]
    fn test_empty_string() {
        let s: Vec<u16> = vec![];
        let sa = sa_is(&s);
        assert!(is_suffix_array_correct(&s, &sa));
    }

    #[test]
    fn test_single_character() {
        let s: Vec<u16> = vec![42];
        let sa = sa_is(&s);
        assert!(is_suffix_array_correct(&s, &sa));
    }

    #[test]
    fn test_two_characters() {
        let s: Vec<u16> = vec![2, 1];
        let sa = sa_is(&s);
        assert!(is_suffix_array_correct(&s, &sa));
    }

    #[test]
    fn test_repeated_characters() {
        let s: Vec<u16> = vec![1, 1, 1, 1];
        let sa = sa_is(&s);
        assert!(is_suffix_array_correct(&s, &sa));
    }

    #[test]
    fn test_ascending_sequence() {
        let s: Vec<u16> = vec![1, 2, 3, 4, 5];
        let sa = sa_is(&s);
        assert!(is_suffix_array_correct(&s, &sa));
    }

    #[test]
    fn test_descending_sequence() {
        let s: Vec<u16> = vec![5, 4, 3, 2, 1];
        let sa = sa_is(&s);
        assert!(is_suffix_array_correct(&s, &sa));
    }

    #[test]
    fn test_random_sequence() {
        let s: Vec<u16> = vec![10, 5, 8, 3, 1, 7, 2, 9, 4, 6];
        let sa = sa_is(&s);
        assert!(is_suffix_array_correct(&s, &sa));
    }

    #[test]
    fn test_large_values() {
        let s: Vec<u16> = vec![65535, 0, 32768, 1, 65534];
        let sa = sa_is(&s);
        assert!(is_suffix_array_correct(&s, &sa));
    }

    #[test]
    fn test_longer_sequence() {
        let s: Vec<u16> = (0..1000).map(|x| (x * 17 + 11) % 256).map(|x| x as u16).collect();
        let sa = sa_is(&s);
        assert!(is_suffix_array_correct(&s, &sa));
    }
}