// mod super::bitstream;
use super::bitstream::BitStream;

const COEFF_TOKEN_TABLE: &[((u32,u32), (u32, &[u8]))] = &[
    // ((trailing_ones, total_coeff), (leading_zeros, [pattern...]))
    ((3,7), (6, &[1,0,0])),
    ((0,8), (9, &[1,0,0,0])),
];

fn read_coeff_token(bs: &mut BitStream) -> (u32, u32) {
    let mut lz = 0;
    let mut candidates: Vec<usize> = loop {
        if bs.next().unwrap() == 0 {
            lz += 1;
        } else { 
            let mut candidates = vec![];
            for i in 0..COEFF_TOKEN_TABLE.len() {
                let entry = COEFF_TOKEN_TABLE[i];
                if (entry.1).0 == lz {
                    candidates.push(i);
                }
            }
            break candidates;
        }
    };
    // first bit is always 1, and we already read it
    let mut pos = 1;
    while candidates.len() > 1 {
        let b = bs.next().unwrap();
        candidates = candidates.iter()
            .filter(|i| (COEFF_TOKEN_TABLE[**i].1).1[pos] == b)
            .collect();
        pos += 1;
    }
    assert!(candidates.len() == 1);
    COEFF_TOKEN_TABLE[candidates[0]].0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_read_coeff_token() {
        // assert_eq!(is_true(), true);
    }
}

