pub struct BitStream<'a> {
    pub buf: &'a [u8],
    pub idx: usize,
}

impl<'a> BitStream<'a> {
    pub fn new(buf: &'a [u8]) -> BitStream<'a> {
        BitStream { buf, idx: 0 }
    }
    pub fn u(&mut self, sz: usize) -> u32 {
        assert!(sz <= 32);
        let mut tmp = 0u32;
        for _ in 0..sz {
            tmp = (tmp<<1) | (self.next().unwrap() as u32);
        }
        tmp
    }
    pub fn ue(&mut self) -> u32 {
        let mut leading_zeros = 0;
        loop {
            let b = self.next().unwrap();
            if b == 0 {
                leading_zeros += 1;
            } else {
                break;
            }
        }
        assert!(leading_zeros <= 32);
        2u32.pow(leading_zeros) - 1 + self.u(leading_zeros as usize)
    }
    pub fn cabac_byte_align(&mut self) {
        while self.idx % 8 != 0 {
            assert!(self.next().unwrap() == 1);
        }
    }
}

impl<'a> Iterator for BitStream<'a> {
    // this should be bool
    // but i prefer typing 0, 1 to true, false
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        let byte_idx = self.idx / 8;
        let bit_idx = self.idx % 8;
        if byte_idx < self.buf.len() {
            let byte = self.buf[byte_idx];
            let bit = (byte >> (7-bit_idx)) & 0x1;
            self.idx += 1;
            assert!(bit == 0 || bit == 1);
            Some(bit)
        } else {
            None
        }
    }
}

