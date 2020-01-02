use std::io;
use std::fmt;

#[derive(Debug)]
enum NalType {
    Sps,
    Pps,
    Sei,
    SliceIdr,
    SliceNonIdr,
}

fn lookup_nal_type(x: u8) -> NalType {
    match x {
        1 => NalType::SliceNonIdr,
        5 => NalType::SliceIdr,
        6 => NalType::Sei,
        7 => NalType::Sps,
        8 => NalType::Pps,
        _ => panic!("unknown nal type: {}", x),
    }
}

struct Nal {
    nal_ref_idc: u8,
    nal_unit_type: NalType,
    rbsp: Vec<u8>,
}

impl Nal {
    fn new(nal_bytes: &[u8]) -> Nal {
        let nal_ref_idc = (nal_bytes[0] & 0b0110_0000) >> 5;
        let nal_unit_type = lookup_nal_type(nal_bytes[0] & 0b0001_1111);
        let mut rbsp = vec![];
        let mut i = 1;
        while i < nal_bytes.len() {
            if i < nal_bytes.len() - 2 && &nal_bytes[i..i+2] == &[0,0,3] {
                rbsp.extend(&[0,0]);
                i += 3;
            } else {
                rbsp.push(nal_bytes[i]);
                i += 1;
            }
        }
        return Nal { nal_ref_idc, nal_unit_type, rbsp };
    }
}

impl fmt::Debug for Nal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Nal {{ ref_idc: {}, type: {:?}, rbsp.len(): {} }}", 
            self.nal_ref_idc, self.nal_unit_type, self.rbsp.len())
    }
}

struct NalReader<T: io::Read> {
    r: T,
    buf: Vec<u8>,
    eos: bool,
}

impl<T: io::Read> NalReader<T> {
    fn new(mut r: T) -> NalReader<T> {
        // starts with start code
        let mut pfx = [0; 4];
        r.read_exact(&mut pfx).unwrap();
        assert!(&pfx == &[0,0,0,1]);
        NalReader { r, buf: vec![], eos: false }
    }
}

fn find_subseq<T: PartialEq>(haystack: &[T], needle: &[T]) -> Option<usize> {
    return haystack.windows(needle.len()).position(|w| w == needle);
}

impl<T: io::Read> Iterator for NalReader<T> {
    type Item = Nal;
    fn next(&mut self) -> Option<Self::Item> {
        if self.eos {
            return None;
        }
        // todo: welcome to copy city!!
        // this is hilariously inefficient
        // and poorly written
        loop {
            let code_pos = find_subseq(&self.buf, &[0,0,0,1]).map(|pos| (pos, 4))
                .or_else(|| find_subseq(&self.buf, &[0,0,1]).map(|pos| (pos, 3)));
            match code_pos {
                Some((idx, len)) => {
                    let nal = Nal::new(&self.buf[..idx]);
                    self.buf = self.buf[idx+len..].to_vec();
                    return Some(nal);
                },
                None => {
                    let mut buf2 = vec![0; 1024];
                    let nread = self.r.read(&mut buf2[..]).unwrap();
                    if nread == 0 {
                        self.eos = true;
                        return Some(Nal::new(&self.buf[..]));
                    }
                    self.buf.extend(&buf2[..nread]);
                },
            }
        }
    }
}

fn main() -> io::Result<()> {
    println!("starting tv264");

    let stdin = io::stdin();
    for nal in NalReader::new(stdin.lock()) {
        println!("got nal: {:?}", nal);
    }

    Ok(())
}
