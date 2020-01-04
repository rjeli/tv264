use std::io;
use std::fmt;
use std::cmp;

mod bitstream;
use bitstream::BitStream;
mod vlc;

struct Nal {
    nal_ref_idc: u8,
    nal_unit_type: u8,
    rbsp: Vec<u8>,
}

impl Nal {
    fn new(nal_bytes: &[u8]) -> Nal {
        let nal_ref_idc = (nal_bytes[0] & 0b0110_0000) >> 5;
        let nal_unit_type = nal_bytes[0] & 0b0001_1111;
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

#[derive(Debug,Copy,Clone,Default)]
struct Sps {
    profile_idc: u32,
    constraint_flags: u32,
    level_idc: u32,
    sps_id: u32,
    chroma_format_idc: u32,
    bit_depth_luma_minus8: u32,
    bit_depth_chroma_minus8: u32,
    qpprime_y_zero_transform_bypass_flag: u32,
    seq_scaling_matrix_present_flag: u32,
    log2_max_frame_num_minus4: u32,
    pic_order_cnt_type: u32,
    log2_max_pic_order_cnt_lsb_minus4: u32,
    max_num_ref_frames: u32,
    gaps_in_frame_num_value_allowed_flag: u32,
    pic_width_in_mbs_minus1: u32,
    pic_height_in_map_units_minus1: u32,
    frame_mbs_only_flag: u32,
    mb_adaptive_frame_field_flag: u32,
}

impl Sps {
    fn parse(bs: &mut BitStream) -> Sps {
        let mut sps = Sps { ..Default::default() };
        sps.profile_idc = bs.u(8);
        sps.constraint_flags = bs.u(6);
        let reserved_zeros = bs.u(2);
        assert!(reserved_zeros == 0);
        sps.level_idc = bs.u(8);
        sps.sps_id = bs.ue();
        assert!(sps.profile_idc == 100);
        if sps.profile_idc == 100 {
            sps.chroma_format_idc = bs.ue();
            assert!(sps.chroma_format_idc != 3);
            sps.bit_depth_luma_minus8 = bs.ue();
            sps.bit_depth_chroma_minus8 = bs.ue();
            sps.qpprime_y_zero_transform_bypass_flag = bs.u(1);
            sps.seq_scaling_matrix_present_flag = bs.u(1);
            assert!(sps.seq_scaling_matrix_present_flag == 0);
        }
        sps.log2_max_frame_num_minus4 = bs.ue();
        sps.pic_order_cnt_type = bs.ue();
        assert!(sps.pic_order_cnt_type == 2);
        // sps.log2_max_pic_order_cnt_lsb_minus4 = bs.ue();
        sps.max_num_ref_frames = bs.ue();
        sps.gaps_in_frame_num_value_allowed_flag = bs.u(1);
        sps.pic_width_in_mbs_minus1 = bs.ue();
        sps.pic_height_in_map_units_minus1 = bs.ue();
        sps.frame_mbs_only_flag = bs.u(1);
        if sps.frame_mbs_only_flag == 0 {
            sps.mb_adaptive_frame_field_flag = bs.u(1);
            assert!(sps.mb_adaptive_frame_field_flag == 0);
        }
        sps
    }
}

#[derive(Debug,Copy,Clone,Default)]
struct Pps {
    pps_id: u32,
    sps_id: u32,
    entropy_coding_mode_flag: u32,
    pic_order_present_flag: u32,
    num_slice_groups_minus1: u32,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
    weighted_pred_flag: u32,
    weighted_bipred_idc: u32,
    pic_init_qp_minus26: i32,
    pic_init_qs_minus26: i32,
    chroma_qp_index_offset: i32,
    deblocking_filter_control_present_flag: u32,
    constrained_intra_pred_flag: u32,
    redundant_pic_cnt_present_flag: u32,
}

impl Pps {
    fn parse(bs: &mut BitStream) -> Pps {
        let mut pps = Pps { ..Default::default() };
        pps.pps_id = bs.ue();
        pps.sps_id = bs.ue();
        pps.entropy_coding_mode_flag = bs.u(1);
        pps.pic_order_present_flag = bs.u(1);
        pps.num_slice_groups_minus1 = bs.ue();
        assert!(pps.num_slice_groups_minus1 == 0);
        pps.num_ref_idx_l0_active_minus1 = bs.ue();
        pps.num_ref_idx_l1_active_minus1 = bs.ue();
        pps.weighted_pred_flag = bs.u(1);
        pps.weighted_bipred_idc = bs.u(2);
        // todo: signed golomb
        pps.pic_init_qp_minus26 = bs.ue() as i32;
        pps.pic_init_qs_minus26 = bs.ue() as i32;
        pps.chroma_qp_index_offset = bs.ue() as i32;
        pps.deblocking_filter_control_present_flag = bs.u(1);
        pps.constrained_intra_pred_flag = bs.u(1);
        pps.redundant_pic_cnt_present_flag = bs.u(1);
        println!("after pps parse, leftover bits: {}", 
            bs.buf.len()*8 - bs.idx);
        pps
    }
}

#[derive(Debug,PartialEq)]
enum SliceType {
    I, P
}
impl Default for SliceType {
    fn default() -> Self { SliceType::I }
}

#[derive(Debug,Default)]
struct Slice {
    is_idr: bool,
    first_mb_addr: u32,
    slice_type: SliceType,
    pps_id: u32,
    frame_num: u32,
    field_pic_flag: u32,
    bottom_field_flag: u32,
    idr_pic_id: u32,
    pic_order_cnt_lsb: u32,
    no_output_of_prior_pics_flag: u32,
    long_term_reference_flag: u32,
    slice_qp_delta: i32,
    disable_deblocking_filter_idc: u32,
    slice_alpha_c0_offset_div2: i32,
    slice_beta_offset_div2: i32,
}

const Z_IDX_TO_XY: &[(u32,u32)] = &[
    (0, 0), (1, 0), (0, 1), (1, 1),
    (2, 0), (3, 0), (2, 1), (3, 1),
    (0, 2), (1, 2), (0, 3), (1, 3),
    (2, 2), (3, 2), (2, 3), (3, 3),
];

impl Slice {
    fn parse(is_idr: bool, bs: &mut BitStream, ctx: &CodecCtx) -> Slice {
        let mut slice = Slice { is_idr, ..Default::default() };
        let sps = ctx.sps.unwrap();
        let pps = ctx.pps.unwrap();
        slice.first_mb_addr = bs.ue();
        let st = bs.ue();
        slice.slice_type = match st {
            7 => SliceType::I,
            5 => SliceType::P,
            _ => panic!("unknown slice type: {}", st),
        };
        slice.pps_id = bs.ue();
        slice.frame_num = bs.u(sps.log2_max_frame_num_minus4 as usize + 4);
        if sps.frame_mbs_only_flag == 0 {
            slice.field_pic_flag = bs.u(1);
            if slice.field_pic_flag == 1 {
                slice.bottom_field_flag = bs.u(1);
            }
        }
        if is_idr {
            slice.idr_pic_id = bs.ue();
        }
        assert!(sps.pic_order_cnt_type == 2);
        assert!(pps.redundant_pic_cnt_present_flag == 0);
        if slice.slice_type != SliceType::I {
            return slice
        }
        // dec_ref_pic_marking
        slice.no_output_of_prior_pics_flag = bs.u(1);
        slice.long_term_reference_flag = bs.u(1);
        slice.slice_qp_delta = bs.ue() as i32;
        if pps.deblocking_filter_control_present_flag == 1 {
            slice.disable_deblocking_filter_idc = bs.ue();
            if slice.disable_deblocking_filter_idc != 1 {
                slice.slice_alpha_c0_offset_div2 = bs.ue() as i32;
                slice.slice_beta_offset_div2 = bs.ue() as i32;
            }
        }
        if pps.entropy_coding_mode_flag == 1 {
            bs.cabac_byte_align();
        }
        println!("after slice header parse, leftover bits: {}", 
            bs.buf.len()*8 - bs.idx);
        let mut current_mb_addr = slice.first_mb_addr;
        loop {
            // macroblock_layer()
            let mb_type = bs.ue();
            println!("mb_type: {}", mb_type);
            // not PCM or NxN
            assert!(mb_type != 25 && mb_type != 0);
            let mb_type_info = match mb_type {
                3 => (2, 0, 0),
                7 => (2, 1, 0),
                _ => panic!("unknown mb_type: {}", mb_type),
            };
            // mb_pred()
            if sps.chroma_format_idc == 1 || sps.chroma_format_idc == 2 {
                let intra_chroma_pred_mode = bs.ue();
                println!("intra_chroma_pred_mode: {}", intra_chroma_pred_mode);
            }
            // coded luma > 0 || coded chroma > 0 || mbpartpredmode == intra_16x16
            let mb_qp_delta = bs.ue() as i32;
            println!("mb_qp_delta: {}", mb_qp_delta);
            // residual(0, 15)
            // residual_luma: startIdx == 0 && MBPPM == Intra_16x16
            // residual_block(i16x16DClevel, 0, 15, 16)

            let w = sps.pic_width_in_mbs_minus1 + 1;
            let mb_x = current_mb_addr % w;
            let mb_y = current_mb_addr / w;
            println!("mb x: {} y: {}", mb_x, mb_y);

            let n_c = 0;
            assert!(n_c < 2);

            let vlc_table: &[(&[u8],(usize,usize))] = &[
                (&[0,0,0,1,0,1], (0,1)),
                (&[0,0,0,1,0,0], (1,2)),
                (&[0,0,0,0,1,0,1], (2,3)),
                (&[0,0,0,0,0,1,1,0], (1,3)),
            ];

            let mut buf = vec![];
            let (trailing_ones, total_coeff) = 'outer: loop {
                for (vlc, res) in vlc_table {
                    if buf == *vlc {
                        break 'outer *res;
                    }
                }
                buf.push(bs.next().unwrap());
                if buf.len() > 20 {
                    panic!("unknown vlc... {:?}", buf);
                }
            };
            println!("trailing_ones: {} total_coeff: {}", trailing_ones, total_coeff);

            let mut coeff_level = vec![0i64; 16];
            let mut level_val = vec![0i64; 16];
            if total_coeff > 0 {
                let mut suffix_len = if total_coeff > 10 && trailing_ones < 3 {
                    1
                } else {
                    0
                };
                for i in 0..total_coeff {
                    if i < trailing_ones {
                        level_val[i] = if bs.u(1) == 0 { 1 } else { -1 };
                    } else {
                        let mut level_prefix = 0;
                        loop {
                            if bs.u(1) == 0 {
                                level_prefix += 1;
                            } else {
                                break;
                            }
                        }
                        let mut level_code: i64 = cmp::min(level_prefix, 15) << suffix_len;
                        if suffix_len > 0 || level_prefix >= 14 {
                            let level_suffix = bs.u(suffix_len);
                            level_code += level_suffix as i64;
                        }
                        if level_prefix >= 15 && suffix_len == 0 {
                            level_code += 15;
                        }
                        if level_prefix >= 16 {
                            level_code += (1<<(level_prefix-3)) - 4096;
                        }
                        if i == trailing_ones && trailing_ones < 3 {
                            level_code += 2;
                        }
                        if level_code % 2 == 0 {
                            level_val[i] = (level_code+2) >> 1;
                        } else {
                            level_val[i] = (-level_code-1) >> 1;
                        }
                        if suffix_len == 0 {
                            suffix_len = 1;
                        }
                        if level_val[i].abs() > (3<<(suffix_len-1)) && suffix_len < 6 {
                            suffix_len += 1;
                        }
                    }
                }
                let mut zeros_left = if total_coeff < 16 {
                    if total_coeff == 1 {
                        let mut lz = 0;
                        loop {
                            if bs.next().unwrap() == 0 {
                                lz += 1;
                            } else {
                                if lz == 0 {
                                    break 0;
                                } else {
                                    if bs.next().unwrap() == 0 {
                                        break 2*lz;
                                    } else {
                                        break 2*lz - 1;
                                    }
                                }
                            }
                        }
                    } else if total_coeff == 2 {
                        if bs.next().unwrap() == 1 {
                            let n2: Vec<u8> = bs.take(2).collect();
                            match &n2[..] {
                                &[1,1] => 0,
                                &[1,0] => 1,
                                &[0,1] => 2,
                                &[0,0] => 3,
                                _ => panic!(),
                            }
                        } else {
                            todo!();
                        }
                    } else {
                        todo!("total_coeff {}", total_coeff);
                    }
                } else {
                    0
                };
                let mut run_val = vec![0; total_coeff];
                for i in 0..total_coeff-1 {
                    run_val[i] = if zeros_left > 0 {
                        todo!(); // run_before
                    } else {
                        0
                    };
                    zeros_left -= run_val[i];
                }
                run_val[total_coeff-1] = zeros_left;
                let mut coeff_num: i32 = -1;
                for i in (0..total_coeff).rev() {
                    coeff_num += run_val[i] + 1;
                    coeff_level[coeff_num as usize] = level_val[i];
                }
            }

            // let end_of_slice = bs.ae();
            break;
        }
        slice
    }
}

/*
fn get_neighbor(mb_addr: u32, idx: usize, dx: i32, dy: i32) -> Option<(u32,u32)> {
    let (x0, y0) = LUMA_IDX_TO_XY[idx];
    let x1 = x0 as i32 + dx;
    let y1 = y0 as i32 + dy;
    None
}

fn res_block(bs: &mut BitStream, s: u32, e: u32, n_coeff: usize) -> Vec<u32> {
    let coeff_level = vec![0; n_coeff];
    // let coeff_token = ce(v)
    // A: -1, 0; B: 0, -1
    // get the x, y of the luma block idx
    coeff_level
}
*/

#[derive(Debug)]
enum ParsedNal {
    Sps(Sps),
    Pps(Pps),
    Sei,
    Slice(Slice),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bitstream() {
        let buf = &[0b0101_0101, 0b0000_1111];
        let r = BitStream::new(buf);
        let bits: Vec<u8> = r.collect();
        assert_eq!(bits, &[0,1,0,1,0,1,0,1, 0,0,0,0,1,1,1,1]);
    }
    #[test]
    fn test_parse_ue_golomb() {
        fn check(input: &[u8], output: u32, len: usize) -> () {
            let mut bs = BitStream::new(input);
            assert_eq!(bs.ue(), output);
            assert_eq!(bs.idx, len);
        }
        check(&[0b1101_0101], 0, 1);
        check(&[0b0101_0101], 1, 3);
        check(&[0b0111_0101], 2, 3);
        check(&[0b0011_0101], 5, 5);
        check(&[0b0001_0011], 8, 7);
    }
}

struct CodecCtx {
    sps: Option<Sps>,
    pps: Option<Pps>,
}

impl CodecCtx {
    fn new() -> CodecCtx {
        CodecCtx { 
            sps: None,
            pps: None,
        }
    }
    fn parse_nal(&self, nal: &Nal) -> ParsedNal {
        let mut bs = BitStream::new(&nal.rbsp[..]);
        if nal.nal_unit_type == 5 {
            assert!(nal.nal_ref_idc != 0);
        }
        match nal.nal_unit_type {
            1 => ParsedNal::Slice(Slice::parse(false, &mut bs, &self)),
            5 => ParsedNal::Slice(Slice::parse(true, &mut bs, &self)),
            6 => ParsedNal::Sei,
            7 => ParsedNal::Sps(Sps::parse(&mut bs)),
            8 => ParsedNal::Pps(Pps::parse(&mut bs)),
            _ => panic!("unknown nal type: {}", nal.nal_unit_type),
        }
    }
    fn update(&mut self, pnal: &ParsedNal) {
        match pnal {
            ParsedNal::Sps(sps) => self.sps = Some(sps.clone()),
            ParsedNal::Pps(pps) => self.pps = Some(pps.clone()),
            _ => (),
        }
    }
}

fn main() -> io::Result<()> {
    println!("starting tv264");

    let stdin = io::stdin();
    let mut ctx = CodecCtx::new();
    for nal in NalReader::new(stdin.lock()) {
        println!("got nal: {:?}", nal);
        let parsed_nal = ctx.parse_nal(&nal);
        println!("got parsed_nal: {:?}", parsed_nal);
        ctx.update(&parsed_nal);
    }

    Ok(())
}
