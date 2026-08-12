// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Q4_0 / Q8_0 Metal MSL `dequant_gguf` parity vs rlx-gguf CPU reference.

#![cfg(target_os = "macos")]

use rlx_metal::kernels::kernels;
use rlx_metal::mtl::{Buffer, Device, MTLResourceOptions, MTLSize};

fn scheme_block_elems(scheme_id: u32) -> usize {
    match scheme_id {
        19..=23 => 32,
        24 => 128, // Q1_0
        _ => 256,
    }
}

fn run_metal_dequant(scheme_id: u32, block_bytes: &[u8], num_blocks: u32) -> Vec<f32> {
    let device = Device::system_default().expect("no Metal device");
    let k = kernels();
    let cmd_q = device.new_command_queue();
    let total_out_elems = num_blocks as usize * scheme_block_elems(scheme_id);
    let weight_bytes = block_bytes.len();
    let dst_byte_off = weight_bytes.div_ceil(16) * 16;
    let arena_bytes = dst_byte_off + total_out_elems * 4;
    let arena: Buffer =
        device.new_buffer(arena_bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        let p = arena.contents() as *mut u8;
        std::ptr::write_bytes(p, 0, arena_bytes);
        std::ptr::copy_nonoverlapping(block_bytes.as_ptr(), p, weight_bytes);
    }

    let cmd_buf = cmd_q.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&k.dequant_gguf);
    enc.set_buffer(0, Some(&arena), 0);
    let w_off_u = 0u64;
    enc.set_bytes(1, 8, &w_off_u as *const u64 as *const _);
    let dst_f32_off = (dst_byte_off / 4) as u64;
    enc.set_bytes(2, 8, &dst_f32_off as *const u64 as *const _);
    enc.set_bytes(3, 4, &scheme_id as *const u32 as *const _);
    enc.set_bytes(4, 4, &num_blocks as *const u32 as *const _);
    enc.set_buffer(5, Some(k.iq_grid_buffer()), 0);
    let grid = MTLSize {
        width: num_blocks as u64,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: num_blocks.clamp(1, 64) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let mut out = vec![0.0f32; total_out_elems];
    unsafe {
        let src = (arena.contents() as *const u8).add(dst_byte_off) as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), total_out_elems);
    }
    out
}

fn parity(scheme_id: u32, packed: &[u8], elems: usize, tol: f32, name: &str) {
    let num_blocks = (elems / scheme_block_elems(scheme_id)) as u32;
    let metal_out = run_metal_dequant(scheme_id, packed, num_blocks);
    let cpu_out = match scheme_id {
        0 => rlx_gguf::dequant_q4_k(packed, elems).unwrap(),
        1 => rlx_gguf::dequant_q5_k(packed, elems).unwrap(),
        2 => rlx_gguf::dequant_q6_k(packed, elems).unwrap(),
        5 => rlx_gguf::dequant_q3_k(packed, elems).unwrap(),
        19 => rlx_gguf::dequant_q4_0(packed, elems).unwrap(),
        20 => rlx_gguf::dequant_q8_0(packed, elems).unwrap(),
        21 => rlx_gguf::dequant_q4_1(packed, elems).unwrap(),
        22 => rlx_gguf::dequant_q5_0(packed, elems).unwrap(),
        23 => rlx_gguf::dequant_q5_1(packed, elems).unwrap(),
        24 => rlx_gguf::q1_dequant::dequant_q1_0(packed, elems).unwrap(),
        26 => rlx_gguf::fv5_dequant::dequant_fv5(packed, elems).unwrap(),
        27 => rlx_gguf::fv5_dequant::dequant_fv5b(packed, elems).unwrap(),
        _ => panic!("bad scheme_id {scheme_id}"),
    };
    assert_eq!(metal_out.len(), elems);
    let mut worst = 0.0f32;
    for i in 0..elems {
        worst = worst.max((metal_out[i] - cpu_out[i]).abs());
    }
    assert!(worst <= tol, "{name}: worst diff {worst} > {tol}");
}

#[test]
fn q8_0_msl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for b in 0..16u8 {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.05 * (b as f32 + 1.0)).to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            *q = ((i as i8).wrapping_mul(3).wrapping_add(b as i8)) as u8;
        }
        packed.extend_from_slice(&block);
    }
    parity(20, &packed, 512, 1e-5, "Q8_0");
}

#[test]
fn q3_k_msl_matches_cpu_reference() {
    // Regression guard for the Metal Q3_K kernel that previously stored only
    // 8 of the 16 six-bit sub-block scales (aux2/aux3 dropped), so the second
    // 128 elements of every super-block dequantized with garbage scales.
    // Vary magnitude per 16-elem sub-block so all 16 scales differ, forcing
    // scales[8..15] to matter.
    // Raw 110-byte blocks: hmask[0..32] | qs[32..96] | scales[96..108] | d[108..110].
    // Both paths decode the SAME bytes; varied scale bytes make all 16 six-bit
    // scales distinct so scales[8..15] must be right.
    let mut packed = Vec::new();
    for b in 0..8u32 {
        let mut block = vec![0u8; 110];
        for (i, byte) in block[..108].iter_mut().enumerate() {
            *byte = ((i as u32)
                .wrapping_mul(37)
                .wrapping_add(b.wrapping_mul(101))
                & 0xff) as u8;
        }
        block[108..110]
            .copy_from_slice(&half::f16::from_f32(0.03 * (b as f32 + 1.0)).to_le_bytes());
        packed.extend_from_slice(&block);
    }
    // Same packed bytes decoded by both paths → identical up to f16 scale
    // conversion; a mismatch means the MSL scale unpacking is wrong.
    parity(5, &packed, 8 * 256, 1e-3, "Q3_K");
}

// K-quants used alongside Q3_K in a Q3_K_S GGUF (Q4_K/Q5_K attention+ffn,
// Q6_K output.weight). Same class of Metal-vs-CPU dequant parity as Q3_K.
fn kquant_block(total: usize, d_off: usize, dmin_off: Option<usize>, b: u32) -> Vec<u8> {
    let mut blk = vec![0u8; total];
    for (i, byte) in blk.iter_mut().enumerate() {
        *byte = ((i as u32)
            .wrapping_mul(37)
            .wrapping_add(b.wrapping_mul(101))
            & 0xff) as u8;
    }
    blk[d_off..d_off + 2]
        .copy_from_slice(&half::f16::from_f32(0.03 * (b as f32 + 1.0)).to_le_bytes());
    if let Some(o) = dmin_off {
        blk[o..o + 2].copy_from_slice(&half::f16::from_f32(0.01 * (b as f32 + 1.0)).to_le_bytes());
    }
    blk
}

#[test]
fn q4_k_msl_matches_cpu_reference() {
    // Q4_K: d(f16)|dmin(f16)|scales[12]|qs[128] = 144 B / 256 elems.
    let mut packed = Vec::new();
    for b in 0..8u32 {
        packed.extend_from_slice(&kquant_block(144, 0, Some(2), b));
    }
    parity(0, &packed, 8 * 256, 1e-3, "Q4_K");
}

#[test]
fn q5_k_msl_matches_cpu_reference() {
    // Q5_K: d(f16)|dmin(f16)|scales[12]|qh[32]|qs[128] = 176 B / 256 elems.
    let mut packed = Vec::new();
    for b in 0..8u32 {
        packed.extend_from_slice(&kquant_block(176, 0, Some(2), b));
    }
    parity(1, &packed, 8 * 256, 1e-3, "Q5_K");
}

#[test]
fn q6_k_msl_matches_cpu_reference() {
    // Q6_K: ql[128]|qh[64]|scales[16]|d(f16) = 210 B / 256 elems (d at end).
    let mut packed = Vec::new();
    for b in 0..8u32 {
        packed.extend_from_slice(&kquant_block(210, 208, None, b));
    }
    parity(2, &packed, 8 * 256, 1e-3, "Q6_K");
}

#[test]
fn q4_0_msl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for b in 0..16u8 {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.08 * (b as f32 + 1.0)).to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            let lo = ((i + b as usize) % 16) as u8;
            let hi = ((i + b as usize + 3) % 16) as u8;
            *q = lo | (hi << 4);
        }
        packed.extend_from_slice(&block);
    }
    parity(19, &packed, 512, 1e-5, "Q4_0");
}

#[test]
fn q4_1_msl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for b in 0..16u8 {
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.08 * (b as f32 + 1.0)).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.01 * b as f32).to_le_bytes());
        for (i, q) in block[4..].iter_mut().enumerate() {
            let lo = ((i + b as usize) % 16) as u8;
            let hi = ((i + b as usize + 3) % 16) as u8;
            *q = lo | (hi << 4);
        }
        packed.extend_from_slice(&block);
    }
    parity(21, &packed, 512, 1e-4, "Q4_1");
}

#[test]
fn q5_0_msl_matches_cpu_reference() {
    let w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.017).cos()).collect();
    let packed = rlx_gguf::quantize::quantize_q5_0(&w).unwrap();
    parity(22, &packed, 512, 1e-4, "Q5_0");
}

#[test]
fn q5_1_msl_matches_cpu_reference() {
    let w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.019).sin()).collect();
    let packed = rlx_gguf::quantize::quantize_q5_1(&w).unwrap();
    parity(23, &packed, 512, 1e-4, "Q5_1");
}

#[test]
fn q1_0_msl_matches_cpu_reference() {
    // Q1_0 (prism-ml Bonsai-27B): f16 d + 16 sign bytes = 18 bytes / 128 elems.
    let mut packed = Vec::new();
    for b in 0..16u8 {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.1 + 0.05 * b as f32).to_le_bytes());
        for j in 0..128usize {
            // Deterministic sign pattern; exercises LSB-first bit ordering
            // across all 16 sign bytes.
            if !(j + b as usize).is_multiple_of(3) {
                block[2 + j / 8] |= 1 << (j % 8);
            }
        }
        packed.extend_from_slice(&block);
    }
    parity(24, &packed, 16 * 128, 1e-4, "Q1_0");
}

/// Pack one FV5 block (104 bytes) from 256 five-value codes in {-2,-1,0,1,2}.
fn pack_fv5_block(codes: &[i8], s_lo: f32, s_hi: f32) -> Vec<u8> {
    let mut b = vec![0u8; 104];
    b[0..4].copy_from_slice(&s_lo.to_le_bytes());
    b[4..8].copy_from_slice(&s_hi.to_le_bytes());
    for (j, &c) in codes.iter().enumerate() {
        let byte = j / 8;
        let bit = 1u8 << (j % 8);
        let (p, ng, hi) = match c {
            1 => (true, false, false),
            2 => (true, false, true),
            -1 => (false, true, false),
            -2 => (false, true, true),
            _ => (false, false, false),
        };
        if p {
            b[8 + byte] |= bit;
        }
        if ng {
            b[40 + byte] |= bit;
        }
        if hi {
            b[72 + byte] |= bit;
        }
    }
    b
}

/// Pack one FV5B block (260 bytes): one f32 scale + 256 int8 codes.
fn pack_fv5b_block(qs: &[i8], s: f32) -> Vec<u8> {
    let mut b = vec![0u8; 260];
    b[0..4].copy_from_slice(&s.to_le_bytes());
    for (i, &q) in qs.iter().enumerate() {
        b[4 + i] = q as u8;
    }
    b
}

#[test]
fn fv5_msl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for blk in 0..4usize {
        let codes: [i8; 256] = std::array::from_fn(|j| match (j + blk) % 5 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => -1,
            _ => -2,
        });
        packed.extend_from_slice(&pack_fv5_block(
            &codes,
            0.05 * (blk as f32 + 1.0),
            0.2 * (blk as f32 + 1.0),
        ));
    }
    parity(26, &packed, 1024, 1e-6, "FV5");
}

#[test]
fn fv5b_msl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for blk in 0..4usize {
        let qs: [i8; 256] =
            std::array::from_fn(|i| ((i as i32 * 7 + blk as i32) % 251 - 125) as i8);
        packed.extend_from_slice(&pack_fv5b_block(&qs, 0.03 * (blk as f32 + 1.0)));
    }
    parity(27, &packed, 1024, 1e-6, "FV5B");
}

#[test]
fn has_metal_dequant_kernel_q8_q4() {
    use rlx_ir::quant::QuantScheme;
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufQ8_0
    ));
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufQ4_0
    ));
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufQ4_1
    ));
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufQ5_0
    ));
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufFV5
    ));
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufFV5B
    ));
}
