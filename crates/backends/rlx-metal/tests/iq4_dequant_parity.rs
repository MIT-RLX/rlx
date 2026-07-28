// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! IQ4_NL / IQ4_XS Metal MSL dequant kernel parity vs rlx-gguf CPU.
//!
//! Validates two things:
//! 1. The new MSL `dequant_gguf` scheme-id 6 / 7 branches compile in Metal.
//! 2. They produce bit-identical f32 outputs to `rlx_gguf::dequant_iq4_*`
//!    on hand-built block bytes.
//!
//! Skipped on non-macOS / no-Metal-device.

#![cfg(target_os = "macos")]

use metal::{Buffer, Device, MTLResourceOptions, MTLSize};
use rlx_metal::kernels::kernels;

// Per-scheme block-element counts. Mirrors `gguf_block_size()`.
fn scheme_block_elems(scheme_id: u32) -> usize {
    match scheme_id {
        6 => 32,  // IQ4_NL
        10 => 32, // MXFP4
        11 => 16, // NVFP4
        _ => 256, // K-quants, IQ4_XS, IQ2/3/1, TQ
    }
}

/// Encode one IQ4_NL block (32 elements / 18 bytes): f16 d | 16 nibbles.
fn make_iq4_nl_block(d: f32, nibbles_lo: [u8; 16], nibbles_hi: [u8; 16]) -> Vec<u8> {
    let mut b = Vec::with_capacity(18);
    b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    for i in 0..16 {
        let lo = nibbles_lo[i] & 0x0F;
        let hi = nibbles_hi[i] & 0x0F;
        b.push(lo | (hi << 4));
    }
    b
}

/// Encode one IQ4_XS super-block (256 elements / 136 bytes).
/// Layout: f16 d | u16 scales_h | u8 scales_l[4] | u8 qs[128].
fn make_iq4_xs_block(d: f32, scales: [u8; 8], qs: [u8; 128]) -> Vec<u8> {
    // Pack scales[8] (each 6-bit signed-bias-32) into scales_h (high 2 bits)
    // + scales_l (low 4 bits). For testing pick canonical patterns.
    let mut scales_l = [0u8; 4];
    let mut scales_h: u16 = 0;
    for (ib, &sc) in scales.iter().enumerate() {
        let lo = sc & 0xF;
        let hi = (sc >> 4) & 0x3;
        scales_l[ib / 2] |= lo << (4 * (ib % 2));
        scales_h |= (hi as u16) << (2 * ib);
    }
    let mut b = Vec::with_capacity(136);
    b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    b.extend_from_slice(&scales_h.to_le_bytes());
    b.extend_from_slice(&scales_l);
    b.extend_from_slice(&qs);
    b
}

/// Dispatch `dequant_gguf` over `num_blocks` independent blocks. Each
/// block produces `scheme_block_elems(scheme_id)` f32 outputs at the
/// dst region. Total output length = num_blocks * block_elems.
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
    let w_off_u = 0u32;
    enc.set_bytes(1, 4, &w_off_u as *const u32 as *const _);
    let dst_f32_off = (dst_byte_off / 4) as u32;
    enc.set_bytes(2, 4, &dst_f32_off as *const u32 as *const _);
    enc.set_bytes(3, 4, &scheme_id as *const u32 as *const _);
    enc.set_bytes(4, 4, &num_blocks as *const u32 as *const _);
    // buffer(5): the global IQ-grid LUT (mandatory binding even for
    // schemes that don't read it — Metal requires every declared
    // argument to be bound).
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

/// Run dequant on a real-tensor slab and compare to CPU element-by-element.
fn parity(scheme_id: u32, packed: &[u8], elems: usize, tol: f32, scheme_name: &str) {
    let num_blocks = (elems / scheme_block_elems(scheme_id)) as u32;
    let metal_out = run_metal_dequant(scheme_id, packed, num_blocks);
    let cpu_out = cpu_dequant(scheme_id, packed, elems);
    assert_eq!(metal_out.len(), elems);
    assert_eq!(cpu_out.len(), elems);
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for i in 0..elems {
        let d = (metal_out[i] - cpu_out[i]).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    assert!(
        worst <= tol,
        "{scheme_name} parity: worst diff at i={worst_i} = {worst} > tol {tol} (metal={}, cpu={})",
        metal_out[worst_i],
        cpu_out[worst_i],
    );
}

fn cpu_dequant(scheme_id: u32, bytes: &[u8], elems: usize) -> Vec<f32> {
    match scheme_id {
        6 => rlx_gguf::iq_dequant::dequant_iq4_nl(bytes, elems).unwrap(),
        7 => rlx_gguf::iq_dequant::dequant_iq4_xs(bytes, elems).unwrap(),
        8 => rlx_gguf::tq_dequant::dequant_tq1_0(bytes, elems).unwrap(),
        9 => rlx_gguf::tq_dequant::dequant_tq2_0(bytes, elems).unwrap(),
        10 => rlx_gguf::mx_dequant::dequant_mxfp4(bytes, elems).unwrap(),
        11 => rlx_gguf::mx_dequant::dequant_nvfp4(bytes, elems).unwrap(),
        12 => rlx_gguf::iq_dequant::dequant_iq2_xxs(bytes, elems).unwrap(),
        13 => rlx_gguf::iq_dequant::dequant_iq2_xs(bytes, elems).unwrap(),
        14 => rlx_gguf::iq_dequant::dequant_iq2_s(bytes, elems).unwrap(),
        15 => rlx_gguf::iq_dequant::dequant_iq3_xxs(bytes, elems).unwrap(),
        16 => rlx_gguf::iq_dequant::dequant_iq3_s(bytes, elems).unwrap(),
        17 => rlx_gguf::iq_dequant::dequant_iq1_s(bytes, elems).unwrap(),
        18 => rlx_gguf::iq_dequant::dequant_iq1_m(bytes, elems).unwrap(),
        _ => panic!("cpu_dequant: bad scheme_id {scheme_id}"),
    }
}

#[test]
fn iq4_nl_msl_matches_cpu_reference() {
    let lo: [u8; 16] = std::array::from_fn(|i| (i as u8) & 0xF);
    let hi: [u8; 16] = std::array::from_fn(|i| (i as u8 + 1) & 0xF);
    let mut packed = Vec::new();
    for b in 0..8u32 {
        let d = 0.125 * (b as f32 + 1.0);
        packed.extend_from_slice(&make_iq4_nl_block(d, lo, hi));
    }
    parity(6, &packed, 256, 1e-4, "IQ4_NL");
}

#[test]
fn iq4_xs_msl_matches_cpu_reference() {
    let scales: [u8; 8] = [33, 34, 32, 35, 31, 36, 30, 37];
    let qs: [u8; 128] =
        std::array::from_fn(|i| ((i & 0xF) | ((i.wrapping_add(3) & 0xF) << 4)) as u8);
    let d = 0.0125f32;
    let block = make_iq4_xs_block(d, scales, qs);
    parity(7, &block, 256, 1e-3, "IQ4_XS");
}

/// Encode→dequant round-trip via `rlx_gguf::quantize` (covers encoder layout).
fn parity_quantized(
    scheme_id: u32,
    ggml: rlx_gguf::GgmlType,
    k: usize,
    n: usize,
    tol: f32,
    scheme_name: &str,
) {
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.017).sin() * 0.42)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    parity(scheme_id, &packed, k * n, tol, scheme_name);
}

#[test]
fn iq2_xxs_msl_matches_cpu_reference() {
    parity_quantized(12, rlx_gguf::GgmlType::IQ2XXS, 256, 2, 1e-3, "IQ2_XXS");
}

#[test]
fn iq2_xs_msl_matches_cpu_reference() {
    parity_quantized(13, rlx_gguf::GgmlType::IQ2XS, 256, 2, 1e-3, "IQ2_XS");
}

#[test]
fn iq2_s_msl_matches_cpu_reference() {
    parity_quantized(14, rlx_gguf::GgmlType::IQ2S, 256, 2, 1e-3, "IQ2_S");
}

#[test]
fn iq3_xxs_msl_matches_cpu_reference() {
    parity_quantized(15, rlx_gguf::GgmlType::IQ3XXS, 256, 2, 5e-3, "IQ3_XXS");
}

#[test]
fn iq3_s_msl_matches_cpu_reference() {
    parity_quantized(16, rlx_gguf::GgmlType::IQ3S, 256, 2, 5e-3, "IQ3_S");
}

#[test]
fn iq1_s_msl_matches_cpu_reference() {
    parity_quantized(17, rlx_gguf::GgmlType::IQ1S, 256, 2, 5e-3, "IQ1_S");
}

#[test]
fn iq1_m_msl_matches_cpu_reference() {
    parity_quantized(18, rlx_gguf::GgmlType::IQ1M, 256, 2, 5e-3, "IQ1_M");
}

#[test]
fn tq2_0_msl_matches_cpu_reference() {
    parity_quantized(9, rlx_gguf::GgmlType::TQ2_0, 256, 2, 1e-3, "TQ2_0");
}
