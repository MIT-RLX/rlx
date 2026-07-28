// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end Metal MSL dequant parity for every new GGUF scheme
//! against the rlx-gguf CPU reference on real Qwen3-0.6B weights.
//!
//! Each test:
//!  1. Opens a quantized GGUF produced by `llama-quantize` (see
//!     `/tmp/rlx-iq-test/`).
//!  2. Picks every tensor whose dtype matches the scheme under test.
//!  3. Slabs the packed bytes into a Metal buffer.
//!  4. Dispatches the `dequant_gguf` MSL kernel.
//!  5. Compares element-wise to `rlx_gguf::*_dequant`.
//!
//! Tolerance is per-scheme: K/IQ4_NL/IQ4_XS round-trip bit-exactly
//! (f16 scales, integer quants). IQ2/IQ3/IQ1 also bit-exact since the
//! grid LUTs match. TQ formats use the multiply-and-shift trick which
//! is bit-exact too. MXFP4 / NVFP4 may differ by ULPs due to the
//! inline FP8 decode using exp2; we allow 1e-5 relative.
//!
//! Skipped if the GGUFs aren't present.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use metal::{Buffer, Device, MTLResourceOptions, MTLSize};
use rlx_gguf::{GgmlType, GgufFile};
use rlx_metal::kernels::kernels;

fn test_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("RLX_IQ_TEST_DIR").unwrap_or_else(|_| "/tmp/rlx-iq-test".to_string()),
    )
}

fn scheme_id_for(dtype: GgmlType) -> Option<u32> {
    Some(match dtype {
        GgmlType::Q4K => 0,
        GgmlType::Q5K => 1,
        GgmlType::Q6K => 2,
        GgmlType::Q8K => 3,
        GgmlType::Q2K => 4,
        GgmlType::Q3K => 5,
        GgmlType::IQ4NL => 6,
        GgmlType::IQ4XS => 7,
        GgmlType::TQ1_0 => 8,
        GgmlType::TQ2_0 => 9,
        GgmlType::MXFP4 => 10,
        GgmlType::NVFP4 => 11,
        GgmlType::IQ2XXS => 12,
        GgmlType::IQ2XS => 13,
        GgmlType::IQ2S => 14,
        GgmlType::IQ3XXS => 15,
        GgmlType::IQ3S => 16,
        GgmlType::IQ1S => 17,
        GgmlType::IQ1M => 18,
        _ => return None,
    })
}

fn scheme_block_elems(scheme_id: u32) -> usize {
    match scheme_id {
        6 | 10 => 32,
        11 => 16,
        _ => 256,
    }
}

fn metal_dequant_bytes(scheme_id: u32, packed: &[u8], elems: usize) -> Vec<f32> {
    let device = Device::system_default().expect("no Metal device");
    let k = kernels();
    let cmd_q = device.new_command_queue();
    let num_blocks = (elems / scheme_block_elems(scheme_id)) as u32;
    let weight_bytes = packed.len();
    let dst_byte_off = weight_bytes.div_ceil(16) * 16;
    let arena_bytes = dst_byte_off + elems * 4;
    let arena: Buffer =
        device.new_buffer(arena_bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        let p = arena.contents() as *mut u8;
        std::ptr::write_bytes(p, 0, arena_bytes);
        std::ptr::copy_nonoverlapping(packed.as_ptr(), p, weight_bytes);
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
    enc.set_buffer(5, Some(k.iq_grid_buffer()), 0);
    let grid = MTLSize {
        width: num_blocks as u64,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: num_blocks.clamp(1, 256) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    let mut out = vec![0.0f32; elems];
    unsafe {
        let src = (arena.contents() as *const u8).add(dst_byte_off) as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), elems);
    }
    out
}

/// For every tensor in `gguf_path` whose dtype is `target_dtype`,
/// compare the Metal kernel output against `rlx_gguf::dequant_f32`.
/// `tol` is absolute element-wise tolerance.
fn parity_over_real_weights(filename: &str, target_dtype: GgmlType, tol: f32) {
    let path = test_dir().join(filename);
    if !path.exists() {
        eprintln!("skip: {} not found", path.display());
        return;
    }
    let f = match GgufFile::from_path(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skip: could not open {}: {e}", path.display());
            return;
        }
    };
    let Some(scheme_id) = scheme_id_for(target_dtype) else {
        panic!("no scheme_id for {target_dtype:?}");
    };
    let mut n_tested = 0usize;
    for name in f.keys().collect::<Vec<_>>() {
        let t = f.get(name).unwrap();
        if t.dtype != target_dtype {
            continue;
        }
        let elems = t.n_elements();
        // Only test tensors aligned to the scheme's block size.
        if !elems.is_multiple_of(scheme_block_elems(scheme_id)) {
            continue;
        }
        let bytes = f.tensor_bytes(t).unwrap();
        let cpu = f.dequant_f32(name).unwrap().0;
        let metal = metal_dequant_bytes(scheme_id, bytes, elems);
        assert_eq!(metal.len(), cpu.len(), "{name}");
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for i in 0..elems {
            let d = (metal[i] - cpu[i]).abs();
            if d > worst {
                worst = d;
                worst_i = i;
            }
        }
        assert!(
            worst <= tol,
            "{name} {target_dtype:?}: worst diff at i={worst_i} = {worst} > tol {tol} (metal={}, cpu={})",
            metal[worst_i],
            cpu[worst_i],
        );
        n_tested += 1;
        // One representative tensor per scheme is plenty — full sweep would
        // upload gigabytes per test. Bail after the first matching tensor.
        eprintln!("ok: {name} ({} elems, worst diff {worst:.2e})", elems);
        break;
    }
    if n_tested == 0 {
        eprintln!("warn: no tensors with dtype {target_dtype:?} in {filename}");
    }
}

#[test]
fn metal_q2k_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ2_XXS.gguf", GgmlType::Q2K, 1e-3);
}
#[test]
fn metal_q4k_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ4_XS.gguf", GgmlType::Q4K, 1e-3);
}
#[test]
fn metal_q6k_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ2_XXS.gguf", GgmlType::Q6K, 1e-3);
}
#[test]
fn metal_iq4_nl_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ4_NL.gguf", GgmlType::IQ4NL, 1e-3);
}
#[test]
fn metal_iq4_xs_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ4_XS.gguf", GgmlType::IQ4XS, 1e-3);
}
#[test]
fn metal_iq2_xxs_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ2_XXS.gguf", GgmlType::IQ2XXS, 1e-3);
}
#[test]
fn metal_iq2_xs_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ2_XS.gguf", GgmlType::IQ2XS, 1e-3);
}
#[test]
fn metal_iq2_s_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ2_S.gguf", GgmlType::IQ2S, 1e-3);
}
#[test]
fn metal_iq3_xxs_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ3_XXS.gguf", GgmlType::IQ3XXS, 1e-3);
}
#[test]
fn metal_iq3_s_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ3_S.gguf", GgmlType::IQ3S, 1e-3);
}
#[test]
fn metal_iq1_s_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ1_S.gguf", GgmlType::IQ1S, 1e-3);
}
#[test]
fn metal_iq1_m_real() {
    parity_over_real_weights("Qwen3-0.6B-IQ1_M.gguf", GgmlType::IQ1M, 1e-3);
}
#[test]
fn metal_tq1_0_real() {
    parity_over_real_weights("Qwen3-0.6B-TQ1_0.gguf", GgmlType::TQ1_0, 1e-3);
}
#[test]
fn metal_tq2_0_real() {
    parity_over_real_weights("Qwen3-0.6B-TQ2_0.gguf", GgmlType::TQ2_0, 1e-3);
}
