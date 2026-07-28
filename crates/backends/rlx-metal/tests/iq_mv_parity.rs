// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fused IQ4_NL / IQ2_XXS GEMV parity vs CPU reference.

#![cfg(target_os = "macos")]

use metal::{Buffer, Device, MTLResourceOptions, MTLSize};
use rlx_ir::quant::QuantScheme;
use rlx_metal::kernels::kernels;

fn run_fused_mv(
    pipeline: &metal::ComputePipelineState,
    scheme: QuantScheme,
    x: &[f32],
    packed: &[u8],
    k: usize,
    n: usize,
    iq_lut: Option<&Buffer>,
) {
    let device = Device::system_default().expect("no Metal device");
    let cmd_q = device.new_command_queue();

    let x_bytes = x.len() * 4;
    let w_bytes = packed.len();
    let out_bytes = n * 4;
    let arena_bytes = (x_bytes + w_bytes + out_bytes).div_ceil(16) * 16;
    let arena: Buffer =
        device.new_buffer(arena_bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        let p = arena.contents() as *mut u8;
        std::ptr::write_bytes(p, 0, arena_bytes);
        std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, p, x_bytes);
        std::ptr::copy_nonoverlapping(packed.as_ptr(), p.add(x_bytes), w_bytes);
    }

    let x_u = 0u64;
    let w_u = x_bytes as u64;
    let dst_u = (x_bytes + w_bytes) as u64;
    let k_u = k as u32;
    let n_u = n as u32;

    let cmd_buf = cmd_q.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&arena), 0);
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    enc.set_bytes(3, 8, &dst_u as *const u64 as *const _);
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    if let Some(lut) = iq_lut {
        enc.set_buffer(6, Some(lut), 0);
    }
    let grid = MTLSize {
        width: n as u64,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256.min(n) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let mut out = vec![0.0f32; n];
    unsafe {
        let src = (arena.contents() as *const u8).add(x_bytes + w_bytes) as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }

    let mut cpu_out = vec![0f32; n];
    rlx_cpu::gguf_matmul::gguf_matmul_bt(x, packed, &mut cpu_out, 1, k, n, scheme);
    let tol = if matches!(
        scheme,
        QuantScheme::GgufIQ3XXS
            | QuantScheme::GgufIQ3S
            | QuantScheme::GgufIQ1S
            | QuantScheme::GgufIQ1M
    ) {
        5e-3
    } else {
        1e-3
    };
    for j in 0..n {
        assert!(
            (out[j] - cpu_out[j]).abs() <= tol,
            "row {j}: fused={} ref={}",
            out[j],
            cpu_out[j]
        );
    }
}

#[test]
fn iq4_nl_mv_matches_cpu_reference() {
    let k = 64usize;
    let n = 8usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.07).cos()).collect();
    let w_row: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.11).sin()).collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::IQ4NL).unwrap();
    run_fused_mv(
        &kernels().iq4_nl_mv_f32,
        QuantScheme::GgufIQ4NL,
        &x,
        &packed,
        k,
        n,
        None,
    );
}

#[test]
fn iq2_xxs_mv_matches_cpu_reference() {
    let k = 256usize;
    let n = 4usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.03).sin()).collect();
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.02).cos() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::IQ2XXS).unwrap();
    let kerns = kernels();
    run_fused_mv(
        &kerns.iq2_xxs_mv_f32,
        QuantScheme::GgufIQ2XXS,
        &x,
        &packed,
        k,
        n,
        Some(kerns.iq_grid_buffer()),
    );
}

#[test]
fn iq2_xs_mv_matches_cpu_reference() {
    let k = 256usize;
    let n = 4usize;
    let x: Vec<f32> = (0..k).map(|j| (j as f32 * 0.03).sin()).collect();
    let w_row: Vec<f32> = (0..k * n)
        .map(|j| ((j as f32) * 0.02).cos() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::IQ2XS).unwrap();
    let kerns = kernels();
    run_fused_mv(
        &kerns.iq2_xs_mv_f32,
        QuantScheme::GgufIQ2XS,
        &x,
        &packed,
        k,
        n,
        Some(kerns.iq_grid_buffer()),
    );
}

#[test]
fn iq3_xxs_mv_matches_cpu_reference() {
    let k = 256usize;
    let n = 4usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.025).cos()).collect();
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.018).sin() * 0.4)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::IQ3XXS).unwrap();
    let kerns = kernels();
    run_fused_mv(
        &kerns.iq3_xxs_mv_f32,
        QuantScheme::GgufIQ3XXS,
        &x,
        &packed,
        k,
        n,
        Some(kerns.iq_grid_buffer()),
    );
}

#[test]
fn iq2_s_mv_matches_cpu_reference() {
    let k = 256usize;
    let n = 4usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.031).sin()).collect();
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.019).cos() * 0.45)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::IQ2S).unwrap();
    let kerns = kernels();
    run_fused_mv(
        &kerns.iq2_s_mv_f32,
        QuantScheme::GgufIQ2S,
        &x,
        &packed,
        k,
        n,
        Some(kerns.iq_grid_buffer()),
    );
}

#[test]
fn iq3_s_mv_matches_cpu_reference() {
    let k = 256usize;
    let n = 4usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.027).cos()).collect();
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.016).sin() * 0.42)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::IQ3S).unwrap();
    let kerns = kernels();
    run_fused_mv(
        &kerns.iq3_s_mv_f32,
        QuantScheme::GgufIQ3S,
        &x,
        &packed,
        k,
        n,
        Some(kerns.iq_grid_buffer()),
    );
}

#[test]
fn iq1_s_mv_matches_cpu_reference() {
    let k = 256usize;
    let n = 4usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.029).sin()).collect();
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.014).cos() * 0.35)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::IQ1S).unwrap();
    let kerns = kernels();
    run_fused_mv(
        &kerns.iq1_s_mv_f32,
        QuantScheme::GgufIQ1S,
        &x,
        &packed,
        k,
        n,
        Some(kerns.iq_grid_buffer()),
    );
}

#[test]
fn iq1_m_mv_matches_cpu_reference() {
    let k = 256usize;
    let n = 4usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.028).cos()).collect();
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.015).sin() * 0.38)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::IQ1M).unwrap();
    let kerns = kernels();
    run_fused_mv(
        &kerns.iq1_m_mv_f32,
        QuantScheme::GgufIQ1M,
        &x,
        &packed,
        k,
        n,
        Some(kerns.iq_grid_buffer()),
    );
}
