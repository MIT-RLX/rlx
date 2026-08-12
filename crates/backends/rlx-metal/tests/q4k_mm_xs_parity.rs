// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `q4k_mm_f32_xs` (x tile staged in threadgroup memory) vs `q4k_mm_f32` (the
//! device-load kernel already validated against the CPU reference), on identical
//! packed bytes.
//!
//! The two kernels do the SAME arithmetic in the same order — `_xs` only changes
//! where activations are read from — so this is an exact-equality check, not an
//! approximate one. What it is really guarding is the cooperative staging: rows
//! past `m` and columns past `n` are clamped (they must still reach every
//! `threadgroup_barrier`), and a clamp that leaked into the store, an off-by-one
//! in the tile-local k offset, or a missing barrier would all show up here.

#![cfg(target_os = "macos")]

use rlx_metal::kernels::kernels;
use rlx_metal::mtl::{Buffer, Device, MTLResourceOptions, MTLSize};

/// Deterministic 144-byte Q4_K super-block: d|dmin|scales[12]|qs[128].
/// Byte-filled pseudo-randomly so all 8 six-bit scale/min pairs differ — a
/// kernel that mixed up scale indices would still pass on uniform data.
fn q4k_block(b: u32) -> Vec<u8> {
    let mut blk = vec![0u8; 144];
    for (i, byte) in blk.iter_mut().enumerate() {
        *byte = ((i as u32)
            .wrapping_mul(37)
            .wrapping_add(b.wrapping_mul(101))
            & 0xff) as u8;
    }
    blk[0..2].copy_from_slice(&half::f16::from_f32(0.03 * (b as f32 % 7.0 + 1.0)).to_le_bytes());
    blk[2..4].copy_from_slice(&half::f16::from_f32(0.01 * (b as f32 % 5.0 + 1.0)).to_le_bytes());
    blk
}

/// Runs one Q4_K GEMM pipeline over a shared arena and returns `dst` as `[m, n]`.
fn run_mm(
    pipeline: &rlx_metal::mtl::ComputePipelineState,
    x: &[f32],
    packed: &[u8],
    m: usize,
    k: usize,
    n: usize,
    staged: bool,
) -> Vec<f32> {
    let device = Device::system_default().expect("no Metal device");
    let cmd_q = device.new_command_queue();

    let x_bytes = x.len() * 4;
    let w_bytes = packed.len();
    let out_bytes = m * n * 4;
    let arena_bytes = (x_bytes + w_bytes + out_bytes).div_ceil(256) * 256;
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
    let (m_u, k_u, n_u) = (m as u32, k as u32, n as u32);

    let cmd_buf = cmd_q.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&arena), 0);
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    enc.set_bytes(3, 8, &dst_u as *const u64 as *const _);
    enc.set_bytes(4, 4, &m_u as *const u32 as *const _);
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);

    const TM: u64 = 8;
    const TG: u64 = 64;
    if staged {
        enc.dispatch_thread_groups(
            MTLSize {
                width: (n as u64).div_ceil(TG),
                height: (m as u64).div_ceil(TM),
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );
    } else {
        enc.dispatch_threads(
            MTLSize {
                width: n as u64,
                height: (m as u64).div_ceil(TM),
                depth: 1,
            },
            MTLSize {
                width: (n as u64).min(TG),
                height: 1,
                depth: 1,
            },
        );
    }
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let mut out = vec![0f32; m * n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            (arena.contents() as *const u8).add(dst_u as usize) as *const f32,
            out.as_mut_ptr(),
            m * n,
        );
    }
    out
}

fn parity(m: usize, k: usize, n: usize) {
    let nblocks = k / 256;
    let mut packed = Vec::with_capacity(n * nblocks * 144);
    for col in 0..n {
        for b in 0..nblocks {
            packed.extend_from_slice(&q4k_block((col * nblocks + b) as u32));
        }
    }
    // Distinct per-row activations: identical rows would hide a row-indexing bug
    // in the staged tile (every row would get the same answer either way).
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) / 64.0 + (i / k) as f32 * 0.01)
        .collect();

    let k4 = kernels();
    let want = run_mm(&k4.q4k_mm_f32, &x, &packed, m, k, n, false);
    let got = run_mm(&k4.q4k_mm_f32_xs, &x, &packed, m, k, n, true);

    assert!(
        want.iter().any(|v| v.abs() > 1e-3),
        "reference output is all ~zero for m={m} k={k} n={n} — test proves nothing"
    );
    // Identical operation order in both kernels, so this must be exact.
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "m={m} k={k} n={n} row={} col={} device={a} staged={b}",
            i / n,
            i % n
        );
    }
}

#[test]
fn xs_matches_device_full_tile() {
    parity(8, 512, 64);
}

#[test]
fn xs_matches_device_partial_row_tile() {
    // m=12: one full 8-row tile plus a 4-row remainder. Rows 12..15 clamp to row
    // 11 while staging; the store must drop them.
    parity(12, 512, 64);
}

#[test]
fn xs_matches_device_single_row() {
    parity(1, 512, 64);
}

#[test]
fn xs_matches_device_ragged_n() {
    // n=70 → the second threadgroup has 6 live columns and 58 clamped ones that
    // must still reach every barrier. n=6 → the ONLY group is mostly clamped.
    parity(5, 256, 70);
    parity(5, 256, 6);
}

#[test]
fn xs_matches_device_many_k_blocks() {
    // 32 super-blocks: the staging loop runs 32 times, so a barrier missing
    // before the next tile overwrite would corrupt later blocks, not the first.
    parity(4, 8192, 64);
}

#[test]
fn xs_matches_device_wide_n() {
    // Multiple full threadgroups in x, exercising the column base offset.
    parity(10, 512, 256);
}
