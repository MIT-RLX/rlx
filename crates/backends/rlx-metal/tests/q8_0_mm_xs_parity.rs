// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `q8_0_mm_f32_xs` (new Q8_0 GEMM) against an independent f64 CPU reference,
//! with `q8_0_mv_f32` (the CPU-validated Q8_0 GEMV) checked alongside it.
//!
//! Q8_0 dequant is trivial (`d * int8`), so what this really guards is the
//! block/tile mismatch: blocks are 32 elements but the staged tile is 256, so
//! the kernel walks 8 blocks per tile and must map each block's `lp = sb*32`
//! offset into tile-local space. An off-by-one there reads the right VALUES from
//! the wrong k positions — which still produces plausible finite output.
//! Signedness is the other trap: `qs` is int8, and reading it unsigned would
//! shift half the weights by 256.
//!
//! Comparing the two KERNELS to each other does not work: these dot products
//! cancel heavily (|sum| can be ~400x smaller than sum|term|), so two valid f32
//! summation orders legitimately differ by ~1e-3 relative. The tolerance here is
//! therefore an absolute f32 error budget derived from the reference's own
//! sum-of-magnitudes, which stays tight in ULP terms; both kernels must fit it,
//! which pins the dequant math rather than just their agreement.

#![cfg(target_os = "macos")]

use rlx_metal::kernels::kernels;
use rlx_metal::mtl::{Buffer, Device, MTLResourceOptions, MTLSize};

/// Deterministic 34-byte Q8_0 block: d(f16) | qs[32] as int8. The fill spans
/// the full signed range so a kernel reading `qs` as unsigned diverges.
fn q8_0_block(b: u32) -> Vec<u8> {
    let mut blk = vec![0u8; 34];
    blk[0..2].copy_from_slice(&half::f16::from_f32(0.02 * (b as f32 % 7.0 + 1.0)).to_le_bytes());
    for (i, byte) in blk[2..].iter_mut().enumerate() {
        *byte = ((i as u32)
            .wrapping_mul(37)
            .wrapping_add(b.wrapping_mul(101))
            & 0xff) as u8;
    }
    blk
}

/// Exact dot product plus the sum of term magnitudes, which sets the f32 error
/// budget: heavy cancellation means the absolute error scales with sum|term|,
/// not with |result|. Dequant is written straight from the format description
/// (`d * int8`), deliberately not shared with the kernels.
fn reference_dot(x: &[f32], packed_col: &[u8], k: usize) -> (f64, f64) {
    let mut sum = 0f64;
    let mut mag = 0f64;
    for b in 0..k / 32 {
        let blk = &packed_col[b * 34..(b + 1) * 34];
        let d = f64::from(half::f16::from_le_bytes([blk[0], blk[1]]).to_f32());
        for l in 0..32usize {
            let q = f64::from(blk[2 + l] as i8);
            let t = f64::from(x[b * 32 + l]) * d * q;
            sum += t;
            mag += t.abs();
        }
    }
    (sum, mag)
}

struct Harness {
    device: Device,
    cmd_q: rlx_metal::mtl::CommandQueue,
}

impl Harness {
    fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let cmd_q = device.new_command_queue();
        Self { device, cmd_q }
    }

    /// `mm=true` → `q8_0_mm_f32_xs` over all `m` rows at once.
    /// `mm=false` → `q8_0_mv_f32` over a single row (`m` must be 1).
    fn run(&self, x: &[f32], packed: &[u8], m: usize, k: usize, n: usize, mm: bool) -> Vec<f32> {
        let x_bytes = x.len() * 4;
        let w_bytes = packed.len();
        let out_bytes = m * n * 4;
        let arena_bytes = (x_bytes + w_bytes + out_bytes).div_ceil(256) * 256;
        let arena: Buffer = self
            .device
            .new_buffer(arena_bytes as u64, MTLResourceOptions::StorageModeShared);
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

        let ks = kernels();
        let cmd_buf = self.cmd_q.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_buffer(0, Some(&arena), 0);
        enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
        enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
        enc.set_bytes(3, 8, &dst_u as *const u64 as *const _);
        if mm {
            enc.set_compute_pipeline_state(&ks.q8_0_mm_f32_xs);
            enc.set_bytes(4, 4, &m_u as *const u32 as *const _);
            enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
            enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
            const TM: u64 = 8;
            const TG: u64 = 64;
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
            assert_eq!(m, 1, "GEMV reference takes one row at a time");
            enc.set_compute_pipeline_state(&ks.q8_0_mv_f32);
            enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
            enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
            enc.dispatch_threads(
                MTLSize {
                    width: n as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: (n as u64).min(64),
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
}

fn parity(m: usize, k: usize, n: usize) {
    let nblocks = k / 32;
    let mut packed = Vec::with_capacity(n * nblocks * 34);
    for col in 0..n {
        for b in 0..nblocks {
            packed.extend_from_slice(&q8_0_block((col * nblocks + b) as u32));
        }
    }
    // Distinct per-row activations — identical rows would hide a row-indexing
    // bug in the staged tile.
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) / 64.0 + (i / k) as f32 * 0.01)
        .collect();

    let h = Harness::new();
    let got = h.run(&x, &packed, m, k, n, true);

    let mut any_nonzero = false;
    for r in 0..m {
        let row = &x[r * k..(r + 1) * k];
        let gemv = h.run(row, &packed, 1, k, n, false);
        for c in 0..n {
            let col = &packed[c * (k / 32) * 34..(c + 1) * (k / 32) * 34];
            let (want, mag) = reference_dot(row, col, k);
            any_nonzero |= want.abs() > 1e-3;
            // f32 accumulation over `k` terms: sqrt(k) * eps * sum|term|, with
            // 8x headroom. Tight in ULP terms — a single mis-set 5th bit is a
            // 16-step weight error, orders of magnitude outside this.
            let tol = 8.0 * (k as f64).sqrt() * f64::from(f32::EPSILON) * mag.max(1.0);
            let g = f64::from(got[r * n + c]);
            let v = f64::from(gemv[c]);
            assert!(
                (want - g).abs() <= tol,
                "GEMM off reference: m={m} k={k} row={r} col={c} want={want} got={g} tol={tol}"
            );
            assert!(
                (want - v).abs() <= tol,
                "GEMV off reference: m={m} k={k} row={r} col={c} want={want} got={v} tol={tol}"
            );
        }
    }
    assert!(
        any_nonzero,
        "reference output is all ~zero for m={m} k={k} n={n} — test proves nothing"
    );
}

#[test]
fn q8_0_mm_matches_gemv_full_tile() {
    parity(8, 512, 64);
}

#[test]
fn q8_0_mm_matches_gemv_partial_row_tile() {
    // 8-row tile plus a 4-row remainder; rows 12..15 clamp to row 11 while
    // staging and must be dropped at the store.
    parity(12, 512, 64);
}

#[test]
fn q8_0_mm_matches_gemv_single_row() {
    parity(1, 512, 64);
}

#[test]
fn q8_0_mm_matches_gemv_ragged_n() {
    // Threadgroups with mostly-clamped columns that must still reach every
    // barrier.
    parity(5, 256, 70);
    parity(5, 256, 6);
}

#[test]
fn q8_0_mm_matches_gemv_many_k_blocks() {
    // 16 super-blocks: a missing barrier before the next tile overwrite would
    // corrupt later blocks, not the first.
    parity(4, 4096, 64);
}

#[test]
fn q8_0_mm_matches_gemv_lm_head_shape() {
    // Tall-and-wide like the real LM head this kernel was added for.
    parity(10, 1024, 512);
}
