// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A matmul whose M/K/N are not multiples of its tile must not write past `C`.
//!
//! `sgemm_simd_4x4` computes a 32×32 output tile per threadgroup and the
//! dispatch rounds the grid up (`div_ceil(32)`), so at `m % 32 != 0` the
//! trailing tile covers rows that do not exist. `simdgroup_store` has no mask:
//! unguarded, those rows went to `C + m*n ..`, which in a planned arena is the
//! *next tensor*. `pick_sgemm` gates the `Simd4x4` variant to 32-alignment for
//! exactly this reason — but the `Mps` variant's MSL stand-in
//! (`dispatch_sgemm_variant`, taken whenever an operand lives in the separate
//! weight MTLBuffer and the single-buffer MPS helpers cannot encode it) runs the
//! same kernel at whatever shape the model has, and `Mps` is unconditionally
//! shape-eligible.
//!
//! Found via nomic-embed-text on Metal: the share-driven weight split put the
//! encoder's Linear weights in the weight buffer, every prefill projection
//! (m=16, k=768, n=2304) took the stand-in, and its 16 phantom tile rows landed
//! on the embedding `Gather`'s output — which read back as zeros, so every
//! embedding the daemon produced was wrong while each op looked fine alone.

#![cfg(target_os = "macos")]

use rlx_metal::blas::metal_sgemm_bufs;
use rlx_metal::device::metal_device;
use std::sync::Mutex;

static PIN_MUTEX: Mutex<()> = Mutex::new(());

/// Bytes of sentinel kept behind `C`. One full 32-row tile is the most the
/// kernel can overrun by, so this covers the whole blast radius.
const GUARD_ROWS: usize = 32;
const SENTINEL: f32 = 1234.5;

fn host_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// `C = A·B` through the real dispatcher, with A, B, C and a sentinel guard
/// packed back to back in one buffer — the layout a planned arena produces.
fn run_case(m: usize, k: usize, n: usize) {
    let dev = metal_device().expect("Metal device");

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 31) as f32 - 15.0) / 64.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 23) as f32 - 11.0) / 64.0)
        .collect();
    let want = host_matmul(&a, &b, m, k, n);

    // 256-byte aligned sections, so every `setBuffer:offset:` is legal.
    let align = |x: usize| (x + 255) & !255;
    let a_off = 0usize;
    let b_off = align(a_off + a.len() * 4);
    let c_off = align(b_off + b.len() * 4);
    let guard_off = c_off + m * n * 4; // deliberately NOT padded: packed arena
    let total = guard_off + GUARD_ROWS * n * 4;

    let buf = dev.alloc_shared(total);
    unsafe {
        let base = buf.contents() as *mut u8;
        std::ptr::copy_nonoverlapping(a.as_ptr(), base.add(a_off) as *mut f32, a.len());
        std::ptr::copy_nonoverlapping(b.as_ptr(), base.add(b_off) as *mut f32, b.len());
        let guard = std::slice::from_raw_parts_mut(base.add(guard_off) as *mut f32, GUARD_ROWS * n);
        guard.fill(SENTINEL);
    }

    let cmd = dev.queue.new_command_buffer();
    {
        let enc = cmd.compute_command_encoder();
        metal_sgemm_bufs(enc, &buf, a_off, &buf, b_off, &buf, c_off, m, k, n);
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();

    let (got, guard) = unsafe {
        let base = buf.contents() as *const u8;
        (
            std::slice::from_raw_parts(base.add(c_off) as *const f32, m * n).to_vec(),
            std::slice::from_raw_parts(base.add(guard_off) as *const f32, GUARD_ROWS * n).to_vec(),
        )
    };

    // The overrun is the regression: anything written behind C is a tensor the
    // memory planner handed to somebody else.
    let touched = guard.iter().position(|v| *v != SENTINEL);
    assert!(
        touched.is_none(),
        "m={m} k={k} n={n}: wrote {} floats past the end of C (first at +{}, value {:?})",
        guard.iter().filter(|v| **v != SENTINEL).count(),
        touched.unwrap(),
        touched.map(|i| guard[i]),
    );

    // A ragged K/N also *mis-reads* A/B at the edge, so check the values too.
    // Apple's simdgroup f32 MACs accumulate at reduced precision, hence the
    // loose tolerance — this is a correctness check, not a numerics one.
    let bad = got
        .iter()
        .zip(&want)
        .position(|(g, w)| (g - w).abs() > 1e-2 * w.abs().max(1.0));
    assert!(
        bad.is_none(),
        "m={m} k={k} n={n}: wrong at {:?} ({:?} vs {:?})",
        bad,
        bad.map(|i| got[i]),
        bad.map(|i| want[i]),
    );
}

#[test]
fn ragged_shapes_do_not_write_past_c() {
    let _pin = PIN_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    // Pin the variant whose MSL stand-in is the 32×32 tile kernel, so the test
    // exercises it on every machine rather than whatever the cost model or the
    // tuning cache happens to pick here.
    rlx_ir::env::set("RLX_METAL_SGEMM_VARIANT", "mps");

    // m=16/k=768/n=2304: nomic-embed-text's prefill projection, the shape that
    // actually corrupted the embedding gather.
    run_case(16, 768, 2304);
    // Ragged in the other two dims as well.
    run_case(16, 760, 2304);
    run_case(40, 768, 2280);
    run_case(8, 72, 40);

    rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
    rlx_metal::device::drain_command_queue();
}

/// The aligned path has to keep working — the mask must not cost correctness
/// where the kernel was already right.
#[test]
fn aligned_shapes_still_match_the_host() {
    let _pin = PIN_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    rlx_ir::env::set("RLX_METAL_SGEMM_VARIANT", "mps");
    run_case(32, 768, 2304);
    run_case(64, 64, 64);
    rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
    rlx_metal::device::drain_command_queue();
}
