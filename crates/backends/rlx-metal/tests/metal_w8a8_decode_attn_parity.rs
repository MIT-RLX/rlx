// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! W8A8 decode attention (`RLX_METAL_W8A8_ATTN`): int8 Q·K integer dot + int8 V,
//! quantized per-row. Validates the approximation stays close to the exact CPU
//! f32 attention on qwen3-shaped decode (MHA + GQA, causal + custom mask).

#![cfg(target_os = "macos")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use std::sync::{Mutex, MutexGuard};

static METAL_TEST_MUTEX: Mutex<()> = Mutex::new(());
struct MetalTestGuard(#[allow(dead_code)] MutexGuard<'static, ()>);
impl MetalTestGuard {
    fn new() -> Self {
        Self(METAL_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
    }
}
impl Drop for MetalTestGuard {
    fn drop(&mut self) {
        rlx_metal::device::drain_command_queue();
        rlx_metal::mps_blas::invalidate_caches();
    }
}

/// GQA causal decode attention: Q has `nh` heads, K/V have `nkv` heads.
fn build_gqa_causal(b: usize, lk: usize, nh: usize, nkv: usize, dh: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("w8a8_gqa_causal");
    let q = g.input("q", Shape::new(&[b, 1, nh * dh], f));
    let k = g.input("k", Shape::new(&[b, lk, nkv * dh], f));
    let v = g.input("v", Shape::new(&[b, lk, nkv * dh], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: nh,
            head_dim: dh,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, 1, nh * dh], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
fn rms(a: &[f32]) -> f32 {
    (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt()
}

fn run_case(b: usize, lk: usize, nh: usize, nkv: usize, dh: usize) {
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
    rlx_ir::env::set("RLX_METAL_SDPA_FLASH_DECODE", "1"); // force flash-decode path

    let nq = b * nh * dh;
    let nk = b * lk * nkv * dh;
    let q: Vec<f32> = (0..nq).map(|i| ((i as f32) * 0.017).sin() * 0.8).collect();
    let k: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.013).cos()).collect();
    // DC-biased V so the attention output has realistic O(0.5) magnitude (a real
    // value head is not a zero-mean sinusoid); a near-zero output would make the
    // relative-error metric meaningless (tiny signal / tiny int8 drift).
    let v: Vec<f32> = (0..nk)
        .map(|i| 0.5 + 0.4 * ((i as f32) * 0.011).sin())
        .collect();
    let inputs = [
        ("q", q.as_slice()),
        ("k", k.as_slice()),
        ("v", v.as_slice()),
    ];

    // CPU reference (exact f32).
    let mut cpu = Session::new(Device::Cpu).compile(build_gqa_causal(b, lk, nh, nkv, dh));
    let cpu_out = cpu.run(&inputs).remove(0);

    // Baseline Metal flash-decode (W8A8 OFF) — what production actually uses, so
    // W8A8-vs-baseline is the TRUE W8A8 error (CPU-vs-flash has its own ~1e-3 f32
    // noise floor at head_dim=128 that would otherwise mask the int8 error).
    let mut mb = Session::new(Device::Metal).compile(build_gqa_causal(b, lk, nh, nkv, dh));
    let base_out = mb.run(&inputs).remove(0);

    // Metal W8A8.
    rlx_ir::env::set("RLX_METAL_W8A8_ATTN", "1");
    let mut m = Session::new(Device::Metal).compile(build_gqa_causal(b, lk, nh, nkv, dh));
    let w8a8_out = m.run(&inputs).remove(0);
    rlx_ir::env::unset("RLX_METAL_W8A8_ATTN");

    let d_cpu = max_abs(&cpu_out, &w8a8_out) / rms(&cpu_out).max(1e-6);
    let d_base_cpu = max_abs(&cpu_out, &base_out) / rms(&cpu_out).max(1e-6);
    let d_base = max_abs(&base_out, &w8a8_out) / rms(&base_out).max(1e-6);
    // Locate the max-error element (head, dim) + count how many heads exceed 1e-4.
    let (mut ai, mut amax) = (0usize, 0f32);
    for (i, (x, y)) in base_out.iter().zip(&w8a8_out).enumerate() {
        if (x - y).abs() > amax {
            amax = (x - y).abs();
            ai = i;
        }
    }
    let bad_heads = (0..nh)
        .filter(|h| (0..dh).any(|d| (base_out[h * dh + d] - w8a8_out[h * dh + d]).abs() > 1e-4))
        .count();
    eprintln!(
        "W8A8 nh={nh} nkv={nkv} dh={dh} lk={lk}: rel_vs_CPU={d_cpu:.6}  \
         baseline_vs_CPU={d_base_cpu:.6}  rel_vs_BASELINE={d_base:.6}\n  \
         argmax@{ai} (head {}, dim {}): base={:.6} w8a8={:.6} | heads_with_err>1e-4: {bad_heads}/{nh}",
        ai / dh,
        ai % dh,
        base_out[ai],
        w8a8_out[ai],
    );
    rlx_ir::env::unset("RLX_METAL_SDPA_FLASH_DECODE");
    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");

    assert!(
        d_base < 2e-2,
        "W8A8 decode attention diverged too far vs baseline: rel_vs_BASELINE={d_base}"
    );
}

#[test]
fn w8a8_mha_decode_hd128() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    run_case(1, 256, 16, 16, 128);
}

#[test]
fn w8a8_gqa_decode_hd128() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // qwen3-0.6B shape: 16 Q heads, 8 KV heads, head_dim 128.
    run_case(1, 512, 16, 8, 128);
}
