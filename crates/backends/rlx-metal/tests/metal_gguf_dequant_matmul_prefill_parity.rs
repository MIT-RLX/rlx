// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Packed GGUF `Op::DequantMatMul` PREFILL parity (m > 1) — Metal vs CPU.
//!
//! Decode (m == 1) uses the fused `q4k_mv_f32*` GEMV kernels. Prefill
//! (seq > 1 → m > 1) takes the thunk `else` branch: the `dequant_gguf` MSL
//! kernel expands the packed Q4_K / Q6_K weight `[n, k]` to an f32 scratch,
//! then `encode_mps_sgemm_bt` runs a real `MPSMatrixMultiplication` sgemm
//! (B^T). This is NOT the MPSGraph path (which never sees packed GGUF — see
//! `can_lower_dequant_in_mps`). These cases prove the m > 1 GEMM matches the
//! CPU `DequantMatMul` reference over the same packed bytes.

#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// Build `[m, n] = x[m, k] @ dequant(w)[n, k]^T` for a packed GGUF weight and
/// compare Metal against the CPU reference over identical packed bytes.
/// Returns the max abs elementwise difference.
fn run_case(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    m: usize,
    k: usize,
    n: usize,
) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return None;
    }

    // Weight is row-major [n, k]: output column c owns k contiguous values.
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("gguf_dq_prefill");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&[("x", x.as_slice())]).remove(0)
    };

    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    assert_eq!(metal.len(), m * n, "metal output len");
    assert_eq!(cpu.len(), m * n, "cpu output len");
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cmax = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let mmax = metal.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!(
        "gguf_dequant_matmul {scheme:?} m={m} k={k} n={n}: max_abs={max_abs:.6e} cpu_max={cmax:.4} metal_max={mmax:.4}"
    );
    Some(max_abs)
}

/// All cases run inside ONE `#[test]` so they execute serially on a single
/// thread. The m > 1 prefill path uses `MPSMatrixMultiplication`, whose global
/// matrix/kernel cache is not safe to build concurrently from the independent
/// `Session`s that parallel test threads create — splitting into separate
/// `#[test]` fns SIGSEGVs under the default multi-thread harness.
/// A CHAIN of DequantMatMuls interspersed with activations — forces many
/// MPSGraph/thunk segment boundaries (`mps_graph_hybrid` splits at every
/// DequantMatMul), the way the 64-layer 27B does. A single DequantMatMul is
/// bit-exact; this checks the metal segmentation/boundary-I/O across many.
#[test]
fn gguf_dequant_matmul_chain_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let m = 4usize;
    let k = 256usize; // square weights so we can chain
    let n_layers = 8usize;
    let scheme = QuantScheme::GgufQ1_0;

    let mut g = Graph::new("gguf_dq_chain");
    let mut cur = g.input("x", Shape::new(&[m, k], DType::F32));
    let mut packs: Vec<(String, Vec<u8>)> = Vec::new();
    for l in 0..n_layers {
        let w_row: Vec<f32> = (0..k * k)
            .map(|i| ((i as f32 + l as f32 * 7.0) * 0.011).sin() * 0.3)
            .collect();
        let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("q");
        let name = format!("w{l}");
        let w = g.param(&name, Shape::new(&[packed.len()], DType::U8));
        let mm = g.add_node(
            Op::DequantMatMul { scheme },
            vec![cur, w],
            Shape::new(&[m, k], DType::F32),
        );
        // activation between layers → forces a segment boundary + keeps
        // values bounded (silu).
        cur = g.add_node(
            Op::Activation(Activation::Silu),
            vec![mm],
            Shape::new(&[m, k], DType::F32),
        );
        packs.push((name, packed));
    }
    g.set_outputs(vec![cur]);

    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        for (name, packed) in &packs {
            c.set_param_typed(name, packed, DType::U8);
        }
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dot: f32 = cpu.iter().zip(&metal).map(|(a, b)| a * b).sum();
    let nc: f32 = cpu.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nm: f32 = metal.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos = dot / (nc * nm + 1e-9);
    eprintln!("gguf_dequant_matmul CHAIN({n_layers}L) Q1_0: max_abs={max_abs:.6e} cos={cos:.6}");
    assert!(
        cos > 0.999,
        "dequant-matmul chain metal diverges: cos={cos} max_abs={max_abs}"
    );
}

/// Isolates the Bonsai-27B **padded-prefill hybrid** divergence: a leading
/// `RmsNorm` (MPSGraph-lowered) feeding a Q1_0 `DequantMatMul` chain (thunk
/// boundaries) at a LARGE `m` (padded seq, e.g. 96). The 27B degenerated
/// (cos≈-0.14) only at max_seq≈96 through `run_via_mps_hybrid`; the m=4 chain
/// test above is bit-exact, so the trigger is the large-m boundary hand-off
/// between an MPSGraph sub-graph output and the next thunk's input.
/// `RLX_HYBRID_M` overrides m (default 96); set 4 to see it pass.
#[test]
fn hybrid_rmsnorm_dequant_large_m_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let m: usize = std::env::var("RLX_HYBRID_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(96);
    let k: usize = std::env::var("RLX_HYBRID_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256); // square so we can chain
    let n_layers = 4usize;
    let scheme = QuantScheme::GgufQ1_0;

    let mut g = Graph::new("hybrid_rmsnorm_dq");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[k], DType::F32));
    let beta = g.param("beta", Shape::new(&[k], DType::F32));
    // RmsNorm → MPSGraph segment; its output feeds the first dequant thunk.
    let mut cur = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![x_in, gamma, beta],
        Shape::new(&[m, k], DType::F32),
    );
    let mut packs: Vec<(String, Vec<u8>)> = Vec::new();
    for l in 0..n_layers {
        let w_row: Vec<f32> = (0..k * k)
            .map(|i| ((i as f32 + l as f32 * 7.0) * 0.011).sin() * 0.3)
            .collect();
        let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("q");
        let name = format!("w{l}");
        let w = g.param(&name, Shape::new(&[packed.len()], DType::U8));
        let mm = g.add_node(
            Op::DequantMatMul { scheme },
            vec![cur, w],
            Shape::new(&[m, k], DType::F32),
        );
        // RmsNorm between layers → another MPSGraph segment boundary, plus
        // keeps values bounded (mirrors qwen35's pre-attn / pre-ffn norms).
        cur = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![mm, gamma, beta],
            Shape::new(&[m, k], DType::F32),
        );
        packs.push((name, packed));
    }
    g.set_outputs(vec![cur]);

    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let gamma_v = vec![1.0f32; k];
    let beta_v = vec![0.0f32; k];
    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param("gamma", &gamma_v);
        c.set_param("beta", &beta_v);
        for (name, packed) in &packs {
            c.set_param_typed(name, packed, DType::U8);
        }
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dot: f32 = cpu.iter().zip(&metal).map(|(a, b)| a * b).sum();
    let nc: f32 = cpu.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nm: f32 = metal.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos = dot / (nc * nm + 1e-9);
    eprintln!(
        "hybrid RmsNorm+Q1_0 chain m={m} k={k} {n_layers}L: max_abs={max_abs:.6e} cos={cos:.6}"
    );
    assert!(
        cos > 0.999,
        "hybrid rmsnorm+dequant m={m} metal diverges: cos={cos} max_abs={max_abs}"
    );
}

/// Fused `q1_0_mm_f32` parity: SEVERAL large-n Q1_0 `DequantMatMul`s in ONE
/// graph (all reading the same `x`), the way each Bonsai-27B layer fires
/// qkv/gate/ffn projections. Exercises the packed-weight-direct GEMM at
/// Bonsai's k=5120/n=6144 scale and asserts non-zero, NaN-free output matching
/// CPU. (NB: the shared-scratch race that zeroed these projections on the old
/// `dequant_gguf → scratch → MPS sgemm` path only reproduces at full 64-layer
/// graph scale — see the e2e Bonsai run — so this guards the fused path's
/// correctness rather than reproducing the race itself.)
#[test]
fn gguf_dequant_matmul_multi_large_n_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let m = 8usize;
    let k = 5120usize; // Bonsai hidden
    let n = 6144usize; // Bonsai GDN inner (large n that raced to zero)
    let n_proj = 4usize;
    let scheme = QuantScheme::GgufQ1_0;

    let mut g = Graph::new("gguf_dq_multi_large");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let mut packs: Vec<(String, Vec<u8>)> = Vec::new();
    let mut outs = Vec::new();
    for p in 0..n_proj {
        let w_row: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 + p as f32 * 3.0) * 0.011).sin() * 0.4)
            .collect();
        let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("q");
        let name = format!("w{p}");
        let w = g.param(&name, Shape::new(&[packed.len()], DType::U8));
        let mm = g.add_node(
            Op::DequantMatMul { scheme },
            vec![x_in, w],
            Shape::new(&[m, n], DType::F32),
        );
        outs.push(mm);
        packs.push((name, packed));
    }
    g.set_outputs(outs);

    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let run = |device: Device| -> Vec<Vec<f32>> {
        let mut c = Session::new(device).compile(g.clone());
        for (name, packed) in &packs {
            c.set_param_typed(name, packed, DType::U8);
        }
        c.run(&[("x", x.as_slice())])
    };
    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    for p in 0..n_proj {
        let mmax = metal[p].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let nan = metal[p].iter().filter(|v| v.is_nan()).count();
        let max_abs = cpu[p]
            .iter()
            .zip(&metal[p])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("proj {p}: metal_max={mmax:.4} nan={nan} max_abs_vs_cpu={max_abs:.4}");
        assert!(
            mmax > 0.0 && nan == 0,
            "proj {p} Metal produced zeros/NaN (scratch race)"
        );
        assert!(max_abs < 2.0, "proj {p} Metal vs CPU max_abs={max_abs}");
    }
}

#[test]
fn gguf_dequant_matmul_prefill_matches_cpu() {
    // (scheme, ggml, m, k, n, tol, label)
    let cases: &[(
        QuantScheme,
        rlx_gguf::GgmlType,
        usize,
        usize,
        usize,
        f32,
        &str,
    )] = &[
        // Task-specified prefill shape: m=4, one Q4_K superblock (k=256), n=8.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            4,
            256,
            8,
            1e-3,
            "Q4_K prefill",
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            4,
            256,
            8,
            1e-3,
            "Q6_K prefill",
        ),
        // k=512 (two superblocks), wider n, larger m → real GEMM tiling.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            7,
            512,
            16,
            1e-3,
            "Q4_K multi-superblock",
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            7,
            512,
            16,
            1e-3,
            "Q6_K multi-superblock",
        ),
        // Decode (m=1) regression guard: fused GEMV path must stay correct.
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            1,
            256,
            8,
            1e-3,
            "Q4_K decode",
        ),
        // Q1_0 (prism-ml Bonsai-27B) — 128-elem blocks, on-device dequant.
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            4,
            256,
            8,
            1e-2,
            "Q1_0 prefill",
        ),
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            7,
            512,
            16,
            2e-2,
            "Q1_0 multi-block",
        ),
        // Bonsai FFN scale: k=5120 (hidden), n=6144 (ffn). Directly the
        // 27B-on-Metal DequantMatMul the synth F32 path can't exercise.
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            8,
            5120,
            6144,
            2.0,
            "Q1_0 bonsai-ffn-scale",
        ),
        // PADDED-PREFILL scale: m=96 (the max_seq bucket that degenerated the
        // real 27B), k=5120, n=10240 (GDN qkv projection). The dump showed
        // this exact node emit ALL zeros (nz=0/983040) on Metal at m=96 while
        // m=8/32 are correct — isolates encode_mps_sgemm_bt at large m·n.
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            96,
            5120,
            10240,
            3.0,
            "Q1_0 bonsai m96 n10240",
        ),
        (
            QuantScheme::GgufQ1_0,
            rlx_gguf::GgmlType::Q1_0,
            96,
            5120,
            6144,
            3.0,
            "Q1_0 bonsai m96 n6144",
        ),
    ];
    for &(scheme, ggml, m, k, n, tol, label) in cases {
        if let Some(max_abs) = run_case(scheme, ggml, m, k, n) {
            assert!(
                max_abs < tol,
                "{label} Metal vs CPU max_abs={max_abs} (tol {tol})"
            );
        }
    }
}

#[test]
fn q1_0_decode_amp_f16_gemv_only_matches_f32() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let m = 1usize;
    let k = 256usize;
    let n = 64usize;
    let scheme = QuantScheme::GgufQ1_0;
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("q1_amp_gemv");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let feeds = [("x", x.as_slice())];
    let run = |device: Device, amp: bool| -> Vec<f32> {
        let mut opts = rlx_runtime::CompileOptions::new();
        if amp {
            opts = opts.policy(rlx_runtime::PrecisionPolicy::AutoMixed);
        }
        let mut c = Session::new(device).compile_with(g.clone(), &opts);
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&feeds).remove(0)
    };

    let cpu = run(Device::Cpu, false);
    let metal_f32 = run(Device::Metal, false);
    let metal_amp = run(Device::Metal, true);
    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (na * nb + 1e-9)
    };
    let c_amp = cos(&cpu, &metal_amp);
    let c_pair = cos(&metal_f32, &metal_amp);
    eprintln!("q1_0 AMP f16 GEMV-only: cos(cpu,amp)={c_amp:.6} cos(f32,amp)={c_pair:.6}");
    assert!(
        c_amp > 0.99,
        "Metal AMP Q1 GEMV diverges vs CPU: cos={c_amp}"
    );
    assert!(
        c_pair > 0.99,
        "Metal AMP vs F32 Q1 GEMV diverges: cos={c_pair}"
    );
}

/// True f16 residual stream under AMP: decode Q1_0 GEMV + residual add.
/// Compares Metal AlwaysF32 vs Metal AutoMixed (f16 acts / residual) and
/// both against CPU F32. Exercises `x_f16` / `dst_f16` / `res_f16` on the
/// simdgroup Q1 kernels (weight-BW path: packed W + half activation traffic).
#[test]
fn q1_0_decode_amp_f16_residual_matches_f32() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let m = 1usize;
    let k = 256usize;
    let n = 64usize; // %8==0 → SG path
    let scheme = QuantScheme::GgufQ1_0;

    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, rlx_gguf::GgmlType::Q1_0).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let res: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).cos() * 0.25).collect();

    let mut g = Graph::new("q1_amp_residual");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let res_in = g.input("res", Shape::new(&[m, n], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let mm = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    let y = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![res_in, mm],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let feeds = [("x", x.as_slice()), ("res", res.as_slice())];
    let run = |device: Device, amp: bool| -> Vec<f32> {
        let mut opts = rlx_runtime::CompileOptions::new();
        if amp {
            opts = opts.policy(rlx_runtime::PrecisionPolicy::AutoMixed);
        }
        let mut c = Session::new(device).compile_with(g.clone(), &opts);
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&feeds).remove(0)
    };

    let cpu = run(Device::Cpu, false);
    let metal_f32 = run(Device::Metal, false);
    let metal_amp = run(Device::Metal, true);

    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (na * nb + 1e-9)
    };
    let max_abs = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let c_f32 = cos(&cpu, &metal_f32);
    let c_amp = cos(&cpu, &metal_amp);
    let c_pair = cos(&metal_f32, &metal_amp);
    eprintln!(
        "q1_0 AMP f16 residual: cos(cpu,metal_f32)={c_f32:.6} cos(cpu,amp)={c_amp:.6} \
         cos(f32,amp)={c_pair:.6} max_abs(f32,amp)={:.4e}",
        max_abs(&metal_f32, &metal_amp)
    );
    assert!(c_f32 > 0.999, "Metal F32 Q1 residual diverges: cos={c_f32}");
    assert!(
        c_amp > 0.99,
        "Metal AMP Q1 residual diverges vs CPU: cos={c_amp}"
    );
    assert!(
        c_pair > 0.99,
        "Metal AMP vs F32 residual stream diverges: cos={c_pair}"
    );
}

#[test]
fn q2_0_decode_gemv_matches_reference() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let k = 256usize;
    let n = 16usize;
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let mut packed = Vec::with_capacity(n * (k / 128) * 34);
    let mut reference = vec![0.0f32; n];
    for row in 0..n {
        for block in 0..(k / 128) {
            let d = 0.125 + (row * 2 + block) as f32 * 0.003;
            let d_half = half::f16::from_f32(d);
            let d_ref = d_half.to_f32();
            packed.extend_from_slice(&d_half.to_le_bytes());
            for byte in 0..32usize {
                let mut bits = 0u8;
                for lane in 0..4usize {
                    let code = ((row + block + byte + lane) & 3) as u8;
                    bits |= code << (lane * 2);
                    let j = block * 128 + byte * 4 + lane;
                    reference[row] += x[j] * (code as f32 - 1.0) * d_ref;
                }
                packed.push(bits);
            }
        }
    }
    let mut g = Graph::new("q2_0_decode_gemv");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ2_0,
        },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Metal).compile(g);
    c.set_param_typed("w", &packed, DType::U8);
    let metal = c.run(&[("x", x.as_slice())]).remove(0);
    let max_abs = reference
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 2e-3, "Q2_0 decode Metal max_abs={max_abs}");
}
