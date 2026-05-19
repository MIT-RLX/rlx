// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Direct correctness test for matmul_coop_f32. We feed a matmul whose
//! intermediate values exceed f16 max (65504). If the kernel is honoring
//! f32 the result is exact; if naga silently downcasts to half on Apple,
//! the result will be wrong by orders of magnitude.

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

#[test]
fn coop_f32_uses_real_f32() {
    let _ = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };

    // M=32, K=8, N=32 — meets the coop alignment (m%32==0, k%8==0, n%32==0).
    // A: all 1.0
    // B: all 70_000.0  (f16 max = 65504, so 70_000 saturates in f16)
    // Expected C: each cell = 8 * 70_000 = 560_000.0
    // If f16-downcast on B: each cell = 8 * 65504 = 524_032.0
    const M: usize = 32;
    const K: usize = 8;
    const N: usize = 32;

    let mut g = Graph::new("coop_f32_probe");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &vec![70000.0_f32; K * N]);
    let outs = exe.run(&[("a", vec![1.0_f32; M * K].as_slice())]);
    let out = &outs[0];

    let expected = (K as f32) * 70000.0; // 560_000.0
    let f16_capped = (K as f32) * 65504.0; // 524_032.0
    let observed = out[0];
    eprintln!("expected (f32) = {expected}, f16-cap = {f16_capped}, observed = {observed}");

    let err_vs_f32 = (observed - expected).abs();
    let err_vs_f16 = (observed - f16_capped).abs();
    eprintln!("err vs f32 = {err_vs_f32}, err vs f16-cap = {err_vs_f16}");

    if err_vs_f32 < 1.0 {
        eprintln!("kernel is honoring f32 (good)");
    } else if err_vs_f16 < 100.0 {
        panic!(
            "kernel is silently downcasting to f16 (observed {observed} ≈ f16-saturated {f16_capped})"
        );
    } else {
        panic!("kernel produces unexpected output ({observed}, expected {expected})");
    }
}

#[test]
fn coop_f32_correct_at_minilm_qkv() {
    // EXACT failing shape: MiniLM6 QKV at b=32 s=3. M=96, K=384, N=1152.
    let _ = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };
    const M: usize = 96;
    const K: usize = 384;
    const N: usize = 1152;

    let mut g = Graph::new("coop_f32_bertk");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    // Deterministic non-trivial values: A[i,k] = 0.1*sin(i+k), B[k,j] = 0.1*cos(k-j)
    let a_data: Vec<f32> = (0..M * K).map(|x| 0.1 * (x as f32).sin()).collect();
    let b_data: Vec<f32> = (0..K * N).map(|x| 0.1 * (x as f32).cos()).collect();

    // Reference matmul on host
    let mut expected = vec![0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut s = 0f32;
            for kk in 0..K {
                s += a_data[i * K + kk] * b_data[kk * N + j];
            }
            expected[i * N + j] = s;
        }
    }

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &b_data);
    let outs = exe.run(&[("a", a_data.as_slice())]);
    let out = &outs[0];

    let max_diff = expected
        .iter()
        .zip(out.iter())
        .map(|(e, o)| (e - o).abs())
        .fold(0.0_f32, f32::max);
    let abs_max_expected = expected.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "max|Δ| = {max_diff}, max|expected| = {abs_max_expected}, rel = {}",
        max_diff / abs_max_expected.max(1e-30)
    );
    assert!(
        max_diff < abs_max_expected * 1e-3,
        "matmul_coop_f32 at BERT-QKV shape diverges from f32 ref: max|Δ|={max_diff}"
    );
}

#[test]
fn coop_f32_correct_chained_matmuls() {
    // Three matmuls in sequence — same shapes as BERT FFN: in→fc1→fc2.
    // Tests whether output of one matmul, consumed as A by the next,
    // is corrupted somehow.
    let _ = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };
    const M: usize = 96;
    const H: usize = 384;
    const I: usize = 1536;

    let mut g = Graph::new("coop_f32_chained");
    let x = g.input("x", Shape::new(&[M, H], DType::F32));
    let w1 = g.param("w1", Shape::new(&[H, I], DType::F32));
    let w2 = g.param("w2", Shape::new(&[I, H], DType::F32));
    let h = g.matmul(x, w1, Shape::new(&[M, I], DType::F32));
    let y = g.matmul(h, w2, Shape::new(&[M, H], DType::F32));
    g.set_outputs(vec![y]);

    let x_data: Vec<f32> = (0..M * H).map(|i| 0.1 * (i as f32).sin()).collect();
    let w1_data: Vec<f32> = (0..H * I).map(|i| 0.05 * (i as f32 * 0.7).cos()).collect();
    let w2_data: Vec<f32> = (0..I * H).map(|i| 0.05 * (i as f32 * 1.3).sin()).collect();

    // CPU reference
    let mut h_ref = vec![0f32; M * I];
    for i in 0..M {
        for j in 0..I {
            let mut s = 0f32;
            for k in 0..H {
                s += x_data[i * H + k] * w1_data[k * I + j];
            }
            h_ref[i * I + j] = s;
        }
    }
    let mut y_ref = vec![0f32; M * H];
    for i in 0..M {
        for j in 0..H {
            let mut s = 0f32;
            for k in 0..I {
                s += h_ref[i * I + k] * w2_data[k * H + j];
            }
            y_ref[i * H + j] = s;
        }
    }

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w1", &w1_data);
    exe.set_param("w2", &w2_data);
    let outs = exe.run(&[("x", x_data.as_slice())]);
    let out = &outs[0];

    let max_diff = y_ref
        .iter()
        .zip(out.iter())
        .map(|(e, o)| (e - o).abs())
        .fold(0.0_f32, f32::max);
    let abs_max = y_ref.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "chained max|Δ| = {max_diff}, max|expected| = {abs_max}, rel = {}",
        max_diff / abs_max.max(1e-30)
    );
    assert!(
        max_diff < abs_max * 1e-2,
        "chained CoopF32 matmuls diverge: max|Δ|={max_diff} max|exp|={abs_max}"
    );
}

#[test]
fn coop_f32_correct_with_bias_via_fmb() {
    // FusedMatMulBiasAct (matmul + bias + GELU). Mimics BERT fc1.
    let _ = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };
    const M: usize = 96;
    const K: usize = 384;
    const N: usize = 1536;

    let mut g = Graph::new("coop_f32_fmb");
    let x = g.input("x", Shape::new(&[M, K], DType::F32));
    let w = g.param("w", Shape::new(&[K, N], DType::F32));
    let b = g.param("b", Shape::new(&[N], DType::F32));
    let y = g.add_node(
        Op::FusedMatMulBiasAct {
            activation: Some(Activation::Gelu),
        },
        vec![x, w, b],
        Shape::new(&[M, N], DType::F32),
    );
    g.set_outputs(vec![y]);

    let x_data: Vec<f32> = (0..M * K).map(|i| 0.1 * (i as f32).sin()).collect();
    let w_data: Vec<f32> = (0..K * N).map(|i| 0.05 * (i as f32 * 0.7).cos()).collect();
    let b_data: Vec<f32> = (0..N).map(|i| 0.01 * (i as f32).sin()).collect();

    // CPU reference: matmul + bias + GELU(tanh-approx)
    let gelu = |v: f32| {
        let c = 0.797_884_6_f32;
        let inner = (c * (v + 0.044715 * v * v * v)).clamp(-15.0, 15.0);
        0.5 * v * (1.0 + inner.tanh())
    };
    let mut y_ref = vec![0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut s = b_data[j];
            for k in 0..K {
                s += x_data[i * K + k] * w_data[k * N + j];
            }
            y_ref[i * N + j] = gelu(s);
        }
    }

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &w_data);
    exe.set_param("b", &b_data);
    let outs = exe.run(&[("x", x_data.as_slice())]);
    let out = &outs[0];

    let max_diff = y_ref
        .iter()
        .zip(out.iter())
        .map(|(e, o)| (e - o).abs())
        .fold(0.0_f32, f32::max);
    let abs_max = y_ref.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "FMB max|Δ| = {max_diff}, max|expected| = {abs_max}, rel = {}",
        max_diff / abs_max.max(1e-30)
    );
    assert!(
        max_diff < abs_max * 1e-2,
        "FMB CoopF32 diverges: max|Δ|={max_diff}"
    );
}
