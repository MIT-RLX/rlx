// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct correctness test for matmul_coop_f32 (Metal simdgroup path).
//! On Vulkan/DX12 the portable coop kernel auto-enables when 8×8 f32
//! cooperative-matrix support is present.

use rlx_ir::infer::GraphExt;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// `simdgroup_float8x8` is a genuine f32 multiply with an f32 accumulator, so
/// these shapes land within f32 rounding of a host f64-free reference.
///
/// This gate used to read `ATOL = 0.05, RTOL = 10.0` — a 1000% relative
/// tolerance — justified as "reduced-precision internal accumulators". There is
/// no such reduced precision. The loose bound was hiding a transposed product
/// (the kernel computed `b·a`; see the operand-role note in
/// `kernels/matmul_coop_f32.wgsl`), which has roughly the same magnitude as the
/// right answer and so passed a magnitude-relative check. Measured max|Δ| at
/// the three shapes below is 2.1e-8 / 3.0e-8 / 2.4e-8.
///
/// Keep this tight. A tolerance a wrong answer can pass is not a test.
fn coop_f32_close(max_diff: f32, abs_max_expected: f32) -> bool {
    const ATOL: f32 = 1e-6;
    const RTOL: f32 = 1e-4;
    max_diff < ATOL.max(abs_max_expected * RTOL)
}

fn require_coop_f32_test() -> bool {
    let dev = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return false;
        }
    };
    let forced = rlx_ir::env::flag("RLX_WGPU_FORCE_COOP_F32");
    let discrete =
        rlx_wgpu::device::coop_discrete_backend() && rlx_wgpu::device::coop_f32_8x8_supported();
    if dev.backend != wgpu::Backend::Metal && !forced && !discrete {
        eprintln!(
            "CoopF32 auto path is Metal-only or discrete 8×8 f32 coop; skipping on {:?} \
             (set RLX_WGPU_FORCE_COOP_F32=1 to probe)",
            dev.backend
        );
        return false;
    }
    true
}

#[test]
fn coop_f32_uses_real_f32() {
    if !require_coop_f32_test() {
        return;
    }

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
    if !require_coop_f32_test() {
        return;
    }
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
        coop_f32_close(max_diff, abs_max_expected),
        "matmul_coop_f32 at BERT-QKV shape diverges from f32 ref: max|Δ|={max_diff}"
    );
}

#[test]
fn coop_f32_correct_chained_matmuls() {
    if !require_coop_f32_test() {
        return;
    }
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
        coop_f32_close(max_diff, abs_max),
        "chained CoopF32 matmuls diverge: max|Δ|={max_diff} max|exp|={abs_max}"
    );
}

#[test]
fn coop_f32_correct_with_bias_via_fmb() {
    if !require_coop_f32_test() {
        return;
    }
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
        coop_f32_close(max_diff, abs_max),
        "FMB CoopF32 diverges: max|Δ|={max_diff}"
    );
}

/// The probe that would have caught the transposed product on day one.
///
/// Every other case in this file multiplies two dense operands, which the old
/// loose tolerance let slide. This one is shaped so that `a·b` and `b·a` differ
/// *structurally*, not just numerically, so no tolerance can paper over it:
///
///   A is a column vector (only k=0 populated), B is a row vector (only k=0).
///   `a·b` is the full rank-1 outer product — every one of the 32×32 outputs is
///   non-zero. `b·a` collapses each 8×8 fragment to a single dot product parked
///   at its [0][0], leaving 63/64 of the output exactly zero.
///
/// The lesson this encodes: the pre-existing tests multiplied by an identity
/// (`A·I`, `I·B`) or by another dense matrix. An identity *commutes*, so it
/// cannot distinguish `a·b` from `b·a` — a swapped-operand bug passes both.
/// Two structurally different non-commuting operands are what it takes.
#[test]
fn coop_f32_operand_order_is_not_commuted() {
    if !require_coop_f32_test() {
        return;
    }
    // One workgroup, one k-tile: the smallest case that exercises the
    // fragment roles at all.
    const M: usize = 32;
    const K: usize = 8;
    const N: usize = 32;

    let mut g = Graph::new("coop_f32_outer");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let mut a_data = vec![0.0_f32; M * K];
    for (i, row) in a_data.chunks_mut(K).enumerate() {
        row[0] = i as f32 + 1.0;
    }
    let mut b_data = vec![0.0_f32; K * N];
    for (j, v) in b_data[..N].iter_mut().enumerate() {
        *v = (j as f32 + 1.0) * 0.01;
    }

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &b_data);
    let outs = exe.run(&[("a", a_data.as_slice())]);
    let out = &outs[0];

    let mut max_diff = 0.0_f32;
    let mut zeros = 0usize;
    for i in 0..M {
        for j in 0..N {
            let want = (i as f32 + 1.0) * (j as f32 + 1.0) * 0.01;
            let got = out[i * N + j];
            max_diff = max_diff.max((got - want).abs());
            if got == 0.0 {
                zeros += 1;
            }
        }
    }
    eprintln!(
        "outer-product max|Δ| = {max_diff}, exact zeros = {zeros}/{}",
        M * N
    );
    // `b·a` leaves 1008/1024 outputs at exactly zero; `a·b` leaves none.
    assert_eq!(
        zeros,
        0,
        "{zeros}/{} outputs are exactly zero — the rank-1 outer product collapsed, \
         which is the signature of a transposed/commuted operand pair",
        M * N
    );
    assert!(
        max_diff < 1e-5,
        "outer product diverges from a·b: max|Δ|={max_diff}"
    );
}

/// The fused split-QKV variant, `kernels/matmul_qkv_coop_f32.wgsl`.
///
/// It is a separate shader from `matmul_coop_f32.wgsl` with the same tile
/// structure, reached only when a `FusedMatMulBiasAct` is followed by three
/// `Narrow`s along the last axis (the Q/K/V split) — so the plain matmul tests
/// above never touch it, and it carried an identical transposed product.
///
/// Q, K and V are checked separately: a fault in the column routing shows up in
/// one slice and not the others.
#[test]
fn coop_f32_qkv_split_matches_reference() {
    if !require_coop_f32_test() {
        return;
    }
    // m%32, k%8, n%32 — and n = 3*H so the three narrows are 32-aligned too.
    const M: usize = 64;
    const K: usize = 128;
    const H: usize = 64;
    const N: usize = 3 * H;

    let mut g = Graph::new("coop_f32_qkv");
    let x = g.input("x", Shape::new(&[M, K], DType::F32));
    let w = g.param("w", Shape::new(&[K, N], DType::F32));
    let bias = g.param("bias", Shape::new(&[N], DType::F32));
    let fmb = g.add_node(
        Op::FusedMatMulBiasAct { activation: None },
        vec![x, w, bias],
        Shape::new(&[M, N], DType::F32),
    );
    let q = g.narrow_(fmb, 1, 0, H);
    let k = g.narrow_(fmb, 1, H, H);
    let v = g.narrow_(fmb, 1, 2 * H, H);
    g.set_outputs(vec![q, k, v]);

    let x_data: Vec<f32> = (0..M * K).map(|i| 0.1 * (i as f32 * 0.37).sin()).collect();
    let w_data: Vec<f32> = (0..K * N).map(|i| 0.05 * (i as f32 * 0.71).cos()).collect();
    let bias_data: Vec<f32> = (0..N).map(|i| 0.01 * (i as f32).sin()).collect();

    let mut full = vec![0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut acc = bias_data[j];
            for kk in 0..K {
                acc += x_data[i * K + kk] * w_data[kk * N + j];
            }
            full[i * N + j] = acc;
        }
    }

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &w_data);
    exe.set_param("bias", &bias_data);
    let outs = exe.run(&[("x", x_data.as_slice())]);

    for (slice, name) in [(0usize, "Q"), (1, "K"), (2, "V")] {
        let got = &outs[slice];
        assert_eq!(got.len(), M * H, "{name} has the wrong length");
        let mut max_diff = 0.0_f32;
        let mut abs_max = 0.0_f32;
        for i in 0..M {
            for j in 0..H {
                let want = full[i * N + slice * H + j];
                max_diff = max_diff.max((got[i * H + j] - want).abs());
                abs_max = abs_max.max(want.abs());
            }
        }
        eprintln!("qkv {name}: max|Δ| = {max_diff}, max|expected| = {abs_max}");
        assert!(
            coop_f32_close(max_diff, abs_max),
            "split-QKV CoopF32 {name} diverges: max|Δ|={max_diff}"
        );
    }
}
