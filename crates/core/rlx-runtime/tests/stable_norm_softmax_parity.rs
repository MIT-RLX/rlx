// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dedicated regression locks for the three numerical backend fixes that
//! unblocked CUDA / generative-net parity in 0.2.13:
//!
//! 1. **Softmax arbitrary axis** (`softmax.cu` stride path) — not last-axis-only.
//! 2. **GroupNorm two-pass variance** (`group_norm.cu`) — not `E[x²]−E[x]²`.
//! 3. **wgpu LayerNorm two-pass variance** (`layernorm.wgsl`) — same identity trap.
//!
//! These sit on top of the large backend-split refactor; without micro-tests a
//! drive-by rewrite can silently restore the broken forms. Each case uses
//! inputs that make the old bug **loud** (non-last softmax axis, large DC
//! offset). Softmax / LayerNorm compare against a pure two-pass / strided
//! reference (CPU Softmax is last-axis-only; CPU LayerNorm still uses the
//! one-pass identity). GroupNorm uses CPU Session (already two-pass).

#![cfg(feature = "cpu")]
#![allow(dead_code)]

use rlx_ir::DType;
// Session/Graph/Shape + device probing are only referenced by the CUDA/wgpu
// parity cases below; gate the imports to match so a default (cpu-only) build
// neither fails to resolve them nor warns about unused imports.
#[cfg(any(
    feature = "cuda",
    feature = "gpu",
    feature = "rocm",
    feature = "vulkan",
    all(feature = "metal", target_os = "macos")
))]
use rlx_ir::{Graph, Shape};
#[cfg(any(
    feature = "cuda",
    feature = "gpu",
    feature = "rocm",
    feature = "vulkan",
    all(feature = "metal", target_os = "macos")
))]
use rlx_runtime::{Device, Session, is_available};

const F: DType = DType::F32;
const EPS: f32 = 1e-5;
/// DC large enough that `E[x²] − mean²` cancels in f32 (ulp(~1e6) ≈ 0.06).
const DC: f32 = 1_000.0;

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: len");
    let e = max_abs(got, want);
    assert!(
        e <= tol,
        "{label}: max_abs={e:.6e} > tol={tol} (n={})",
        got.len()
    );
    eprintln!("{label}: max_abs={e:.3e}");
}

#[cfg(any(
    feature = "cuda",
    feature = "gpu",
    feature = "rocm",
    feature = "vulkan",
    all(feature = "metal", target_os = "macos")
))]
fn run(device: Device, g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(device)
        .compile(g)
        .run(inputs)
        .pop()
        .expect("one output")
}

/// Softmax along `axis` with the same strided layout the CUDA kernel uses:
/// `base = o * axis_len * stride + s`, element `j` at `base + j * stride`.
#[cfg(any(feature = "cuda", feature = "gpu"))]
fn ref_softmax(x: &[f32], dims: &[usize], axis: usize) -> Vec<f32> {
    let rank = dims.len();
    assert!(axis < rank);
    let axis_len = dims[axis];
    let stride: usize = dims[axis + 1..].iter().product::<usize>().max(1);
    let outer: usize = dims[..axis].iter().product::<usize>().max(1);
    let num_rows = outer * stride;
    let mut y = x.to_vec();
    for r in 0..num_rows {
        let o = r / stride;
        let s = r % stride;
        let base = o * axis_len * stride + s;
        let mut m = f32::NEG_INFINITY;
        for j in 0..axis_len {
            m = m.max(x[base + j * stride]);
        }
        let mut sum = 0.0f64;
        for j in 0..axis_len {
            let e = (x[base + j * stride] - m).exp() as f64;
            y[base + j * stride] = e as f32;
            sum += e;
        }
        let inv = (1.0 / sum) as f32;
        for j in 0..axis_len {
            y[base + j * stride] *= inv;
        }
    }
    y
}

/// Two-pass LayerNorm over the last axis (matches the fixed wgpu kernel).
fn ref_layer_norm_two_pass(
    x: &[f32],
    rows: usize,
    inner: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    let n_inv = 1.0 / inner as f32;
    for r in 0..rows {
        let off = r * inner;
        let row = &x[off..off + inner];
        let mean = row.iter().sum::<f32>() * n_inv;
        let var = row
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            * n_inv;
        let inv = 1.0 / (var + EPS).sqrt();
        for i in 0..inner {
            y[off + i] = (row[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
    y
}

/// One-pass `E[x²]−mean²` LayerNorm — the form the wgpu bug used. Exposed so
/// the DC-offset case asserts the two formulas actually diverge (otherwise the
/// micro-test is not stressing cancellation).
fn ref_layer_norm_one_pass(
    x: &[f32],
    rows: usize,
    inner: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    let n_inv = 1.0 / inner as f32;
    for r in 0..rows {
        let off = r * inner;
        let row = &x[off..off + inner];
        let (sum, sumsq) = row
            .iter()
            .fold((0.0f32, 0.0f32), |(s, ss), &v| (s + v, ss + v * v));
        let mean = sum * n_inv;
        let var = (sumsq * n_inv - mean * mean).max(0.0);
        let inv = 1.0 / (var + EPS).sqrt();
        for i in 0..inner {
            y[off + i] = (row[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
    y
}

fn dc_ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| DC + (((i + seed) % 17) as f32 - 8.0) * 0.05)
        .collect()
}

/// Pin: CUDA Softmax with `stride != 1` (axis not last) matches the strided
/// reference. Last-axis-only kernels produce garbage here.
#[test]
#[cfg(feature = "cuda")]
fn cuda_softmax_non_last_axis_matches_reference() {
    if !is_available(Device::Cuda) {
        eprintln!("skip cuda_softmax_non_last_axis (CUDA unavailable)");
        return;
    }
    // [outer=2, axis=5, stride=3] → stride=3, num_rows=6.
    let dims = [2usize, 5, 3];
    let n: usize = dims.iter().product();
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.15)
        .collect();
    let want = ref_softmax(&x, &dims, 1);

    let mut g = Graph::new("sm_axis1");
    let xin = g.input("x", Shape::new(&dims, F));
    let y = g.softmax(xin, 1, Shape::new(&dims, F));
    g.set_outputs(vec![y]);
    let got = run(Device::Cuda, g, &[("x", x.as_slice())]);
    assert_close("cuda softmax axis=1 [2,5,3]", &got, &want, 1e-4);

    // Also pin axis=0 (outer=1, stride=15).
    let want0 = ref_softmax(&x, &dims, 0);
    let mut g0 = Graph::new("sm_axis0");
    let xin0 = g0.input("x", Shape::new(&dims, F));
    let y0 = g0.softmax(xin0, 0, Shape::new(&dims, F));
    g0.set_outputs(vec![y0]);
    let got0 = run(Device::Cuda, g0, &[("x", x.as_slice())]);
    assert_close("cuda softmax axis=0 [2,5,3]", &got0, &want0, 1e-4);
}

/// Pin: CUDA GroupNorm two-pass variance under a large DC offset. The old
/// one-pass form cancels (`E[x²] ≈ mean²`) and corrupts SplitUNet-style nets.
#[test]
#[cfg(feature = "cuda")]
fn cuda_group_norm_dc_offset_matches_cpu() {
    if !is_available(Device::Cuda) {
        eprintln!("skip cuda_group_norm_dc_offset (CUDA unavailable)");
        return;
    }
    let (n, c, h, w, groups) = (2usize, 8usize, 4usize, 4usize, 2usize);
    let x = dc_ramp(n * c * h * w, 1);
    let gamma = dc_ramp(c, 3)
        .into_iter()
        .map(|v| v - DC + 1.0)
        .collect::<Vec<_>>();
    let beta = dc_ramp(c, 7)
        .into_iter()
        .map(|v| v - DC)
        .collect::<Vec<_>>();

    let mut g = Graph::new("gn_dc");
    let xin = g.input("x", Shape::new(&[n, c, h, w], F));
    let gin = g.input("gamma", Shape::new(&[c], F));
    let bin = g.input("beta", Shape::new(&[c], F));
    let y = g.group_norm(xin, gin, bin, groups, EPS);
    g.set_outputs(vec![y]);

    let inputs: &[(&str, &[f32])] = &[
        ("x", x.as_slice()),
        ("gamma", gamma.as_slice()),
        ("beta", beta.as_slice()),
    ];
    let cpu = run(Device::Cpu, g.clone(), inputs);
    let cuda = run(Device::Cuda, g, inputs);
    // 5e-4: two-pass CUDA vs CPU agree; one-pass cancellation under DC≈1e3
    // is O(1)–O(10) abs error — still orders of magnitude above this tol.
    assert_close("cuda GroupNorm DC-offset", &cuda, &cpu, 5e-4);
}

/// Pin: wgpu LayerNorm two-pass under a large DC offset. One-pass cancellation
/// previously collapsed cosine similarity to ~0.3–0.8 on ViT/DINOv2-style rows.
#[test]
#[cfg(feature = "gpu")]
fn wgpu_layer_norm_dc_offset_matches_two_pass_reference() {
    if !is_available(Device::Gpu) {
        eprintln!("skip wgpu_layer_norm_dc_offset (wgpu unavailable)");
        return;
    }
    let (rows, inner) = (4usize, 64usize);
    let x = dc_ramp(rows * inner, 11);
    let gamma: Vec<f32> = (0..inner).map(|i| 0.8 + (i as f32) * 0.001).collect();
    let beta: Vec<f32> = (0..inner).map(|i| -0.1 + (i as f32) * 0.0005).collect();

    // Sanity: one-pass and two-pass must disagree under this DC — otherwise the
    // case is too mild to lock the fix.
    let one = ref_layer_norm_one_pass(&x, rows, inner, &gamma, &beta);
    let two = ref_layer_norm_two_pass(&x, rows, inner, &gamma, &beta);
    let cancel = max_abs(&one, &two);
    assert!(
        cancel > 1e-2,
        "DC={DC} should make one-pass diverge from two-pass (got {cancel:.3e})"
    );

    let mut g = Graph::new("ln_dc");
    let xin = g.input("x", Shape::new(&[rows, inner], F));
    let gin = g.input("gamma", Shape::new(&[inner], F));
    let bin = g.input("beta", Shape::new(&[inner], F));
    let y = g.layer_norm(xin, gin, bin, -1, EPS, Shape::new(&[rows, inner], F));
    g.set_outputs(vec![y]);
    let inputs: &[(&str, &[f32])] = &[
        ("x", x.as_slice()),
        ("gamma", gamma.as_slice()),
        ("beta", beta.as_slice()),
    ];
    // CPU LayerNorm is now TWO-PASS (was one-pass E[x²]−mean²) → matches the
    // two-pass reference under this DC offset (validates the CPU change).
    let cpu = run(Device::Cpu, g.clone(), inputs);
    assert_close("cpu LayerNorm two-pass vs reference", &cpu, &two, 1e-3);
    let got = run(Device::Gpu, g, inputs);
    // Tolerance note: the wgpu kernel does a parallel pairwise-tree reduction,
    // *more* accurate than this reference's plain-f32 sequential `sum()`. Under
    // the extreme DC=1000 fixture the ~2e-4 residual is the reference's own
    // summation error, not the kernel's — still ~50× below the one-pass
    // cancellation this test guards (`cancel > 1e-2` above).
    assert_close("wgpu LayerNorm DC-offset vs two-pass", &got, &two, 1e-3);
}

/// CPU-only check that the DC fixture really stresses one-pass cancellation —
/// keeps the lock meaningful on CUDA-less / wgpu-less hosts.
#[test]
fn dc_offset_exposes_one_pass_layer_norm_cancellation() {
    let (rows, inner) = (2usize, 32usize);
    let x = dc_ramp(rows * inner, 3);
    let gamma = vec![1.0f32; inner];
    let beta = vec![0.0f32; inner];
    let one = ref_layer_norm_one_pass(&x, rows, inner, &gamma, &beta);
    let two = ref_layer_norm_two_pass(&x, rows, inner, &gamma, &beta);
    let e = max_abs(&one, &two);
    assert!(
        e > 1e-2,
        "expected one-pass vs two-pass divergence under DC={DC}, got {e:.3e}"
    );
}

/// Pin: wgpu LayerNorm workgroup-tree reduction with `inner` > the 64-wide
/// workgroup — exercises the per-thread striding loop (with a partial final
/// stride, `inner` not a multiple of 64) + the shared-memory tree, under a DC
/// offset (so the two-pass precision is also stressed). The prior scalar
/// one-thread-per-row kernel had no striding to get wrong; this locks the
/// parallel reduction.
#[test]
#[cfg(feature = "gpu")]
fn wgpu_layer_norm_striding_matches_two_pass_reference() {
    if !is_available(Device::Gpu) {
        eprintln!("skip wgpu_layer_norm_striding (wgpu unavailable)");
        return;
    }
    let (rows, inner) = (3usize, 130usize); // 130 = 2*64 + 2 → last stride is partial
    let x = dc_ramp(rows * inner, 23);
    let gamma: Vec<f32> = (0..inner).map(|i| 0.8 + (i as f32) * 0.001).collect();
    let beta: Vec<f32> = (0..inner).map(|i| -0.1 + (i as f32) * 0.0005).collect();
    let two = ref_layer_norm_two_pass(&x, rows, inner, &gamma, &beta);
    let mut g = Graph::new("ln_stride");
    let xin = g.input("x", Shape::new(&[rows, inner], F));
    let gin = g.input("gamma", Shape::new(&[inner], F));
    let bin = g.input("beta", Shape::new(&[inner], F));
    let y = g.layer_norm(xin, gin, bin, -1, EPS, Shape::new(&[rows, inner], F));
    g.set_outputs(vec![y]);
    let got = run(
        Device::Gpu,
        g,
        &[
            ("x", x.as_slice()),
            ("gamma", gamma.as_slice()),
            ("beta", beta.as_slice()),
        ],
    );
    // 1e-3: parallel-tree residual vs the plain-f32 reference under DC=1000 (see
    // the DC-offset test's tolerance note) — still ~20× below one-pass cancellation.
    assert_close(
        "wgpu LayerNorm inner=130 (striding) vs two-pass",
        &got,
        &two,
        1e-3,
    );
}

/// Two-pass RmsNorm over the last axis: `mean(x²) = mean((x−mean)²) + mean²`
/// (matches the fixed wgpu kernel's RmsNorm branch).
#[cfg(any(
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    all(feature = "metal", target_os = "macos")
))]
fn ref_rms_norm_two_pass(
    x: &[f32],
    rows: usize,
    inner: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    let n_inv = 1.0 / inner as f32;
    for r in 0..rows {
        let off = r * inner;
        let row = &x[off..off + inner];
        let mean = row.iter().sum::<f32>() * n_inv;
        let dev = row
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            * n_inv;
        let inv = 1.0 / (dev + mean * mean + EPS).sqrt();
        for i in 0..inner {
            // `+ beta[i]` matches the CPU oracle (RmsNorm carries beta).
            y[off + i] = row[i] * inv * gamma[i] + beta[i];
        }
    }
    y
}

/// Pin: wgpu RmsNorm shares the LayerNorm workgroup-tree reduction (op flag); it
/// only differs in the final scale. Validate the striding path under a DC offset
/// too. RmsNorm is BETTER-conditioned than LayerNorm here (mean² dominates
/// `inv_rms`, so it does not amplify the mean's rounding), hence the tighter tol.
#[test]
#[cfg(feature = "gpu")]
fn wgpu_rms_norm_striding_matches_two_pass_reference() {
    if !is_available(Device::Gpu) {
        eprintln!("skip wgpu_rms_norm_striding (wgpu unavailable)");
        return;
    }
    let (rows, inner) = (3usize, 130usize);
    let x = dc_ramp(rows * inner, 29);
    let gamma: Vec<f32> = (0..inner).map(|i| 0.8 + (i as f32) * 0.001).collect();
    // NON-ZERO beta: RmsNorm now adds beta (matches CPU). A dropped-beta kernel
    // (the prior wgpu bug) fails this.
    let beta: Vec<f32> = (0..inner).map(|i| -0.3 + (i as f32) * 0.002).collect();
    let two = ref_rms_norm_two_pass(&x, rows, inner, &gamma, &beta);
    let mut g = Graph::new("rms_stride");
    let xin = g.input("x", Shape::new(&[rows, inner], F));
    let gin = g.input("gamma", Shape::new(&[inner], F));
    let bin = g.input("beta", Shape::new(&[inner], F));
    let y = g.add_node(
        rlx_ir::Op::RmsNorm { axis: -1, eps: EPS },
        vec![xin, gin, bin],
        Shape::new(&[rows, inner], F),
    );
    g.set_outputs(vec![y]);
    let inputs: &[(&str, &[f32])] = &[
        ("x", x.as_slice()),
        ("gamma", gamma.as_slice()),
        ("beta", beta.as_slice()),
    ];
    // CPU is the sequential two-pass oracle → tight; confirms the reference
    // matches the actual (now two-pass) CPU kernel.
    let cpu = run(Device::Cpu, g.clone(), inputs);
    assert_close("cpu RmsNorm two-pass+beta vs reference", &cpu, &two, 1e-5);
    // wgpu tree vs the two-pass+beta reference/CPU.
    let got = run(Device::Gpu, g, inputs);
    assert_close(
        "wgpu RmsNorm inner=130 vs CPU two-pass+beta",
        &got,
        &two,
        3e-4,
    );
}

/// Metal RmsNorm is now two-pass + beta (matches the CPU oracle). Same DC +
/// non-zero-beta fixture; validates the Metal kernel change on-device.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_rms_norm_matches_cpu_oracle() {
    if !is_available(Device::Metal) {
        eprintln!("skip metal_rms_norm (Metal unavailable)");
        return;
    }
    let (rows, inner) = (3usize, 130usize);
    let x = dc_ramp(rows * inner, 29);
    let gamma: Vec<f32> = (0..inner).map(|i| 0.8 + (i as f32) * 0.001).collect();
    let beta: Vec<f32> = (0..inner).map(|i| -0.3 + (i as f32) * 0.002).collect();
    let two = ref_rms_norm_two_pass(&x, rows, inner, &gamma, &beta);
    let mut g = Graph::new("rms_metal");
    let xin = g.input("x", Shape::new(&[rows, inner], F));
    let gin = g.input("gamma", Shape::new(&[inner], F));
    let bin = g.input("beta", Shape::new(&[inner], F));
    let y = g.add_node(
        rlx_ir::Op::RmsNorm { axis: -1, eps: EPS },
        vec![xin, gin, bin],
        Shape::new(&[rows, inner], F),
    );
    g.set_outputs(vec![y]);
    let inputs: &[(&str, &[f32])] = &[
        ("x", x.as_slice()),
        ("gamma", gamma.as_slice()),
        ("beta", beta.as_slice()),
    ];
    let got = run(Device::Metal, g, inputs);
    assert_close(
        "metal RmsNorm inner=130 vs CPU two-pass+beta",
        &got,
        &two,
        3e-4,
    );
}

/// CUDA/ROCm shared `layernorm.cu` RmsNorm branch → two-pass + beta (+ the
/// compile.rs beta-placeholder fix). Same DC + non-zero-beta fixture.
#[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
fn rms_norm_two_pass_beta_case() -> (Graph, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (rows, inner) = (3usize, 130usize);
    let x = dc_ramp(rows * inner, 29);
    let gamma: Vec<f32> = (0..inner).map(|i| 0.8 + (i as f32) * 0.001).collect();
    let beta: Vec<f32> = (0..inner).map(|i| -0.3 + (i as f32) * 0.002).collect();
    let two = ref_rms_norm_two_pass(&x, rows, inner, &gamma, &beta);
    let mut g = Graph::new("rms");
    let xin = g.input("x", Shape::new(&[rows, inner], F));
    let gin = g.input("gamma", Shape::new(&[inner], F));
    let bin = g.input("beta", Shape::new(&[inner], F));
    let y = g.add_node(
        rlx_ir::Op::RmsNorm { axis: -1, eps: EPS },
        vec![xin, gin, bin],
        Shape::new(&[rows, inner], F),
    );
    g.set_outputs(vec![y]);
    (g, x, gamma, beta, two)
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_rms_norm_matches_cpu_oracle() {
    if !is_available(Device::Cuda) {
        return;
    }
    let (g, x, gamma, beta, two) = rms_norm_two_pass_beta_case();
    let inputs: &[(&str, &[f32])] = &[("x", &x), ("gamma", &gamma), ("beta", &beta)];
    let got = run(Device::Cuda, g, inputs);
    assert_close(
        "cuda RmsNorm inner=130 vs CPU two-pass+beta",
        &got,
        &two,
        3e-4,
    );
}

#[test]
#[cfg(feature = "rocm")]
fn rocm_rms_norm_matches_cpu_oracle() {
    if !is_available(Device::Rocm) {
        return;
    }
    let (g, x, gamma, beta, two) = rms_norm_two_pass_beta_case();
    let inputs: &[(&str, &[f32])] = &[("x", &x), ("gamma", &gamma), ("beta", &beta)];
    let got = run(Device::Rocm, g, inputs);
    assert_close(
        "rocm RmsNorm inner=130 vs CPU two-pass+beta",
        &got,
        &two,
        3e-4,
    );
}

#[test]
#[cfg(feature = "vulkan")]
fn vulkan_rms_norm_matches_cpu_oracle() {
    if !is_available(Device::Vulkan) {
        return;
    }
    let (g, x, gamma, beta, two) = rms_norm_two_pass_beta_case();
    let inputs: &[(&str, &[f32])] = &[("x", &x), ("gamma", &gamma), ("beta", &beta)];
    let got = run(Device::Vulkan, g, inputs);
    assert_close(
        "vulkan RmsNorm inner=130 vs CPU two-pass+beta",
        &got,
        &two,
        3e-4,
    );
}

/// LayerNorm was one-pass `E[x²]−mean²` on CPU/Metal/CUDA/ROCm (DC-cancellation);
/// upgraded to two-pass everywhere. Fixture: DC offset + inner>64 (striding).
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    all(feature = "metal", target_os = "macos")
))]
fn layer_norm_dc_case() -> (Graph, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (rows, inner) = (4usize, 130usize);
    let x = dc_ramp(rows * inner, 11);
    let gamma: Vec<f32> = (0..inner).map(|i| 0.8 + (i as f32) * 0.001).collect();
    let beta: Vec<f32> = (0..inner).map(|i| -0.1 + (i as f32) * 0.0005).collect();
    let two = ref_layer_norm_two_pass(&x, rows, inner, &gamma, &beta);
    let mut g = Graph::new("ln_dc");
    let xin = g.input("x", Shape::new(&[rows, inner], F));
    let gin = g.input("gamma", Shape::new(&[inner], F));
    let bin = g.input("beta", Shape::new(&[inner], F));
    let y = g.layer_norm(xin, gin, bin, -1, EPS, Shape::new(&[rows, inner], F));
    g.set_outputs(vec![y]);
    (g, x, gamma, beta, two)
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_layer_norm_dc_offset_matches_two_pass() {
    if !is_available(Device::Metal) {
        return;
    }
    let (g, x, gamma, beta, two) = layer_norm_dc_case();
    let inputs: &[(&str, &[f32])] = &[("x", &x), ("gamma", &gamma), ("beta", &beta)];
    let got = run(Device::Metal, g, inputs);
    assert_close("metal LayerNorm DC vs two-pass", &got, &two, 1e-3);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_layer_norm_dc_offset_matches_two_pass() {
    if !is_available(Device::Cuda) {
        return;
    }
    let (g, x, gamma, beta, two) = layer_norm_dc_case();
    let inputs: &[(&str, &[f32])] = &[("x", &x), ("gamma", &gamma), ("beta", &beta)];
    // msi CPU is x86_64+AVX2 → also validates the AVX CPU two-pass LayerNorm.
    let cpu = run(Device::Cpu, g.clone(), inputs);
    assert_close("cpu(avx) LayerNorm DC vs two-pass", &cpu, &two, 1e-3);
    let got = run(Device::Cuda, g, inputs);
    assert_close("cuda LayerNorm DC vs two-pass", &got, &two, 1e-3);
}

#[test]
#[cfg(feature = "rocm")]
fn rocm_layer_norm_dc_offset_matches_two_pass() {
    if !is_available(Device::Rocm) {
        return;
    }
    let (g, x, gamma, beta, two) = layer_norm_dc_case();
    let inputs: &[(&str, &[f32])] = &[("x", &x), ("gamma", &gamma), ("beta", &beta)];
    let got = run(Device::Rocm, g, inputs);
    assert_close("rocm LayerNorm DC vs two-pass", &got, &two, 1e-3);
}

/// Pin: wgpu last-axis Softmax now uses a workgroup-tree reduction (parallel max
/// + per-thread Kahan exp-sum + thread-0 Kahan merge). Exercises the striding
/// loop (`inner`=130 not a multiple of 64) + a wide value range (max-subtraction
/// + exp-sum matter). Softmax is well-conditioned after the max shift, so a
/// tight tol holds.
#[test]
#[cfg(feature = "gpu")]
fn wgpu_softmax_last_axis_striding_matches_reference() {
    if !is_available(Device::Gpu) {
        eprintln!("skip wgpu_softmax_striding (wgpu unavailable)");
        return;
    }
    let dims = [3usize, 130];
    let n: usize = dims.iter().product();
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 13 % 97) as f32 - 48.0) * 0.2)
        .collect();
    let want = ref_softmax(&x, &dims, 1);
    let mut g = Graph::new("sm_wgpu");
    let xin = g.input("x", Shape::new(&dims, F));
    let y = g.softmax(xin, 1, Shape::new(&dims, F));
    g.set_outputs(vec![y]);
    let got = run(Device::Gpu, g, &[("x", x.as_slice())]);
    assert_close("wgpu softmax last-axis inner=130", &got, &want, 1e-5);
}
