// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::SynthMatMul { SynthKind::Codebook }` — on-chip codebook weight-synthesis
//! matmul (single-level vector quantization). Verifies:
//!   1. the native CPU kernel against a hand-written reference, and
//!   2. the portable decompose oracle (Cast + Gather + Reshape + Transpose +
//!      MatMul) reproduces the native CPU result.
//!
//! The weight is stored transposed (`[n, k]`): `indices[j, b]` selects centroid
//! `codebook[indices[j,b]] ∈ ℝ^{entry_dim}`, reconstructing the `entry_dim`
//! weights at `W[j, b·entry_dim ..]`. Output is `y = x · Wᵀ`.

use rlx_ir::*;
use rlx_runtime::{Device, Session};

// The Metal synth tests toggle process-global `RLX_METAL_SYNTH_*` env vars to pick
// the dispatch path; serialize them so a concurrent test never observes another's
// flags. (Default Rust runs test fns in parallel.)
#[cfg(all(target_os = "macos", feature = "metal"))]
static METAL_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// out[i,j] = Σ_b Σ_t x[i, b·d+t] · codebook[indices[j,b], t]
fn reference(
    x: &[f32],
    indices: &[u8],
    codebook: &[f32],
    m: usize,
    k: usize,
    n: usize,
    d: usize,
) -> Vec<f32> {
    let kb = k / d;
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for b in 0..kb {
                let code = indices[j * kb + b] as usize;
                for t in 0..d {
                    acc += x[i * k + b * d + t] * codebook[code * d + t];
                }
            }
            out[i * n + j] = acc;
        }
    }
    out
}

struct Case {
    m: usize,
    k: usize,
    n: usize,
    d: usize,
    entries: usize,
}

fn make_inputs(c: &Case) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
    let kb = c.k / c.d;
    let x: Vec<f32> = (0..c.m * c.k)
        .map(|i| (i as f32 * 0.11).sin() * 1.3)
        .collect();
    // Deterministic codes in [0, entries).
    let indices: Vec<u8> = (0..c.n * kb).map(|i| (i % c.entries) as u8).collect();
    let codebook: Vec<f32> = (0..c.entries * c.d)
        .map(|i| (i as f32 * 0.037).cos() * 1.1)
        .collect();
    (x, indices, codebook)
}

fn build(c: &Case) -> Graph {
    let mut g = Graph::new("synth_matmul");
    let kb = c.k / c.d;
    let x = g.input("x", Shape::new(&[c.m, c.k], DType::F32));
    let indices = g.param("indices", Shape::new(&[c.n, kb], DType::U8));
    let codebook = g.input("codebook", Shape::new(&[c.entries, c.d], DType::F32));
    let y = g.synth_matmul(
        x,
        indices,
        codebook,
        SynthKind::Codebook {
            entry_dim: c.d as u32,
            num_entries: c.entries as u32,
        },
        Shape::new(&[c.m, c.n], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn run_native(c: &Case, x: &[f32], indices: &[u8], codebook: &[f32]) -> Vec<f32> {
    let mut compiled = Session::new(Device::Cpu).compile(build(c));
    compiled.set_param_typed("indices", indices, DType::U8);
    compiled
        .run(&[("x", x), ("codebook", codebook)])
        .into_iter()
        .next()
        .unwrap()
}

fn run_decomposed(c: &Case, x: &[f32], indices: &[u8], codebook: &[f32]) -> Vec<f32> {
    use rlx_fusion::LowerSynthMatMul;
    use rlx_fusion::pass::Pass;

    let lowered = LowerSynthMatMul.run(build(c));
    assert!(
        !lowered
            .nodes()
            .iter()
            .any(|nd| matches!(nd.op, Op::SynthMatMul { .. })),
        "decompose left a SynthMatMul node"
    );
    let mut compiled = Session::new(Device::Cpu).compile(lowered);
    compiled.set_param_typed("indices", indices, DType::U8);
    compiled
        .run(&[("x", x), ("codebook", codebook)])
        .into_iter()
        .next()
        .unwrap()
}

fn cases() -> Vec<Case> {
    vec![
        // GEMV (decode) path: m == 1 → fused reconstruct-in-loop.
        Case {
            m: 1,
            k: 64,
            n: 6,
            d: 4,
            entries: 8,
        },
        // GEMM (prefill) path: m > 1 → reconstruct once + BLAS.
        Case {
            m: 5,
            k: 64,
            n: 6,
            d: 4,
            entries: 8,
        },
        // Large-M path: exercises the Metal `_mm` kernel (M>8, not split-K).
        Case {
            m: 16,
            k: 128,
            n: 10,
            d: 4,
            entries: 8,
        },
        // entry_dim == 1 (scalar palette) edge case.
        Case {
            m: 3,
            k: 32,
            n: 4,
            d: 1,
            entries: 5,
        },
    ]
}

#[test]
fn cpu_native_matches_reference() {
    for c in cases() {
        let (x, indices, codebook) = make_inputs(&c);
        let out = run_native(&c, &x, &indices, &codebook);
        assert_eq!(out.len(), c.m * c.n);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
        let want = reference(&x, &indices, &codebook, c.m, c.k, c.n, c.d);
        let err = max_abs_err(&out, &want);
        eprintln!(
            "synth_matmul native vs reference (m={},d={}): err={err:e}",
            c.m, c.d
        );
        assert!(err < 1e-3, "native diverges from reference: err {err}");
    }
}

#[test]
fn decompose_matches_native() {
    for c in cases() {
        let (x, indices, codebook) = make_inputs(&c);
        let native = run_native(&c, &x, &indices, &codebook);
        let decomp = run_decomposed(&c, &x, &indices, &codebook);
        assert_eq!(native.len(), decomp.len());
        let err = max_abs_err(&native, &decomp);
        eprintln!(
            "synth_matmul decompose vs native (m={},d={}): err={err:e}",
            c.m, c.d
        );
        assert!(err < 1e-3, "decompose diverges from native: err {err}");
    }
}

// Metal now owns SynthMatMul natively (MSL `synth_matmul_codebook`), reading the
// packed U8 indices + f32 codebook directly from the unified-memory arena — so it
// does NOT ride the generic Cast→Gather decompose (which mis-reads U8 params on
// Metal). This validates the native kernel against the CPU reference.
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_matches_cpu() {
    let _g = METAL_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    for c in cases() {
        let (x, indices, codebook) = make_inputs(&c);
        let cpu = run_native(&c, &x, &indices, &codebook);
        let mut compiled = Session::new(Device::Metal).compile(build(&c));
        compiled.set_param_typed("indices", &indices, DType::U8);
        let met = compiled
            .run(&[("x", &x), ("codebook", &codebook)])
            .into_iter()
            .next()
            .unwrap();
        let err = max_abs_err(&cpu, &met);
        eprintln!(
            "synth_matmul metal vs cpu (m={},d={}): err={err:e}",
            c.m, c.d
        );
        assert!(err < 1e-3, "metal diverges from cpu: err {err}");
    }
}

// SynthMatMul BACKWARD on Metal. The VJP reconstructs Wᵀ via `Cast(u8→i64)→Gather`
// (for `dx`) and scatters via `Cast(u8→f32)→ScatterAdd` (for `d_codebook`). Both
// casts read the packed-u8 `indices` param, which lives 1-byte-packed in Metal's
// f32-uniform arena. The `Cast(u8→i64)` fast path (`CastTruncF32`) used to read it
// as f32 (4 B/elem) → garbage indices; now packed sub-4-byte integer sources route
// through the true-width host cast, so the whole backward matches CPU. This makes
// SynthMatMul fully trainable on Metal (like the all-f32 KAN spline).
#[test]
#[cfg(all(target_os = "macos", feature = "metal", feature = "training"))]
fn metal_backward_matches_cpu() {
    use rlx_autodiff::grad_with_loss;
    use rlx_ir::op::{BinaryOp, ReduceOp};

    let _g = METAL_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }

    // ‖y‖² loss over a codebook synth-matmul; diff w.r.t. x AND codebook.
    fn build_backward(c: &Case) -> Graph {
        let kb = c.k / c.d;
        let mut g = Graph::new("synth_bwd");
        let x = g.param("x", Shape::new(&[c.m, c.k], DType::F32));
        let cb = g.param("codebook", Shape::new(&[c.entries, c.d], DType::F32));
        let idx = g.param("indices", Shape::new(&[c.n, kb], DType::U8));
        let y = g.synth_matmul(
            x,
            idx,
            cb,
            SynthKind::Codebook {
                entry_dim: c.d as u32,
                num_entries: c.entries as u32,
            },
            Shape::new(&[c.m, c.n], DType::F32),
        );
        let y2 = g.add_node(
            Op::Binary(BinaryOp::Mul),
            vec![y, y],
            g.node(y).shape.clone(),
        );
        let flat = g.add_node(
            Op::Reshape {
                new_shape: vec![(c.m * c.n) as i64],
            },
            vec![y2],
            Shape::new(&[c.m * c.n], DType::F32),
        );
        let loss = g.add_node(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes: vec![0],
                keep_dim: false,
            },
            vec![flat],
            Shape::from_dims(&[], DType::F32),
        );
        g.set_outputs(vec![loss]);
        grad_with_loss(&g, &[x, cb])
    }

    for c in cases() {
        let (x, indices, codebook) = make_inputs(&c);

        let run = |dev: Device| -> (Vec<f32>, Vec<f32>) {
            let mut compiled = Session::new(dev).compile(build_backward(&c));
            compiled.set_param("x", &x);
            compiled.set_param("codebook", &codebook);
            compiled.set_param_typed("indices", &indices, DType::U8);
            let outs = compiled.run(&[("d_output", &[1.0f32])]); // [loss, dx, d_codebook]
            (outs[1].clone(), outs[2].clone())
        };

        let (dx_cpu, dcb_cpu) = run(Device::Cpu);
        let (dx_met, dcb_met) = run(Device::Metal);
        // Gradients scale with output magnitude (∝ m·k), and the m>8 prefill grad
        // routes through MPS matmul (different fp32 accumulation than CPU sgemm) +
        // scatter-add ordering — so compare RELATIVE to the gradient magnitude, not
        // an absolute bound. Metal is correct: dx matches to ~1e-4, d_codebook ~1e-4
        // relative across all cases.
        let rel_err = |cpu: &[f32], met: &[f32]| {
            let scale = cpu.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
            max_abs_err(cpu, met) / scale
        };
        let dx_err = rel_err(&dx_cpu, &dx_met);
        let dcb_err = rel_err(&dcb_cpu, &dcb_met);
        eprintln!(
            "synth_matmul backward metal vs cpu (m={},d={}): dx_rel={dx_err:e} dcb_rel={dcb_err:e}",
            c.m, c.d
        );
        assert!(
            dx_err < 2e-3,
            "metal dx diverges from cpu: rel err {dx_err}"
        );
        assert!(
            dcb_err < 2e-3,
            "metal d_codebook diverges from cpu: rel err {dcb_err}"
        );
    }
}

// Threadgroup-tiled fused kernel (opt-in RLX_METAL_SYNTH_TILED, m>8 f32 path):
// `simdgroup_float8x8` MMAs with the weight tile reconstructed on-chip. Verifies
// correctness across medium-M shapes incl. M/N/K NOT multiples of 32 (exercises the
// bounds-checked loads + staged store). It's opt-in because it's measured slower
// than the recon→MPS default (see the kernel doc), but must stay bit-correct.
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_tiled_matches_cpu() {
    let _g = METAL_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // Edge cases: m,n,k not multiples of 32; d=1 and d=4.
    let tiled_cases = [
        Case {
            m: 16,
            k: 256,
            n: 128,
            d: 4,
            entries: 8,
        },
        Case {
            m: 48,
            k: 96,
            n: 40,
            d: 4,
            entries: 16,
        }, // all non-mult-of-32
        Case {
            m: 64,
            k: 128,
            n: 256,
            d: 4,
            entries: 32,
        },
        Case {
            m: 33,
            k: 64,
            n: 17,
            d: 1,
            entries: 5,
        }, // d=1, ragged
    ];
    // SAFETY: test-local env toggle; the tiled path is correct regardless, so even a
    // concurrent Metal test that observes the flag stays bit-correct.
    unsafe { std::env::set_var("RLX_METAL_SYNTH_TILED", "1") };
    for c in &tiled_cases {
        let (x, indices, codebook) = make_inputs(c);
        let cpu = run_native(c, &x, &indices, &codebook);
        let mut compiled = Session::new(Device::Metal).compile(build(c));
        compiled.set_param_typed("indices", &indices, DType::U8);
        let met = compiled
            .run(&[("x", &x), ("codebook", &codebook)])
            .into_iter()
            .next()
            .unwrap();
        let err = max_abs_err(&cpu, &met);
        eprintln!(
            "synth_matmul TILED vs cpu (m={},n={},k={},d={}): err={err:e}",
            c.m, c.n, c.k, c.d
        );
        assert!(err < 1e-3, "tiled diverges from cpu: err {err}");
    }
    unsafe { std::env::remove_var("RLX_METAL_SYNTH_TILED") };
}

// f16 ("relaxed precision") tiled kernel: `simdgroup_half8x8` MMAs (f16 inputs, f32
// accumulate) — Apple's matrix-unit fast path. External I/O stays f32, so it's
// checked vs the f32 CPU reference with an f16-appropriate RELATIVE tolerance (f16
// inputs round to ~1e-3; the K-long accumulation stays f32).
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_tiled_f16_matches_cpu() {
    let _g = METAL_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let cases = [
        Case {
            m: 16,
            k: 256,
            n: 128,
            d: 4,
            entries: 8,
        },
        Case {
            m: 64,
            k: 128,
            n: 256,
            d: 4,
            entries: 32,
        },
        Case {
            m: 48,
            k: 96,
            n: 40,
            d: 4,
            entries: 16,
        },
    ];
    unsafe { std::env::set_var("RLX_METAL_SYNTH_TILED", "1") };
    unsafe { std::env::set_var("RLX_METAL_SYNTH_TILED_F16", "1") };
    for c in &cases {
        let (x, indices, codebook) = make_inputs(c);
        let cpu = run_native(c, &x, &indices, &codebook);
        let mut compiled = Session::new(Device::Metal).compile(build(c));
        compiled.set_param_typed("indices", &indices, DType::U8);
        let met = compiled
            .run(&[("x", &x), ("codebook", &codebook)])
            .into_iter()
            .next()
            .unwrap();
        let scale = cpu.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        let rel = max_abs_err(&cpu, &met) / scale;
        eprintln!(
            "synth_matmul TILED_f16 vs cpu (m={},n={},k={},d={}): rel={rel:e}",
            c.m, c.n, c.k, c.d
        );
        assert!(rel < 3e-2, "f16 tiled diverges from cpu: rel {rel}");
    }
    unsafe { std::env::remove_var("RLX_METAL_SYNTH_TILED_F16") };
    unsafe { std::env::remove_var("RLX_METAL_SYNTH_TILED") };
}

// Opt-in f16 reconstruct → MPS-f16 prefill path (RLX_METAL_SYNTH_RECON_F16): the m>8
// recon→MPS default, but casting x→f16 + reconstructing the weight in f16 (half the
// scratch) + MPS hgemm + casting the result back to f32. ~1.3× + 2× smaller scratch,
// relaxed precision — checked vs the f32 CPU reference with an f16 tolerance.
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_recon_f16_matches_cpu() {
    let _g = METAL_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // m>8 → the recon→MPS path (with the flag, its f16 variant).
    let cases = [
        Case {
            m: 16,
            k: 128,
            n: 64,
            d: 4,
            entries: 8,
        },
        Case {
            m: 64,
            k: 256,
            n: 128,
            d: 4,
            entries: 16,
        },
    ];
    unsafe { std::env::set_var("RLX_METAL_SYNTH_RECON_F16", "1") };
    for c in &cases {
        let (x, indices, codebook) = make_inputs(c);
        let cpu = run_native(c, &x, &indices, &codebook);
        let mut compiled = Session::new(Device::Metal).compile(build(c));
        compiled.set_param_typed("indices", &indices, DType::U8);
        let met = compiled
            .run(&[("x", &x), ("codebook", &codebook)])
            .into_iter()
            .next()
            .unwrap();
        let scale = cpu.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        let rel = max_abs_err(&cpu, &met) / scale;
        eprintln!(
            "synth_matmul RECON_F16 vs cpu (m={},n={},k={}): rel={rel:e}",
            c.m, c.n, c.k
        );
        assert!(rel < 3e-2, "f16 recon diverges from cpu: rel {rel}");
    }
    unsafe { std::env::remove_var("RLX_METAL_SYNTH_RECON_F16") };
}

// FINDING (measured: err ≈ 2.69): wgpu fails SynthMatMul via the decompose with
// the SAME u8-index Cast→Gather failure as Metal — its f32-uniform arena keeps
// u8 params packed, so the generic Cast/Gather reads garbage indices. So wgpu
// (and Vulkan) need a NATIVE SynthMatMul kernel like Metal; only CUDA/ROCm
// (native-dtype arenas) get reconstruct→GEMM for free via the decompose. This
// probe is kept (ignored) as the regression that a native wgpu kernel must fix.
#[test]
#[ignore = "wgpu f32-uniform arena can't decompose a u8-indexed op; needs a native kernel (like Metal)"]
#[cfg(feature = "webgpu")]
fn wgpu_matches_cpu() {
    if !rlx_runtime::is_available(Device::WebGpu) {
        return;
    }
    for c in cases() {
        let (x, indices, codebook) = make_inputs(&c);
        let cpu = run_native(&c, &x, &indices, &codebook);
        let mut compiled = Session::new(Device::WebGpu).compile(build(&c));
        compiled.set_param_typed("indices", &indices, DType::U8);
        let wg = compiled
            .run(&[("x", &x), ("codebook", &codebook)])
            .into_iter()
            .next()
            .unwrap();
        let err = max_abs_err(&cpu, &wg);
        eprintln!(
            "synth_matmul wgpu vs cpu (m={},d={}): err={err:e}",
            c.m, c.d
        );
        assert!(err < 1e-3, "wgpu diverges from cpu: err {err}");
    }
}
