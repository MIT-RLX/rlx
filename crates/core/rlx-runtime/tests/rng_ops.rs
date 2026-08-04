// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! In-graph RNG ops: compile, execute, and runtime policy override.

use rlx_ir::{DType, Graph, Op, RngOptions, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

fn rng_normal_graph(seed_key: u64) -> Graph {
    let mut g = Graph::new("rng_normal");
    let template = g.input("template", Shape::new(&[2, 3], DType::F32));
    let out = g.add_node(
        Op::RngNormal {
            mean: 0.1,
            scale: 2.0,
            key: seed_key,
            op_seed: Some(7.0),
        },
        vec![template],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![out]);
    g
}

#[test]
fn rng_normal_philox_is_deterministic() {
    let g = rng_normal_graph(1);
    let opts = CompileOptions::new().rng(RngOptions::philox(99));
    let mut exe = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    let template = vec![0f32; 6];
    let a = exe.run(&[("template", &template)]).remove(0);
    let b = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(a, b);
    assert_ne!(a, template);
}

#[test]
fn rng_zero_backend_matches_template_shape() {
    let g = rng_normal_graph(2);
    let opts = CompileOptions::new().rng(RngOptions::zero());
    let mut exe = Session::new(Device::Cpu).compile_with(g, &opts);
    let template = vec![1f32; 6];
    let out = exe.run(&[("template", &template)]).remove(0);
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn set_rng_changes_output_without_recompile() {
    let g = rng_normal_graph(3);
    let opts = CompileOptions::new().rng(RngOptions::philox(1));
    let mut exe = Session::new(Device::Cpu).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let philox = exe.run(&[("template", &template)]).remove(0);
    exe.set_rng(RngOptions::zero());
    let zero = exe.run(&[("template", &template)]).remove(0);
    assert!(zero.iter().all(|&v| v == 0.0));
    exe.set_rng(RngOptions::philox(1));
    let philox_again = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(philox, philox_again);
    assert_ne!(philox, zero);
}

#[test]
fn rng_backend_switch_via_compile_options() {
    let g = rng_normal_graph(4);
    let template = vec![0f32; 6];
    let mut ort = Session::new(Device::Cpu)
        .compile_with(g.clone(), &CompileOptions::new().rng(RngOptions::ort(7)));
    let mut philox = Session::new(Device::Cpu)
        .compile_with(g, &CompileOptions::new().rng(RngOptions::philox(7)));
    let a = ort.run(&[("template", &template)]).remove(0);
    let b = philox.run(&[("template", &template)]).remove(0);
    assert_ne!(a, b);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn rng_normal_philox_is_deterministic_metal() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let g = rng_normal_graph(5);
    let opts = CompileOptions::new().rng(RngOptions::philox(99));
    let mut exe = Session::new(Device::Metal).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let a = exe.run(&[("template", &template)]).remove(0);
    let b = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(a, b);
    assert_ne!(a, template);
}

// ── Metal native Philox RNG ─────────────────────────────────────────────
// Metal is unified memory, so Philox/Zero now dispatch an MSL kernel inline in
// the compute batch (no commit/sync/CPU-fill); Ort/Bnns still host-fill.

// Unit range → bit-exact vs CPU (integer Philox + exact `u32→unit`, no FMA).
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn rng_uniform_cpu_metal_bit_parity() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let opts = CompileOptions::new().rng(RngOptions::philox(123));
    let template = vec![0f32; 20];
    let mut cpu = Session::new(Device::Cpu).compile_with(rng_uniform_graph(9, 0.0, 1.0), &opts);
    let mut metal = Session::new(Device::Metal).compile_with(rng_uniform_graph(9, 0.0, 1.0), &opts);
    let a = cpu.run(&[("template", &template)]).remove(0);
    let b = metal.run(&[("template", &template)]).remove(0);
    assert_eq!(
        a, b,
        "Metal Philox unit-uniform must bit-match the CPU stream"
    );
}

// Normal (Box–Muller ln/cos/sqrt): GPU vs CPU differ by transcendental ULPs.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn rng_normal_cpu_metal_close_parity() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let opts = CompileOptions::new().rng(RngOptions::philox(77));
    let template = vec![0f32; 6];
    let mut cpu = Session::new(Device::Cpu).compile_with(rng_normal_graph(21), &opts);
    let mut metal = Session::new(Device::Metal).compile_with(rng_normal_graph(21), &opts);
    let a = cpu.run(&[("template", &template)]).remove(0);
    let b = metal.run(&[("template", &template)]).remove(0);
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-3, "Metal Philox normal vs CPU max abs diff {max}");
}

// Native Philox → native Zero (rng_fill_zero MSL) → Philox, driven by set_rng.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn set_rng_switches_native_backend_metal() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let g = rng_normal_graph(14);
    let opts = CompileOptions::new().rng(RngOptions::philox(5));
    let mut exe = Session::new(Device::Metal).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let philox = exe.run(&[("template", &template)]).remove(0);
    exe.set_rng(RngOptions::zero());
    let zero = exe.run(&[("template", &template)]).remove(0);
    assert!(zero.iter().all(|&v| v == 0.0), "zero fill: {zero:?}");
    exe.set_rng(RngOptions::philox(5));
    let again = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(philox, again);
    assert_ne!(philox, zero);
}

// Ort backend must still route to the unified-memory host fill (bit-exact CPU).
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn rng_normal_ort_host_parity_metal() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let opts = CompileOptions::new().rng(RngOptions::ort(7));
    let template = vec![0f32; 6];
    let mut cpu = Session::new(Device::Cpu).compile_with(rng_normal_graph(31), &opts);
    let mut metal = Session::new(Device::Metal).compile_with(rng_normal_graph(31), &opts);
    let a = cpu.run(&[("template", &template)]).remove(0);
    let b = metal.run(&[("template", &template)]).remove(0);
    assert_eq!(a, b, "Metal Ort host fill must match CPU exactly");
}

#[cfg(feature = "gpu")]
#[test]
fn rng_normal_philox_is_deterministic_wgpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        return;
    }
    let g = rng_normal_graph(7);
    let opts = CompileOptions::new().rng(RngOptions::philox(99));
    let mut exe = Session::new(Device::Gpu).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let a = exe.run(&[("template", &template)]).remove(0);
    let b = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(a, b);
    assert_ne!(a, template);
}

// ── wgpu native Philox RNG ──────────────────────────────────────────────
// Philox/Zero now run a one-shot WGSL compute dispatch (WGSL has no u64, so the
// Philox 32×32→64 mul is emulated) writing the arena on-GPU — no D2H→CPU→H2D.

// Unit range → bit-exact vs CPU (integer Philox stream, no FMA-affected affine).
#[cfg(feature = "gpu")]
#[test]
fn rng_uniform_cpu_wgpu_bit_parity() {
    if !rlx_runtime::is_available(Device::Gpu) {
        return;
    }
    let opts = CompileOptions::new().rng(RngOptions::philox(123));
    let template = vec![0f32; 20];
    let mut cpu = Session::new(Device::Cpu).compile_with(rng_uniform_graph(9, 0.0, 1.0), &opts);
    let mut gpu = Session::new(Device::Gpu).compile_with(rng_uniform_graph(9, 0.0, 1.0), &opts);
    let a = cpu.run(&[("template", &template)]).remove(0);
    let b = gpu.run(&[("template", &template)]).remove(0);
    assert_eq!(
        a, b,
        "wgpu Philox unit-uniform must bit-match the CPU stream"
    );
}

// Normal (Box–Muller): GPU vs CPU differ by transcendental ULPs.
#[cfg(feature = "gpu")]
#[test]
fn rng_normal_cpu_wgpu_close_parity() {
    if !rlx_runtime::is_available(Device::Gpu) {
        return;
    }
    let opts = CompileOptions::new().rng(RngOptions::philox(77));
    let template = vec![0f32; 6];
    let mut cpu = Session::new(Device::Cpu).compile_with(rng_normal_graph(21), &opts);
    let mut gpu = Session::new(Device::Gpu).compile_with(rng_normal_graph(21), &opts);
    let a = cpu.run(&[("template", &template)]).remove(0);
    let b = gpu.run(&[("template", &template)]).remove(0);
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-3, "wgpu Philox normal vs CPU max abs diff {max}");
}

// Native Philox → native Zero (rng_fill_zero WGSL) → Philox, driven by set_rng.
#[cfg(feature = "gpu")]
#[test]
fn set_rng_switches_native_backend_wgpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        return;
    }
    let g = rng_normal_graph(14);
    let opts = CompileOptions::new().rng(RngOptions::philox(5));
    let mut exe = Session::new(Device::Gpu).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let philox = exe.run(&[("template", &template)]).remove(0);
    exe.set_rng(RngOptions::zero());
    let zero = exe.run(&[("template", &template)]).remove(0);
    assert!(zero.iter().all(|&v| v == 0.0), "zero fill: {zero:?}");
    exe.set_rng(RngOptions::philox(5));
    let again = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(philox, again);
    assert_ne!(philox, zero);
}

// ── CUDA graph capture × RNG ────────────────────────────────────────────
// On-device Philox / Zero RNG is now graph-capture-safe (previously ANY RNG
// step disabled CUDA-Graph capture for the whole schedule). Run these under
// `RLX_CUDA_EXEC_MODE=graph` to exercise the capture + replay + invalidation
// path; both assertions are also correct under eager dispatch.

#[cfg(feature = "cuda")]
#[test]
fn rng_normal_philox_is_deterministic_cuda() {
    if !rlx_runtime::is_available(Device::Cuda) {
        return;
    }
    let g = rng_normal_graph(11);
    let opts = CompileOptions::new().rng(RngOptions::philox(99));
    let mut exe = Session::new(Device::Cuda).compile_with(g, &opts);
    let template = vec![0f32; 6];
    // 3 runs: capture on run 1, replay on 2 & 3 under graph mode.
    let a = exe.run(&[("template", &template)]).remove(0);
    let b = exe.run(&[("template", &template)]).remove(0);
    let c = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(a, b);
    assert_eq!(b, c);
    assert_ne!(a, template);
}

// The precise regression guard for this change: under graph mode the Philox
// fill is captured; `set_rng` must drop that capture, or a stale Philox graph
// would replay in place of the zero fill.
#[cfg(feature = "cuda")]
#[test]
fn rng_graph_capture_invalidates_on_set_rng_cuda() {
    if !rlx_runtime::is_available(Device::Cuda) {
        return;
    }
    let g = rng_normal_graph(12);
    let opts = CompileOptions::new().rng(RngOptions::philox(5));
    let mut exe = Session::new(Device::Cuda).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let philox = exe.run(&[("template", &template)]).remove(0); // captures
    let philox2 = exe.run(&[("template", &template)]).remove(0); // replays
    assert_eq!(philox, philox2);

    exe.set_rng(RngOptions::zero()); // must drop the captured Philox graph
    let zero = exe.run(&[("template", &template)]).remove(0);
    assert!(
        zero.iter().all(|&v| v == 0.0),
        "stale captured Philox graph leaked into the zero fill: {zero:?}"
    );

    exe.set_rng(RngOptions::philox(5)); // drops the zero capture, recaptures Philox
    let philox_again = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(philox, philox_again);
    assert_ne!(philox, zero);
}

// ── ROCm native Philox RNG ──────────────────────────────────────────────
// ROCm now fills Philox / Zero RNG on-device via the shared `rng_philox.cu`
// (bit-matched to `rlx_ir::Philox4x32`), replacing the D2H→CPU→H2D host
// bubble and making RNG hipGraph-capture-safe.

// `low`/`high` parameterized: a `[0,1)` range makes the affine map an identity
// (`0 + u*1 == u`, exact on CPU and GPU), isolating the Philox integer stream;
// a general range fuses `low + u*(high-low)` into a single-rounding FMA on the
// GPU (vs two roundings on CPU x86) → a benign ~1-ULP difference.
#[cfg(any(
    feature = "rocm",
    feature = "gpu",
    all(feature = "metal", target_os = "macos")
))]
fn rng_uniform_graph(seed_key: u64, low: f32, high: f32) -> Graph {
    let mut g = Graph::new("rng_uniform");
    let template = g.input("template", Shape::new(&[4, 5], DType::F32));
    let out = g.add_node(
        Op::RngUniform {
            low,
            high,
            key: seed_key,
            op_seed: None,
        },
        vec![template],
        Shape::new(&[4, 5], DType::F32),
    );
    g.set_outputs(vec![out]);
    g
}

// Definitive kernel-correctness proof: on the unit range the uniform stream is
// integer-exact (Philox hash + counter layout + `u32→unit` with no FMA-affected
// affine), so it must BIT-MATCH the CPU `Philox4x32` stream.
#[cfg(feature = "rocm")]
#[test]
fn rng_uniform_cpu_rocm_bit_parity() {
    if !rlx_runtime::is_available(Device::Rocm) {
        return;
    }
    let opts = CompileOptions::new().rng(RngOptions::philox(123));
    let template = vec![0f32; 20];
    let mut cpu = Session::new(Device::Cpu).compile_with(rng_uniform_graph(9, 0.0, 1.0), &opts);
    let mut rocm = Session::new(Device::Rocm).compile_with(rng_uniform_graph(9, 0.0, 1.0), &opts);
    let a = cpu.run(&[("template", &template)]).remove(0);
    let b = rocm.run(&[("template", &template)]).remove(0);
    assert_eq!(
        a, b,
        "ROCm Philox unit-uniform must bit-match the CPU stream"
    );
}

// General range: identical Philox stream, but the GPU FMA on the affine map
// rounds once vs the CPU's twice → allow a few ULPs (still catches real bugs).
#[cfg(feature = "rocm")]
#[test]
fn rng_uniform_general_range_close_parity_rocm() {
    if !rlx_runtime::is_available(Device::Rocm) {
        return;
    }
    let opts = CompileOptions::new().rng(RngOptions::philox(123));
    let template = vec![0f32; 20];
    let mut cpu = Session::new(Device::Cpu).compile_with(rng_uniform_graph(9, -2.0, 3.0), &opts);
    let mut rocm = Session::new(Device::Rocm).compile_with(rng_uniform_graph(9, -2.0, 3.0), &opts);
    let a = cpu.run(&[("template", &template)]).remove(0);
    let b = rocm.run(&[("template", &template)]).remove(0);
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-5, "ROCm uniform affine vs CPU max abs diff {max}");
}

// Normal uses Box–Muller (ln/cos/sqrt), so GPU vs CPU differ by transcendental
// ULPs — require closeness, not bit-equality. Catches algorithmic mistakes.
#[cfg(feature = "rocm")]
#[test]
fn rng_normal_cpu_rocm_close_parity() {
    if !rlx_runtime::is_available(Device::Rocm) {
        return;
    }
    let opts = CompileOptions::new().rng(RngOptions::philox(77));
    let template = vec![0f32; 6];
    let mut cpu = Session::new(Device::Cpu).compile_with(rng_normal_graph(21), &opts);
    let mut rocm = Session::new(Device::Rocm).compile_with(rng_normal_graph(21), &opts);
    let a = cpu.run(&[("template", &template)]).remove(0);
    let b = rocm.run(&[("template", &template)]).remove(0);
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-3, "ROCm Philox normal vs CPU max abs diff {max}");
}

// ROCm runs via `ExecMode::Stream` through the `Session` API (hipGraph capture
// is reached only through a direct `compile_with(ExecMode::Graph)`), so this
// exercises the on-device dispatch + runtime policy switch — native Philox
// normal, native Zero, then Philox again. The capture-safety discriminant
// itself is unit-tested in `rlx-rocm` (`graph_capture_tests`).
#[cfg(feature = "rocm")]
#[test]
fn set_rng_switches_native_backend_rocm() {
    if !rlx_runtime::is_available(Device::Rocm) {
        return;
    }
    let g = rng_normal_graph(13);
    let opts = CompileOptions::new().rng(RngOptions::philox(5));
    let mut exe = Session::new(Device::Rocm).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let philox = exe.run(&[("template", &template)]).remove(0);
    exe.set_rng(RngOptions::zero()); // native rng_fill_zero kernel
    let zero = exe.run(&[("template", &template)]).remove(0);
    assert!(zero.iter().all(|&v| v == 0.0), "zero fill: {zero:?}");
    exe.set_rng(RngOptions::philox(5));
    let again = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(philox, again);
    assert_ne!(philox, zero);
}
