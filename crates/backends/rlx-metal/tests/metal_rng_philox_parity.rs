// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native Metal Philox4×32-10 RNG vs the CPU `rlx_ir::Philox4x32` reference.
//!
//! `Op::RngUniform` / `Op::RngNormal` under `RngBackend::Philox` used to bounce
//! off the GPU (D2H → CPU Philox → H2D). They now dispatch the `rng_*_philox`
//! MSL kernels inline in the compute batch. This test proves parity against the
//! host stream that the CUDA/ROCm rollout also matches:
//!   * unit-uniform `[0,1)` — BIT-EXACT (integer Philox hash + `u32→unit`, no
//!     affine FMA), asserted on `f32::to_bits`.
//!   * affine-uniform `[low,high)` — the GPU fuses `low+u*(high-low)` into one
//!     FMA vs the CPU's two roundings → a benign few-ULP gap.
//!   * normal (Box–Muller `ln`/`sqrt`/`cos`) — GPU transcendentals differ from
//!     libm by a handful of ULPs; bounded by abs-diff, not bit-equality.
//!
//! All cases run in ONE `#[test]` so the independent `Session`s execute serially
//! (matches the other Metal parity harnesses). Exact max-ULP / abs-diff figures
//! are printed under `--nocapture`.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, RngOptions, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

/// `Op::RngUniform` over a fixed `[n]` output, seeded via the per-node `key`.
fn rng_uniform_graph(n: usize, seed_key: u64, low: f32, high: f32) -> Graph {
    let mut g = Graph::new("rng_uniform");
    let template = g.input("template", Shape::new(&[n], DType::F32));
    let out = g.add_node(
        Op::RngUniform {
            low,
            high,
            key: seed_key,
            op_seed: None,
        },
        vec![template],
        Shape::new(&[n], DType::F32),
    );
    g.set_outputs(vec![out]);
    g
}

/// `Op::RngNormal` over a fixed `[n]` output.
fn rng_normal_graph(n: usize, seed_key: u64, mean: f32, scale: f32) -> Graph {
    let mut g = Graph::new("rng_normal");
    let template = g.input("template", Shape::new(&[n], DType::F32));
    let out = g.add_node(
        Op::RngNormal {
            mean,
            scale,
            key: seed_key,
            op_seed: None,
        },
        vec![template],
        Shape::new(&[n], DType::F32),
    );
    g.set_outputs(vec![out]);
    g
}

/// Total-order ULP distance between two f32s (sign-aware; handles the
/// negatives that `RngNormal` / affine-`RngUniform` produce, where a raw
/// `to_bits` abs-diff would be meaningless across the sign bit).
fn ulp_diff(a: f32, b: f32) -> u64 {
    fn mono(x: f32) -> u32 {
        let bits = x.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits // negatives: descend below +0 monotonically
        } else {
            bits | 0x8000_0000
        }
    }
    (mono(a) as i64 - mono(b) as i64).unsigned_abs()
}

fn run(device: Device, g: Graph, seed: u64, n: usize) -> Vec<f32> {
    let opts = CompileOptions::new().rng(RngOptions::philox(seed));
    let template = vec![0f32; n];
    let mut exe = Session::new(device).compile_with(g, &opts);
    exe.run(&[("template", template.as_slice())]).remove(0)
}

#[test]
fn metal_philox_rng_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    // ── 1. unit-uniform [0,1): BIT-EXACT vs CPU across seeds / sizes ──
    // `0 + u*(1-0)` is exact on both sides, so the whole Philox stream (hash,
    // counter/lane layout, `u32→unit`) is exercised bit-for-bit.
    let mut worst_unit_ulp = 0u64;
    for &seed in &[1u64, 42, 123, 0x9E37_79B9] {
        for &n in &[1usize, 3, 4, 5, 17, 64, 257, 4096] {
            let g = rng_uniform_graph(n, seed, 0.0, 1.0);
            let metal = run(Device::Metal, g.clone(), seed, n);
            let cpu = run(Device::Cpu, g, seed, n);
            assert_eq!(metal.len(), n);
            for (i, (&m, &c)) in metal.iter().zip(&cpu).enumerate() {
                let d = ulp_diff(m, c);
                worst_unit_ulp = worst_unit_ulp.max(d);
                assert_eq!(
                    m.to_bits(),
                    c.to_bits(),
                    "unit-uniform not bit-exact seed={seed} n={n} i={i}: \
                     metal {m} ({:#010x}) vs cpu {c} ({:#010x})",
                    m.to_bits(),
                    c.to_bits()
                );
            }
        }
    }
    eprintln!("unit-uniform [0,1): BIT-EXACT vs CPU (max ULP = {worst_unit_ulp})");
    assert_eq!(worst_unit_ulp, 0, "unit-uniform must be bit-exact");

    // ── 2. affine-uniform [low,high): FMA single-rounding → ~1 ULP ──
    // `low + u*(high-low)`: the GPU fuses this into one rounded FMA, the CPU
    // rounds the multiply then the add. The gap is ≤~1 ULP *at the scale of the
    // affine operands* (the `high-low` span). We therefore guard abs-diff against
    // `k · span · f32::EPSILON`, not a raw total-order ULP count — the latter
    // explodes for ranges that cross zero, where cancellation lands the result
    // near 0.0 (tiny value, dense floats) even though the rounding error is fixed
    // at the operand magnitude. The reported total-order ULP shows that effect.
    let n = 4096usize;
    for &(low, high) in &[(-2.0f32, 3.0f32), (0.25, 0.75), (-100.0, 100.0)] {
        let seed = 123u64;
        let g = rng_uniform_graph(n, seed, low, high);
        let metal = run(Device::Metal, g.clone(), seed, n);
        let cpu = run(Device::Cpu, g, seed, n);
        let (mut u, mut a) = (0u64, 0.0f32);
        for (&m, &c) in metal.iter().zip(&cpu) {
            u = u.max(ulp_diff(m, c));
            a = a.max((m - c).abs());
        }
        let span = (high - low).abs();
        let tol = 3.0 * span * f32::EPSILON; // ≤ ~1 FMA ULP at the operand scale
        eprintln!(
            "affine-uniform [{low},{high}): max abs = {a:e} (tol {tol:e}), \
             total-order max ULP = {u}"
        );
        assert!(
            a <= tol,
            "affine-uniform [{low},{high}) abs-diff {a:e} exceeds ~1 span-ULP {tol:e}"
        );
    }

    // ── 3. normal (Box–Muller): transcendental ULPs, bounded by abs-diff ──
    let mut worst_normal_ulp = 0u64;
    let mut worst_normal_abs = 0.0f32;
    for &(mean, scale) in &[(0.0f32, 1.0f32), (0.1, 2.0), (-1.0, 0.5)] {
        let seed = 77u64;
        let g = rng_normal_graph(n, seed, mean, scale);
        let metal = run(Device::Metal, g.clone(), seed, n);
        let cpu = run(Device::Cpu, g, seed, n);
        let (mut u, mut a) = (0u64, 0.0f32);
        for (&m, &c) in metal.iter().zip(&cpu) {
            u = u.max(ulp_diff(m, c));
            a = a.max((m - c).abs());
        }
        eprintln!("normal(mean={mean}, scale={scale}): max ULP = {u}, max abs = {a:e}");
        worst_normal_ulp = worst_normal_ulp.max(u);
        worst_normal_abs = worst_normal_abs.max(a);
    }
    // ULP is dominated by samples that straddle zero (huge ULP, tiny abs), so the
    // meaningful guard is abs-diff — the same bound the ROCm/Metal rollout uses.
    eprintln!("normal overall: max ULP = {worst_normal_ulp}, max abs = {worst_normal_abs:e}");
    assert!(
        worst_normal_abs < 1e-3,
        "normal abs-diff too large: {worst_normal_abs:e}"
    );
}
