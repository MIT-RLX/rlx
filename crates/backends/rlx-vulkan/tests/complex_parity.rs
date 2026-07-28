// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-Vulkan parity for on-device complex simulation on the f32-uniform arena.
//!
//! Representation (fixed): `C64 = 2 f32 lanes [re, im]`;
//! `C128 = 4 f32 lanes df64 [re_hi, re_lo, im_hi, im_lo]`. Casts from f32 real
//! sources have `lo = 0`, so every complex cast is a pure lane MOVE (bit-exact).
//! C64 arithmetic (add/sub/mul/div) is a standalone `binary_c64` dispatch that
//! reads BOTH lanes per element (it cannot ride the fused scalar-per-thread
//! region path). C128 *arithmetic* is out of scope (rlx-cpu has none either).
//!
//! Gates:
//!   1. Cast parity — real↔C64, real↔C128, C64↔C128 bit-exact vs rlx-cpu.
//!   2. C64 arithmetic — Add/Sub bit-exact; Mul/Div ≤1e-6 rel; scalar broadcast.
//!   3. df64 boundary round-trip — host f64 → widen(split) → C128 → narrow
//!      (combine) → host f64 (π, 1/3, f32-exact); rel err ≤ 1e-14.
//!   4. Slot-sizing — a C64/C128 output reads back 2N / 4N f32 lanes (8N / 16N B).
//!
//! Runs only when a Vulkan device is present; otherwise a graceful no-op.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::backend::{narrow_f32_to_bytes, widen_bytes_to_f32};
use rlx_vulkan::backend::VulkanExecutable;
use std::sync::{Mutex, MutexGuard, OnceLock};

fn available() -> bool {
    rlx_vulkan::is_available()
}

/// Serialize device use across the parallel test threads. rlx-vulkan submits to
/// a process-global `VkQueue`, and Vulkan requires all submissions to a single
/// queue to be *externally synchronized*; the default multi-threaded test runner
/// otherwise races concurrent executables on that shared queue and intermittently
/// reads back zeros. That is a harness artifact — every gate below is bit-exact
/// single-threaded on both native (NVIDIA) and portability (MoltenVK) drivers.
/// The guard recovers a poisoned lock (`into_inner`) so one test's assertion
/// failure doesn't cascade into spurious failures in the others.
fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

// ── host-byte builders ────────────────────────────────────────────────────

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
/// C64 host bytes from interleaved `[re, im]` f32 pairs (`comps.len() == 2N`).
fn c64_bytes(comps: &[f32]) -> Vec<u8> {
    comps.iter().flat_map(|x| x.to_le_bytes()).collect()
}
/// C128 host bytes from interleaved `[re, im]` f64 pairs (`comps.len() == 2N`).
fn c128_bytes(comps: &[f64]) -> Vec<u8> {
    comps.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ── golden (rlx-cpu) + candidate (Vulkan) runners ─────────────────────────

/// rlx-cpu reference: `Cast(src_dt → dst_dt)`, returns native output bytes.
fn cpu_cast_bytes(n: usize, src_dt: DType, dst_dt: DType, input_bytes: &[u8]) -> Vec<u8> {
    use rlx::prelude::*;
    let mut g = Graph::new("ccast_ref");
    let x = g.input("x", Shape::new(&[n], src_dt));
    let y = g.add_node(Op::Cast { to: dst_dt }, vec![x], Shape::new(&[n], dst_dt));
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.run_typed(&[("x", input_bytes, src_dt)]).remove(0).0
}

/// Vulkan candidate: `Cast(src_dt → dst_dt)`. Feeds f32 lanes (widened from host
/// bytes via the shared boundary) and re-narrows the output lanes to host bytes,
/// so the comparison is in the same host representation rlx-cpu emits.
fn vulkan_cast_bytes(
    n: usize,
    src_dt: DType,
    dst_dt: DType,
    input_bytes: &[u8],
) -> (Vec<u8>, usize) {
    let lanes_in = widen_bytes_to_f32(input_bytes, src_dt);
    let mut g = Graph::new("ccast_vk");
    let x = g.input("x", Shape::new(&[n], src_dt));
    let y = g.add_node(Op::Cast { to: dst_dt }, vec![x], Shape::new(&[n], dst_dt));
    g.set_outputs(vec![y]);
    let mut exe = VulkanExecutable::compile(g);
    let lanes_out = exe.run(&[("x", &lanes_in)]).remove(0);
    let lane_count = lanes_out.len();
    (narrow_f32_to_bytes(&lanes_out, dst_dt), lane_count)
}

/// One cast direction, asserted bit-exact Vulkan-vs-cpu in host bytes.
fn assert_cast_parity(n: usize, src_dt: DType, dst_dt: DType, input_bytes: &[u8]) {
    let cpu = cpu_cast_bytes(n, src_dt, dst_dt, input_bytes);
    let (gpu, _) = vulkan_cast_bytes(n, src_dt, dst_dt, input_bytes);
    assert_eq!(
        gpu, cpu,
        "cast {src_dt:?}->{dst_dt:?} vulkan vs cpu byte mismatch"
    );
}

/// rlx-cpu reference: C64 `binary(op)` (possibly broadcasting), output lanes.
fn cpu_binary_c64_lanes(
    n: usize,
    na: usize,
    nb: usize,
    op: BinaryOp,
    a_bytes: &[u8],
    b_bytes: &[u8],
) -> Vec<f32> {
    use rlx::prelude::*;
    let mut g = Graph::new("cbin_ref");
    let a = g.input("a", Shape::new(&[na], DType::C64));
    let b = g.input("b", Shape::new(&[nb], DType::C64));
    let y = g.add_node(Op::Binary(op), vec![a, b], Shape::new(&[n], DType::C64));
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let out = c
        .run_typed(&[("a", a_bytes, DType::C64), ("b", b_bytes, DType::C64)])
        .remove(0)
        .0;
    // Native C64 bytes → f32 lanes (reinterpret, 2N lanes).
    out.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Vulkan candidate: C64 `binary(op)`, returns output lanes (2N).
fn vulkan_binary_c64_lanes(
    n: usize,
    na: usize,
    nb: usize,
    op: BinaryOp,
    a_bytes: &[u8],
    b_bytes: &[u8],
) -> Vec<f32> {
    let a_lanes = widen_bytes_to_f32(a_bytes, DType::C64);
    let b_lanes = widen_bytes_to_f32(b_bytes, DType::C64);
    let mut g = Graph::new("cbin_vk");
    let a = g.input("a", Shape::new(&[na], DType::C64));
    let b = g.input("b", Shape::new(&[nb], DType::C64));
    let y = g.add_node(Op::Binary(op), vec![a, b], Shape::new(&[n], DType::C64));
    g.set_outputs(vec![y]);
    let mut exe = VulkanExecutable::compile(g);
    exe.run(&[("a", &a_lanes), ("b", &b_lanes)]).remove(0)
}

// ── Gate 1: cast parity ───────────────────────────────────────────────────

#[test]
fn cast_real_to_c64_and_back() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping real↔C64");
        return;
    }
    let _g = gpu_lock();
    let reals = [1.5f32, -2.25, 3.0, 0.0, -7.75, 42.0];
    // real → C64
    assert_cast_parity(reals.len(), DType::F32, DType::C64, &f32_bytes(&reals));
    // C64 → real (imaginary parts dropped): 3 complex elements.
    let comps = [1.5f32, 2.5, -3.0, 4.0, 0.0, -1.0];
    assert_cast_parity(comps.len() / 2, DType::C64, DType::F32, &c64_bytes(&comps));
}

#[test]
fn cast_real_to_c128_and_back() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping real↔C128");
        return;
    }
    let _g = gpu_lock();
    let reals = [1.5f32, -2.25, 3.0, 0.0, -7.75, 42.0];
    // real → C128 (lo lanes = 0, exact).
    assert_cast_parity(reals.len(), DType::F32, DType::C128, &f32_bytes(&reals));
    // C128 → real (real hi lane kept). f32-exact f64 components round-trip.
    let comps = [1.5f64, 2.5, -3.0, 4.0, 0.0, -1.0];
    assert_cast_parity(
        comps.len() / 2,
        DType::C128,
        DType::F32,
        &c128_bytes(&comps),
    );
}

#[test]
fn cast_c64_c128_both_ways() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping C64↔C128");
        return;
    }
    let _g = gpu_lock();
    // C64 → C128 widen (2 complex elements): re/im lanes preserved, lo=0.
    let c64 = [1.5f32, -2.5, 3.25, -4.75];
    assert_cast_parity(c64.len() / 2, DType::C64, DType::C128, &c64_bytes(&c64));
    // C128 → C64 narrow: hi lanes kept (f32-exact components stay exact).
    let c128 = [1.5f64, -2.5, 3.25, -4.75];
    assert_cast_parity(c128.len() / 2, DType::C128, DType::C64, &c128_bytes(&c128));
}

// ── Gate 2: C64 arithmetic ────────────────────────────────────────────────

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "lane count mismatch: {a:?} vs {b:?}");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
/// Max complex relative error over N elements (lanes `[re, im]`).
fn max_rel_complex(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut m = 0.0f32;
    for k in 0..a.len() / 2 {
        let (ar, ai) = (a[2 * k], a[2 * k + 1]);
        let (br, bi) = (b[2 * k], b[2 * k + 1]);
        let dmag = ((ar - br).powi(2) + (ai - bi).powi(2)).sqrt();
        let ref_mag = (br * br + bi * bi).sqrt().max(1e-12);
        m = m.max(dmag / ref_mag);
    }
    m
}

#[test]
fn c64_add_sub_bit_exact() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping C64 add/sub");
        return;
    }
    let _g = gpu_lock();
    let a = [1.5f32, 2.5, -3.0, 4.0, 0.5, -0.25];
    let b = [-1.0f32, 0.75, 2.0, -6.5, 3.0, 1.25];
    let n = a.len() / 2;
    for op in [BinaryOp::Add, BinaryOp::Sub] {
        let cpu = cpu_binary_c64_lanes(n, n, n, op, &c64_bytes(&a), &c64_bytes(&b));
        let gpu = vulkan_binary_c64_lanes(n, n, n, op, &c64_bytes(&a), &c64_bytes(&b));
        assert_eq!(gpu, cpu, "C64 {op:?} must be bit-exact vulkan vs cpu");
    }
}

#[test]
fn c64_mul_div_close() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping C64 mul/div");
        return;
    }
    let _g = gpu_lock();
    let a = [1.5f32, 2.5, -3.0, 4.0, 0.5, -0.25];
    let b = [-1.0f32, 0.75, 2.0, -6.5, 3.0, 1.25];
    let n = a.len() / 2;
    for op in [BinaryOp::Mul, BinaryOp::Div] {
        let cpu = cpu_binary_c64_lanes(n, n, n, op, &c64_bytes(&a), &c64_bytes(&b));
        let gpu = vulkan_binary_c64_lanes(n, n, n, op, &c64_bytes(&a), &c64_bytes(&b));
        let rel = max_rel_complex(&gpu, &cpu);
        assert!(
            rel <= 1e-6,
            "C64 {op:?} rel err {rel} > 1e-6\ncpu={cpu:?}\ngpu={gpu:?}"
        );
    }
}

#[test]
fn c64_scalar_vector_broadcast_mul() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping C64 broadcast");
        return;
    }
    let _g = gpu_lock();
    // scalar (1 complex elem) × vector (3 complex elems) → 3.
    let scalar = [2.0f32, -1.0]; // (2 - i)
    let vec = [1.0f32, 1.0, -2.0, 0.5, 0.0, 3.0];
    let n = 3;
    let cpu = cpu_binary_c64_lanes(
        n,
        1,
        n,
        BinaryOp::Mul,
        &c64_bytes(&scalar),
        &c64_bytes(&vec),
    );
    let gpu = vulkan_binary_c64_lanes(
        n,
        1,
        n,
        BinaryOp::Mul,
        &c64_bytes(&scalar),
        &c64_bytes(&vec),
    );
    // Mul is not bit-exact (FMA/order), but a scalar×vector has small enough
    // magnitudes here that both back-ends agree closely.
    let rel = max_rel_complex(&gpu, &cpu);
    assert!(
        rel <= 1e-6,
        "C64 broadcast mul rel err {rel} > 1e-6\ncpu={cpu:?}\ngpu={gpu:?}"
    );
    // Sanity: element 0 = (2-i)(1+i) = 3 + i.
    assert!(
        max_abs(&gpu[0..2], &[3.0, 1.0]) < 1e-5,
        "broadcast mul elem0 {:?}",
        &gpu[0..2]
    );
}

// ── Gate 3: df64 boundary round-trip ──────────────────────────────────────

#[test]
fn df64_boundary_round_trip() {
    // Pure host-boundary check (no device needed): host f64 → widen(SPLIT) →
    // C128 lanes → narrow(COMBINE) → host f64. Sweeps π, 1/3, and exactly
    // f32-representable values (must round-trip bit-exact).
    let reals = [
        std::f64::consts::PI,
        1.0 / 3.0,
        0.5,
        -2.25,
        1.0e-9,
        987654.321012345,
    ];
    let mut host = Vec::new();
    for &r in &reals {
        host.extend_from_slice(&r.to_le_bytes());
        host.extend_from_slice(&(-r * 0.5).to_le_bytes());
    }
    let lanes = widen_bytes_to_f32(&host, DType::C128);
    assert_eq!(lanes.len(), reals.len() * 4);
    let back = narrow_f32_to_bytes(&lanes, DType::C128);
    assert_eq!(back.len(), host.len());
    for (k, &r) in reals.iter().enumerate() {
        let re = f64::from_le_bytes(back[k * 16..k * 16 + 8].try_into().unwrap());
        let im = f64::from_le_bytes(back[k * 16 + 8..k * 16 + 16].try_into().unwrap());
        let rel = |a: f64, b: f64| {
            if b == 0.0 {
                a.abs()
            } else {
                (a - b).abs() / b.abs()
            }
        };
        assert!(rel(re, r) <= 1e-14, "re[{k}]={re} vs {r}");
        assert!(rel(im, -r * 0.5) <= 1e-14, "im[{k}]={im} vs {}", -r * 0.5);
    }
    // f32-exact values are bit-exact through the boundary.
    for &exact in &[0.5f64, -2.25, 3.0] {
        let mut h = Vec::new();
        h.extend_from_slice(&exact.to_le_bytes());
        h.extend_from_slice(&0.0f64.to_le_bytes());
        let l = widen_bytes_to_f32(&h, DType::C128);
        assert_eq!(
            narrow_f32_to_bytes(&l, DType::C128),
            h,
            "f32-exact {exact} not bit-exact"
        );
    }
}

// ── Gate 4: slot sizing ───────────────────────────────────────────────────

#[test]
fn slot_sizing_c64_c128() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping slot sizing");
        return;
    }
    let _g = gpu_lock();
    let reals = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let n = reals.len();
    // C64 output slot must read back 2N lanes (8N bytes) — a `elems * 4` slot
    // sizing would truncate it to N (the imaginary lanes would be dropped).
    let (_, c64_lanes) = vulkan_cast_bytes(n, DType::F32, DType::C64, &f32_bytes(&reals));
    assert_eq!(
        c64_lanes,
        2 * n,
        "C64[{n}] must read back 2N f32 lanes (8N bytes)"
    );
    // C128 output slot must read back 4N lanes (16N bytes).
    let (_, c128_lanes) = vulkan_cast_bytes(n, DType::F32, DType::C128, &f32_bytes(&reals));
    assert_eq!(
        c128_lanes,
        4 * n,
        "C128[{n}] must read back 4N f32 lanes (16N bytes)"
    );
}

// ── Gate 5: materialized complex Expand (lane-aware broadcast) ─────────────
//
// A complex `Op::Expand` returned directly (not consumed solely by a binary) is
// MATERIALIZED — it exercises the element-indexed `reindex` kernel, which copies
// one f32 per ELEMENT. A complex element spans 2 (C64) / 4 (C128) f32 lanes, so
// a naive expand shatters the `[re,im]` pairing. This gate pins the lane-aware
// fix (append an innermost lane axis so whole complex elements copy as a group).

/// rlx-cpu reference: complex `Expand(in_dims → out_dims)`, NATIVE output bytes.
fn cpu_expand_bytes(in_dims: &[usize], out_dims: &[usize], dt: DType, in_bytes: &[u8]) -> Vec<u8> {
    use rlx::prelude::*;
    let mut g = Graph::new("cexp_ref");
    let x = g.input("x", Shape::new(in_dims, dt));
    let tgt: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let y = g.add_node(
        Op::Expand { target_shape: tgt },
        vec![x],
        Shape::new(out_dims, dt),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.run_typed(&[("x", in_bytes, dt)]).remove(0).0
}

/// Vulkan candidate: complex `Expand`, output lanes re-narrowed to NATIVE bytes.
fn vulkan_expand_bytes(
    in_dims: &[usize],
    out_dims: &[usize],
    dt: DType,
    in_bytes: &[u8],
) -> Vec<u8> {
    let lanes_in = widen_bytes_to_f32(in_bytes, dt);
    let mut g = Graph::new("cexp_vk");
    let x = g.input("x", Shape::new(in_dims, dt));
    let tgt: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let y = g.add_node(
        Op::Expand { target_shape: tgt },
        vec![x],
        Shape::new(out_dims, dt),
    );
    g.set_outputs(vec![y]);
    let mut exe = VulkanExecutable::compile(g);
    let lanes_out = exe.run(&[("x", &lanes_in)]).remove(0);
    narrow_f32_to_bytes(&lanes_out, dt)
}

#[test]
fn expand_complex_materialized() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping complex Expand");
        return;
    }
    let _g = gpu_lock();
    // C64[1,2] -> [3,2]: broadcast the OUTER dim. 2 complex elems [re,im].
    let c64 = [1.5f32, 2.5, -3.0, 4.0];
    let cpu = cpu_expand_bytes(&[1, 2], &[3, 2], DType::C64, &c64_bytes(&c64));
    let gpu = vulkan_expand_bytes(&[1, 2], &[3, 2], DType::C64, &c64_bytes(&c64));
    assert_eq!(
        gpu, cpu,
        "C64 materialized Expand [1,2]->[3,2] vulkan vs cpu mismatch"
    );

    // C128[2,1] -> [2,3]: broadcast the INNER dim (f32-exact df64 values).
    let c128 = [1.5f64, -2.5, 3.25, -4.75];
    let cpu = cpu_expand_bytes(&[2, 1], &[2, 3], DType::C128, &c128_bytes(&c128));
    let gpu = vulkan_expand_bytes(&[2, 1], &[2, 3], DType::C128, &c128_bytes(&c128));
    assert_eq!(
        gpu, cpu,
        "C128 materialized Expand [2,1]->[2,3] vulkan vs cpu mismatch"
    );
}

// ── Gate 6: OTHER element-indexed movement ops must be lane-aware too ───────
//
// Expand was one of a CLASS: every element-indexed movement op (Transpose,
// Narrow, Concat, Gather, …) rides the `reindex`/`gather` kernels that copy one
// f32 per element and shatter complex unless made lane-aware. This gate audits
// the common single-/two-input ones. Direct port of the cuda/wgpu Gate 6.

/// rlx-cpu reference for a single-input op → NATIVE bytes.
fn cpu_op1_bytes(in_dims: &[usize], out_dims: &[usize], dt: DType, op: Op, xb: &[u8]) -> Vec<u8> {
    use rlx::prelude::*;
    let mut g = Graph::new("cop1_ref");
    let x = g.input("x", Shape::new(in_dims, dt));
    let y = g.add_node(op, vec![x], Shape::new(out_dims, dt));
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.run_typed(&[("x", xb, dt)]).remove(0).0
}
/// Vulkan candidate for a single-input op → NATIVE bytes (narrowed).
fn vulkan_op1_bytes(
    in_dims: &[usize],
    out_dims: &[usize],
    dt: DType,
    op: Op,
    xb: &[u8],
) -> Vec<u8> {
    let lanes_in = widen_bytes_to_f32(xb, dt);
    let mut g = Graph::new("cop1_vk");
    let x = g.input("x", Shape::new(in_dims, dt));
    let y = g.add_node(op, vec![x], Shape::new(out_dims, dt));
    g.set_outputs(vec![y]);
    let mut exe = VulkanExecutable::compile(g);
    let lanes_out = exe.run(&[("x", &lanes_in)]).remove(0);
    narrow_f32_to_bytes(&lanes_out, dt)
}

#[test]
fn transpose_complex() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping complex Transpose");
        return;
    }
    let _g = gpu_lock();
    // C64[2,3] --perm[1,0]--> [3,2]. 6 complex elems. A naive per-f32
    // transpose reindexes single lanes and shatters the [re,im] pairing.
    let c64: Vec<f32> = (0..12).map(|i| i as f32 + 0.5).collect();
    let op = Op::Transpose { perm: vec![1, 0] };
    let cpu = cpu_op1_bytes(&[2, 3], &[3, 2], DType::C64, op.clone(), &c64_bytes(&c64));
    let gpu = vulkan_op1_bytes(&[2, 3], &[3, 2], DType::C64, op, &c64_bytes(&c64));
    assert_eq!(gpu, cpu, "C64 Transpose vulkan vs cpu mismatch");

    // C128[2,2] --perm[1,0]--> [2,2] (f32-exact df64 values).
    let c128: Vec<f64> = vec![1.5, -2.5, 3.25, -4.75, 0.5, -6.0, 7.0, -8.25];
    let op = Op::Transpose { perm: vec![1, 0] };
    let cpu = cpu_op1_bytes(
        &[2, 2],
        &[2, 2],
        DType::C128,
        op.clone(),
        &c128_bytes(&c128),
    );
    let gpu = vulkan_op1_bytes(&[2, 2], &[2, 2], DType::C128, op, &c128_bytes(&c128));
    assert_eq!(gpu, cpu, "C128 Transpose vulkan vs cpu mismatch");
}

#[test]
fn narrow_complex() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping complex Narrow");
        return;
    }
    let _g = gpu_lock();
    // C64[4] --narrow axis0 [1..3)--> [2]. Keep complex elems 1,2.
    let c64 = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let op = Op::Narrow {
        axis: 0,
        start: 1,
        len: 2,
    };
    let cpu = cpu_op1_bytes(&[4], &[2], DType::C64, op.clone(), &c64_bytes(&c64));
    let gpu = vulkan_op1_bytes(&[4], &[2], DType::C64, op, &c64_bytes(&c64));
    assert_eq!(gpu, cpu, "C64 Narrow vulkan vs cpu mismatch");

    // C64[2,3] --narrow axis1 [1..3)--> [2,2]: inner (trailing) dim is copied,
    // so the lane axis must be innermost of the contiguous copy.
    let c64b: Vec<f32> = (0..12).map(|i| i as f32 + 0.25).collect();
    let op = Op::Narrow {
        axis: 1,
        start: 1,
        len: 2,
    };
    let cpu = cpu_op1_bytes(&[2, 3], &[2, 2], DType::C64, op.clone(), &c64_bytes(&c64b));
    let gpu = vulkan_op1_bytes(&[2, 3], &[2, 2], DType::C64, op, &c64_bytes(&c64b));
    assert_eq!(gpu, cpu, "C64 Narrow axis1 vulkan vs cpu mismatch");
}

#[test]
fn concat_complex() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping complex Concat");
        return;
    }
    let _g = gpu_lock();
    // C64[2] ++ C64[3] along axis0 -> C64[5].
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];
    let cpu = {
        use rlx::prelude::*;
        let mut g = Graph::new("ccat_ref");
        let x = g.input("a", Shape::new(&[2], DType::C64));
        let y = g.input("b", Shape::new(&[3], DType::C64));
        let z = g.add_node(
            Op::Concat { axis: 0 },
            vec![x, y],
            Shape::new(&[5], DType::C64),
        );
        g.set_outputs(vec![z]);
        let mut c = Session::new(Device::Cpu).compile(g);
        c.run_typed(&[
            ("a", &c64_bytes(&a), DType::C64),
            ("b", &c64_bytes(&b), DType::C64),
        ])
        .remove(0)
        .0
    };
    let gpu = {
        let al = widen_bytes_to_f32(&c64_bytes(&a), DType::C64);
        let bl = widen_bytes_to_f32(&c64_bytes(&b), DType::C64);
        let mut g = Graph::new("ccat_vk");
        let x = g.input("a", Shape::new(&[2], DType::C64));
        let y = g.input("b", Shape::new(&[3], DType::C64));
        let z = g.add_node(
            Op::Concat { axis: 0 },
            vec![x, y],
            Shape::new(&[5], DType::C64),
        );
        g.set_outputs(vec![z]);
        let mut exe = VulkanExecutable::compile(g);
        let out = exe.run(&[("a", &al), ("b", &bl)]).remove(0);
        narrow_f32_to_bytes(&out, DType::C64)
    };
    assert_eq!(gpu, cpu, "C64 Concat vulkan vs cpu mismatch");
}

#[test]
fn gather_complex() {
    if !available() {
        eprintln!("[complex_parity] no Vulkan device — skipping complex Gather");
        return;
    }
    let _g = gpu_lock();
    // C64[4] table --gather axis0 by idx [2,0,3,1]--> C64[4]. Each complex
    // element is 2 f32 lanes [re,im]; a naive per-f32 gather reads lane `i`
    // of the WRONG element and shatters the [re,im] pairing. Indices are I64.
    let table = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]; // 4 complex elems
    let idx: [i64; 4] = [2, 0, 3, 1];
    let idx_bytes: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let cpu = {
        use rlx::prelude::*;
        let mut g = Graph::new("cgat_ref");
        let t = g.input("t", Shape::new(&[4], DType::C64));
        let ix = g.input("ix", Shape::new(&[4], DType::I64));
        let z = g.add_node(
            Op::Gather { axis: 0 },
            vec![t, ix],
            Shape::new(&[4], DType::C64),
        );
        g.set_outputs(vec![z]);
        let mut c = Session::new(Device::Cpu).compile(g);
        c.run_typed(&[
            ("t", &c64_bytes(&table), DType::C64),
            ("ix", &idx_bytes, DType::I64),
        ])
        .remove(0)
        .0
    };
    let gpu = {
        let tl = widen_bytes_to_f32(&c64_bytes(&table), DType::C64);
        let il = widen_bytes_to_f32(&idx_bytes, DType::I64);
        let mut g = Graph::new("cgat_vk");
        let t = g.input("t", Shape::new(&[4], DType::C64));
        let ix = g.input("ix", Shape::new(&[4], DType::I64));
        let z = g.add_node(
            Op::Gather { axis: 0 },
            vec![t, ix],
            Shape::new(&[4], DType::C64),
        );
        g.set_outputs(vec![z]);
        let mut exe = VulkanExecutable::compile(g);
        let out = exe.run(&[("t", &tl), ("ix", &il)]).remove(0);
        narrow_f32_to_bytes(&out, DType::C64)
    };
    assert_eq!(gpu, cpu, "C64 Gather vulkan vs cpu mismatch");
}
