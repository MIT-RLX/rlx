// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-ROCm parity for on-device complex simulation on the f32-uniform arena.
//! Direct port of the Expand gate from `rlx-cuda/tests/complex_parity.rs`. On a
//! ROCm-less host every device test no-ops (`is_available()==false`), matching
//! the rest of the rlx-rocm test suite.
//!
//! Representation (fixed): `C64 = 2 f32 lanes [re, im]`;
//! `C128 = 4 f32 lanes df64 [re_hi, re_lo, im_hi, im_lo]`.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_rocm::backend::RocmExecutable;
use rlx_runtime::backend::{narrow_f32_to_bytes, widen_bytes_to_f32};
use rlx_runtime::{Device, Session};

fn available() -> bool {
    rlx_rocm::is_available()
}

// ── host-byte builders ────────────────────────────────────────────────────

/// C64 host bytes from interleaved `[re, im]` f32 pairs (`comps.len() == 2N`).
fn c64_bytes(comps: &[f32]) -> Vec<u8> {
    comps.iter().flat_map(|x| x.to_le_bytes()).collect()
}
/// C128 host bytes from interleaved `[re, im]` f64 pairs (`comps.len() == 2N`).
fn c128_bytes(comps: &[f64]) -> Vec<u8> {
    comps.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ── Gate: materialized complex Expand (lane-aware broadcast) ───────────────
//
// A complex `Op::Expand` returned directly (not consumed solely by a binary) is
// MATERIALIZED — it exercises the element-indexed expand kernel, which copies
// one f32 per ELEMENT. A complex element spans 2 (C64) / 4 (C128) f32 lanes, so
// a naive expand shatters the `[re,im]` pairing. This gate pins the lane-aware
// fix (append an innermost lane axis so whole complex elements copy as a group).

/// rlx-cpu reference: complex `Expand(in_dims → out_dims)`, NATIVE output bytes.
fn cpu_expand_bytes(in_dims: &[usize], out_dims: &[usize], dt: DType, in_bytes: &[u8]) -> Vec<u8> {
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

/// ROCm candidate: complex `Expand`, output lanes re-narrowed to NATIVE bytes.
fn rocm_expand_bytes(in_dims: &[usize], out_dims: &[usize], dt: DType, in_bytes: &[u8]) -> Vec<u8> {
    let lanes_in = widen_bytes_to_f32(in_bytes, dt);
    let mut g = Graph::new("cexp_rocm");
    let x = g.input("x", Shape::new(in_dims, dt));
    let tgt: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let y = g.add_node(
        Op::Expand { target_shape: tgt },
        vec![x],
        Shape::new(out_dims, dt),
    );
    g.set_outputs(vec![y]);
    let mut exe = RocmExecutable::compile(g);
    let lanes_out = exe.run(&[("x", &lanes_in)]).remove(0);
    narrow_f32_to_bytes(&lanes_out, dt)
}

#[test]
fn expand_complex_materialized() {
    if !available() {
        eprintln!("[complex_parity] no rocm device — skipping complex Expand");
        return;
    }
    // C64[1,2] -> [3,2]: broadcast the OUTER dim. 2 complex elems [re,im].
    let c64 = [1.5f32, 2.5, -3.0, 4.0];
    let cpu = cpu_expand_bytes(&[1, 2], &[3, 2], DType::C64, &c64_bytes(&c64));
    let gpu = rocm_expand_bytes(&[1, 2], &[3, 2], DType::C64, &c64_bytes(&c64));
    assert_eq!(
        gpu, cpu,
        "C64 materialized Expand [1,2]->[3,2] rocm vs cpu mismatch"
    );

    // C128[2,1] -> [2,3]: broadcast the INNER dim (f32-exact df64 values).
    let c128 = [1.5f64, -2.5, 3.25, -4.75];
    let cpu = cpu_expand_bytes(&[2, 1], &[2, 3], DType::C128, &c128_bytes(&c128));
    let gpu = rocm_expand_bytes(&[2, 1], &[2, 3], DType::C128, &c128_bytes(&c128));
    assert_eq!(
        gpu, cpu,
        "C128 materialized Expand [2,1]->[2,3] rocm vs cpu mismatch"
    );
}

// ── Gate: OTHER element-indexed movement ops must be lane-aware too ──────────
//
// Expand was one of a CLASS: every element-indexed movement op (Transpose,
// Narrow, Concat, Gather, …) copies one f32 per element and shatters complex
// unless made lane-aware. This gate audits the common single-/two-input ones.
// Direct port of `rlx-cuda/tests/complex_parity.rs` Gate 6.

/// rlx-cpu reference for a single-input op → NATIVE bytes.
fn cpu_op1_bytes(in_dims: &[usize], out_dims: &[usize], dt: DType, op: Op, xb: &[u8]) -> Vec<u8> {
    let mut g = Graph::new("cop1_ref");
    let x = g.input("x", Shape::new(in_dims, dt));
    let y = g.add_node(op, vec![x], Shape::new(out_dims, dt));
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.run_typed(&[("x", xb, dt)]).remove(0).0
}
/// ROCm candidate for a single-input op → NATIVE bytes (narrowed).
fn rocm_op1_bytes(in_dims: &[usize], out_dims: &[usize], dt: DType, op: Op, xb: &[u8]) -> Vec<u8> {
    let lanes_in = widen_bytes_to_f32(xb, dt);
    let mut g = Graph::new("cop1_rocm");
    let x = g.input("x", Shape::new(in_dims, dt));
    let y = g.add_node(op, vec![x], Shape::new(out_dims, dt));
    g.set_outputs(vec![y]);
    let mut exe = RocmExecutable::compile(g);
    let lanes_out = exe.run(&[("x", &lanes_in)]).remove(0);
    narrow_f32_to_bytes(&lanes_out, dt)
}

#[test]
fn transpose_complex() {
    if !available() {
        eprintln!("[complex_parity] no rocm device — skipping complex Transpose");
        return;
    }
    // C64[2,3] --perm[1,0]--> [3,2]. 6 complex elems. A naive per-f32
    // transpose reindexes single lanes and shatters the [re,im] pairing.
    let c64: Vec<f32> = (0..12).map(|i| i as f32 + 0.5).collect();
    let op = Op::Transpose { perm: vec![1, 0] };
    let cpu = cpu_op1_bytes(&[2, 3], &[3, 2], DType::C64, op.clone(), &c64_bytes(&c64));
    let gpu = rocm_op1_bytes(&[2, 3], &[3, 2], DType::C64, op, &c64_bytes(&c64));
    assert_eq!(gpu, cpu, "C64 Transpose rocm vs cpu mismatch");

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
    let gpu = rocm_op1_bytes(&[2, 2], &[2, 2], DType::C128, op, &c128_bytes(&c128));
    assert_eq!(gpu, cpu, "C128 Transpose rocm vs cpu mismatch");
}

#[test]
fn narrow_complex() {
    if !available() {
        eprintln!("[complex_parity] no rocm device — skipping complex Narrow");
        return;
    }
    // C64[4] --narrow axis0 [1..3)--> [2]. Keep complex elems 1,2.
    let c64 = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let op = Op::Narrow {
        axis: 0,
        start: 1,
        len: 2,
    };
    let cpu = cpu_op1_bytes(&[4], &[2], DType::C64, op.clone(), &c64_bytes(&c64));
    let gpu = rocm_op1_bytes(&[4], &[2], DType::C64, op, &c64_bytes(&c64));
    assert_eq!(gpu, cpu, "C64 Narrow rocm vs cpu mismatch");

    // C64[2,3] --narrow axis1 [1..3)--> [2,2]: inner (trailing) dim is copied,
    // so the lane axis must be innermost of the contiguous copy.
    let c64b: Vec<f32> = (0..12).map(|i| i as f32 + 0.25).collect();
    let op = Op::Narrow {
        axis: 1,
        start: 1,
        len: 2,
    };
    let cpu = cpu_op1_bytes(&[2, 3], &[2, 2], DType::C64, op.clone(), &c64_bytes(&c64b));
    let gpu = rocm_op1_bytes(&[2, 3], &[2, 2], DType::C64, op, &c64_bytes(&c64b));
    assert_eq!(gpu, cpu, "C64 Narrow axis1 rocm vs cpu mismatch");
}

#[test]
fn concat_complex() {
    if !available() {
        eprintln!("[complex_parity] no rocm device — skipping complex Concat");
        return;
    }
    // C64[2] ++ C64[3] along axis0 -> C64[5].
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];
    let cpu = {
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
        let mut g = Graph::new("ccat_rocm");
        let x = g.input("a", Shape::new(&[2], DType::C64));
        let y = g.input("b", Shape::new(&[3], DType::C64));
        let z = g.add_node(
            Op::Concat { axis: 0 },
            vec![x, y],
            Shape::new(&[5], DType::C64),
        );
        g.set_outputs(vec![z]);
        let mut exe = RocmExecutable::compile(g);
        let out = exe.run(&[("a", &al), ("b", &bl)]).remove(0);
        narrow_f32_to_bytes(&out, DType::C64)
    };
    assert_eq!(gpu, cpu, "C64 Concat rocm vs cpu mismatch");
}

#[test]
fn gather_complex() {
    if !available() {
        eprintln!("[complex_parity] no rocm device — skipping complex Gather");
        return;
    }
    // C64[4] table --gather axis0 by idx [2,0,3,1]--> C64[4]. Each complex
    // element is 2 f32 lanes [re,im]; a naive per-f32 gather reads lane `i`
    // of the WRONG element and shatters the [re,im] pairing.
    let table = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]; // 4 complex elems
    let idx: [i64; 4] = [2, 0, 3, 1];
    let idx_bytes: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
    let cpu = {
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
        let mut g = Graph::new("cgat_rocm");
        let t = g.input("t", Shape::new(&[4], DType::C64));
        let ix = g.input("ix", Shape::new(&[4], DType::I64));
        let z = g.add_node(
            Op::Gather { axis: 0 },
            vec![t, ix],
            Shape::new(&[4], DType::C64),
        );
        g.set_outputs(vec![z]);
        let mut exe = RocmExecutable::compile(g);
        let out = exe.run(&[("t", &tl), ("ix", &il)]).remove(0);
        narrow_f32_to_bytes(&out, DType::C64)
    };
    assert_eq!(gpu, cpu, "C64 Gather rocm vs cpu mismatch");
}
