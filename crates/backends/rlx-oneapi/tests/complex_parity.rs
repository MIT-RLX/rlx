// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! On-device complex simulation for the oneAPI (Level Zero) backend.
//!
//! Representation (fixed, shared with rlx-wgpu / rlx-cuda / rlx-vulkan):
//!   C64  = 2 f32 lanes `[re, im]`                          (8 B/elem)
//!   C128 = 4 f32 lanes df64 `[re_hi, re_lo, im_hi, im_lo]`  (16 B/elem)
//! Casts from f32 real sources have `lo = 0`, so every complex cast is a pure
//! lane MOVE (`complex_cast`); C64 arithmetic (`binary_c64`) reads BOTH lanes.
//! C128 arithmetic + C64 max/min/pow are rejected (rlx-cpu has none either).
//!
//! Unlike the GPU-only backends' complex_parity tests (which no-op without a
//! device), the oneAPI executable ALWAYS has a correct path: with no Level Zero
//! device (this macOS dev box / CI) the whole graph runs through the rlx-cpu
//! reference via `run_host`, whose complex handling shares the SAME lane
//! semantics as the on-device kernels. So these gates run everywhere and pin the
//! shared complex logic (cast modes, C64 formulas, df64 boundary, and — most
//! importantly — the lane-count slot sizing + readback, which would otherwise
//! truncate a complex output to its real parts).

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_oneapi::backend::OneApiExecutable;

fn run1(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    OneApiExecutable::compile(g)
        .run(inputs)
        .into_iter()
        .next()
        .unwrap()
}

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "lane-count mismatch: {a:?} vs {b:?}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "lane {i}: {x} vs {y} (tol {tol})");
    }
}

/// A C64 `Binary(op)` over `[n]`-shaped operands; returns 2N output lanes.
fn c64_binary(op: BinaryOp, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("c64_bin");
    let x = g.input("a", Shape::new(&[n], DType::C64));
    let y = g.input("b", Shape::new(&[n], DType::C64));
    let z = g.add_node(Op::Binary(op), vec![x, y], Shape::new(&[n], DType::C64));
    g.set_outputs(vec![z]);
    run1(g, &[("a", a), ("b", b)])
}

// ── Gate 1: C64 arithmetic (add/sub bit-exact, mul/div ≤1e-6) ──────────────

#[test]
fn c64_add_sub_exact() {
    // (1+2i, 3+4i) and (5+6i, 7-8i).
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, -8.0];
    // add = (6+8i, 10-4i); sub = (-4-4i, -4+12i).
    approx(
        &c64_binary(BinaryOp::Add, 2, &a, &b),
        &[6.0, 8.0, 10.0, -4.0],
        0.0,
    );
    approx(
        &c64_binary(BinaryOp::Sub, 2, &a, &b),
        &[-4.0, -4.0, -4.0, 12.0],
        0.0,
    );
}

#[test]
fn c64_mul_div() {
    // (1+2i) ∘ (3+4i).
    let a = [1.0f32, 2.0];
    let b = [3.0f32, 4.0];
    // mul = (3-8) + (4+6)i = -5+10i.
    approx(&c64_binary(BinaryOp::Mul, 1, &a, &b), &[-5.0, 10.0], 1e-6);
    // div = (1+2i)/(3+4i); d=25 → (11/25, 2/25) = (0.44, 0.08).
    approx(&c64_binary(BinaryOp::Div, 1, &a, &b), &[0.44, 0.08], 1e-6);
}

#[test]
fn c64_scalar_broadcast() {
    // a = (1+1i, 2+2i) over [2]; b = (10+0i) over [1] → broadcast add.
    let mut g = Graph::new("c64_bcast");
    let x = g.input("a", Shape::new(&[2], DType::C64));
    let y = g.input("b", Shape::new(&[1], DType::C64));
    let z = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![x, y],
        Shape::new(&[2], DType::C64),
    );
    g.set_outputs(vec![z]);
    // (11+1i, 12+2i).
    approx(
        &run1(g, &[("a", &[1.0, 1.0, 2.0, 2.0]), ("b", &[10.0, 0.0])]),
        &[11.0, 1.0, 12.0, 2.0],
        0.0,
    );
}

#[test]
#[should_panic(expected = "undefined for complex")]
fn c64_max_rejected() {
    // Max is undefined for complex → reject (matches rlx-cpu).
    let _ = c64_binary(BinaryOp::Max, 1, &[1.0, 2.0], &[3.0, 4.0]);
}

// ── Gate 2: complex casts (pure lane moves) ────────────────────────────────

fn cast(n: usize, src: DType, dst: DType, input: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("ccast");
    let x = g.input("x", Shape::new(&[n], src));
    let y = g.add_node(Op::Cast { to: dst }, vec![x], Shape::new(&[n], dst));
    g.set_outputs(vec![y]);
    run1(g, &[("x", input)])
}

#[test]
fn cast_real_c64_roundtrip() {
    // real → C64: im lanes are 0.
    let reals = [1.5f32, -2.5, 3.0];
    approx(
        &cast(3, DType::F32, DType::C64, &reals),
        &[1.5, 0.0, -2.5, 0.0, 3.0, 0.0],
        0.0,
    );
    // C64 → real: imaginary parts dropped.
    let c64 = [1.5f32, 2.5, -3.0, 4.0];
    approx(&cast(2, DType::C64, DType::F32, &c64), &[1.5, -3.0], 0.0);
}

#[test]
fn cast_c64_c128_roundtrip() {
    // C64 (1.5+2.5i, -3+4i) → C128 df64 (lo lanes 0 for f32-exact values).
    let c64 = [1.5f32, 2.5, -3.0, 4.0];
    let c128 = cast(2, DType::C64, DType::C128, &c64);
    approx(&c128, &[1.5, 0.0, 2.5, 0.0, -3.0, 0.0, 4.0, 0.0], 0.0);
    // C128 → C64 drops the df64 `lo` lanes (keeps `hi`).
    approx(&cast(2, DType::C128, DType::C64, &c128), &c64, 0.0);
}

#[test]
fn cast_real_to_c128_lane_sizing() {
    // real → C128: a C128 output occupies 4 lanes/elem; without lane-aware slot
    // sizing + readback this would truncate to N lanes (the real parts only).
    let reals = [1.5f32, -2.25]; // f32-exact → df64 lo = 0
    let out = cast(2, DType::F32, DType::C128, &reals);
    assert_eq!(out.len(), 8, "C128 output must read back 4N f32 lanes");
    approx(&out, &[1.5, 0.0, 0.0, 0.0, -2.25, 0.0, 0.0, 0.0], 0.0);
}

#[test]
#[should_panic(expected = "F64 real component")]
fn cast_f64_complex_rejected() {
    // A complex cast touching an F64 real component is rejected (no faithful
    // f32-lane storage) — matches the on-device path.
    let _ = cast(2, DType::F64, DType::C64, &[1.0, 2.0]);
}

// ── Gate 3: materialized complex Expand (lane-aware broadcast) ──────────────
//
// A complex `Op::Expand` returned directly (not folded into a following binary)
// MATERIALIZES the broadcast. On the GPU-only backends (wgpu / cuda) this rides
// an element-indexed expand kernel that copies ONE f32 per output ELEMENT — but
// a complex element spans 2 (C64) / 4 (C128) f32 lanes, so a naive per-element
// copy shatters the `[re, im]` pairing (those backends fix it by appending an
// innermost lane axis so whole complex elements copy as a group).
//
// oneAPI has NO native expand kernel: `Op::Expand` is not in `native_kernel`,
// so it routes to the rlx-cpu host-fallback on BOTH paths — `run_host` here,
// and `run_l0`'s no-native-kernel branch on Intel HW. rlx-cpu's Expand copies
// whole `size_bytes()` elements (8 B C64 / 16 B C128), never a single f32, so
// the `[re, im]` / df64 lanes stay paired without any oneAPI-side lane axis.
//
// This gate is a REAL runtime check on this Mac (whole graph via `run_host`):
// it pins the full chain — rlx-cpu's element-sized broadcast AND oneAPI's
// lane-aware readback (`arena_lane_count`). A readback that used the logical
// element count would truncate a complex output to a fraction of its lanes.

/// A materialized complex `Op::Expand(in_dims → out_dims)`; returns the output
/// f32 lanes (2/elem C64, 4/elem C128).
fn expand(in_dims: &[usize], out_dims: &[usize], dt: DType, input: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("cexp");
    let x = g.input("x", Shape::new(in_dims, dt));
    let tgt: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let y = g.add_node(
        Op::Expand { target_shape: tgt },
        vec![x],
        Shape::new(out_dims, dt),
    );
    g.set_outputs(vec![y]);
    run1(g, &[("x", input)])
}

#[test]
fn expand_complex_materialized() {
    // C64[1,2] -> [3,2]: broadcast the OUTER dim. 2 complex elems (4 [re,im]
    // lanes) replicated 3× → 12 lanes. A shattering expand would mis-pair /
    // drop the imaginary lanes (e.g. `[1.5,2.5, 1.5,2.5, 1.5,2.5, 0,0,…]`).
    let c64 = [1.5f32, 2.5, -3.0, 4.0];
    let c64_out = [
        1.5, 2.5, -3.0, 4.0, 1.5, 2.5, -3.0, 4.0, 1.5, 2.5, -3.0, 4.0,
    ];
    approx(&expand(&[1, 2], &[3, 2], DType::C64, &c64), &c64_out, 0.0);

    // C128[2,1] -> [2,3]: broadcast the INNER dim. 2 complex elems, 4 df64 lanes
    // each (lo = 0 for f32-exact values), each replicated 3× → 24 lanes.
    let c128 = [1.5f32, 0.0, -2.5, 0.0, 3.25, 0.0, -4.75, 0.0];
    let c128_out = [
        1.5, 0.0, -2.5, 0.0, 1.5, 0.0, -2.5, 0.0, 1.5, 0.0, -2.5, 0.0, 3.25, 0.0, -4.75, 0.0, 3.25,
        0.0, -4.75, 0.0, 3.25, 0.0, -4.75, 0.0,
    ];
    approx(
        &expand(&[2, 1], &[2, 3], DType::C128, &c128),
        &c128_out,
        0.0,
    );
}

// ── Gate 4: OTHER element-indexed movement ops must stay lane-paired too ─────
//
// Expand was one of a CLASS: every element-indexed movement op — Transpose,
// Narrow, Concat, Gather — copies data per logical ELEMENT, and a complex
// element spans 2 (C64) / 4 (C128) f32 lanes, so a naive per-f32 move shatters
// the `[re, im]` pairing. On the GPU-only backends (wgpu / cuda) each of these
// rides an element-indexed kernel that had to gain an innermost lane axis.
//
// oneAPI has NO native kernel for ANY of these ops: none are in `native_kernel`,
// so they route to the rlx-cpu host-fallback on BOTH paths — `run_host` here,
// and `run_l0`'s no-native-kernel branch on Intel HW. rlx-cpu's movement ops
// copy whole `size_bytes()` elements (8 B C64 / 16 B C128 — Gather even does an
// i64-granular copy for the 8-byte C64 case), never a single f32, so the lanes
// stay paired without any oneAPI-side lane axis. These gates are REAL runtime
// checks on this Mac (whole graph via `run_host`): they pin the full chain —
// rlx-cpu's element-sized movement AND oneAPI's lane-aware readback
// (`arena_lane_count` / `host::eval`'s `is_complex()` branch), which would
// otherwise truncate a complex output to a fraction of its lanes.

/// A single-input movement op over C64 data; returns the output f32 lanes
/// (2/elem). `in_dims`/`out_dims` are element (not lane) shapes.
fn move1(in_dims: &[usize], out_dims: &[usize], op: Op, input: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("cmove1");
    let x = g.input("x", Shape::new(in_dims, DType::C64));
    let y = g.add_node(op, vec![x], Shape::new(out_dims, DType::C64));
    g.set_outputs(vec![y]);
    run1(g, &[("x", input)])
}

#[test]
fn transpose_complex() {
    // C64[2,3] --perm[1,0]--> [3,2]. 6 complex elems `[re, im]`; a per-f32
    // transpose would move lane indices independently and mis-pair re/im.
    let c64: Vec<f32> = (0..12).map(|i| i as f32 + 0.5).collect();
    // Row-major [2,3] elems: (0.5+1.5i)(2.5+3.5i)(4.5+5.5i) / (6.5+7.5i)(8.5+9.5i)(10.5+11.5i).
    // Transposed [3,2]: out(i,j)=in(j,i).
    let golden = [
        0.5, 1.5, 6.5, 7.5, // row0: in(0,0), in(1,0)
        2.5, 3.5, 8.5, 9.5, // row1: in(0,1), in(1,1)
        4.5, 5.5, 10.5, 11.5, // row2: in(0,2), in(1,2)
    ];
    approx(
        &move1(&[2, 3], &[3, 2], Op::Transpose { perm: vec![1, 0] }, &c64),
        &golden,
        0.0,
    );
}

#[test]
fn narrow_complex() {
    // C64[4] --narrow axis0 [1..3)--> [2]. Keep complex elems 1,2 whole.
    // elems: (0+1i)(2+3i)(4+5i)(6+7i) → keep (2+3i)(4+5i).
    let c64 = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    approx(
        &move1(
            &[4],
            &[2],
            Op::Narrow {
                axis: 0,
                start: 1,
                len: 2,
            },
            &c64,
        ),
        &[2.0, 3.0, 4.0, 5.0],
        0.0,
    );
}

#[test]
fn concat_complex() {
    // C64[2] ++ C64[3] along axis0 -> C64[5]; every element copied whole.
    let a = [1.0f32, 2.0, 3.0, 4.0]; // (1+2i)(3+4i)
    let b = [5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0]; // (5+6i)(7+8i)(9+10i)
    let mut g = Graph::new("ccat");
    let x = g.input("a", Shape::new(&[2], DType::C64));
    let y = g.input("b", Shape::new(&[3], DType::C64));
    let z = g.add_node(
        Op::Concat { axis: 0 },
        vec![x, y],
        Shape::new(&[5], DType::C64),
    );
    g.set_outputs(vec![z]);
    approx(
        &run1(g, &[("a", &a), ("b", &b)]),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        0.0,
    );
}

#[test]
fn gather_complex() {
    // C64[4] table --gather axis0 by idx [2,0,3,1]--> C64[4]. Each complex
    // element is 2 f32 lanes `[re, im]`; a naive per-f32 gather would read lane
    // `i` of the WRONG element and shatter the pairing. Indices MUST be I64:
    // for the 8-byte C64 table rlx-cpu's `exec_gather` takes the i64-granular
    // path (copies whole 8-byte elements as one i64) and reads the index buffer
    // as i64 — an I32 index buffer would be read as f32 lanes (garbage rows).
    let table = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]; // (0+0.5i)(1+1.5i)(2+2.5i)(3+3.5i)
    let idx = [2.0f32, 0.0, 3.0, 1.0]; // I64 values widened to one f32 lane/elem
    let mut g = Graph::new("cgat");
    let t = g.input("t", Shape::new(&[4], DType::C64));
    let ix = g.input("ix", Shape::new(&[4], DType::I64));
    let z = g.add_node(
        Op::Gather { axis: 0 },
        vec![t, ix],
        Shape::new(&[4], DType::C64),
    );
    g.set_outputs(vec![z]);
    // gathered: elem2, elem0, elem3, elem1.
    approx(
        &run1(g, &[("t", &table), ("ix", &idx)]),
        &[2.0, 2.5, 0.0, 0.5, 3.0, 3.5, 1.0, 1.5],
        0.0,
    );
}
