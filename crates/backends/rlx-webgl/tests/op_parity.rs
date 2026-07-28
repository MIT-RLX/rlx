// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Per-op-family parity: the WebGL lowering (planner + CPU executor) must match
// RLX's CPU backend. Covers the kernels added beyond the MLP set.

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};

fn f32s(shape: &[usize], f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..shape.iter().product::<usize>()).map(f).collect()
}

/// Run `g` through RLX's CPU backend and through the WebGL lowering; assert equal.
fn parity(g: Graph, inputs: &[(&str, &[f32])]) {
    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g.clone());
    let reference = sess.run(inputs);

    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let got = rlx_webgl::run_cpu(&plan, inputs).expect("run_cpu");

    assert_eq!(reference.len(), got.len(), "output count");
    for (oi, (r, gv)) in reference.iter().zip(&got).enumerate() {
        assert_eq!(r.len(), gv.len(), "output {oi} length");
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "output {oi}[{i}]: reference={a} webgl={b}"
            );
        }
    }
}

fn shape2(r: usize, c: usize) -> Shape {
    Shape::new(&[r, c], DType::F32)
}

#[test]
fn binary_max_min_pow() {
    let a = f32s(&[2, 3], |i| i as f32 - 2.5);
    let b = f32s(&[2, 3], |i| 1.5 - i as f32);
    for op in [BinaryOp::Max, BinaryOp::Min, BinaryOp::Pow] {
        let mut g = Graph::new("bin");
        let ai = g.input("a", shape2(2, 3));
        let bi = g.input("b", shape2(2, 3));
        // Pow needs a positive base to stay real.
        let (av, bv) = if op == BinaryOp::Pow {
            (
                f32s(&[2, 3], |i| 0.5 + i as f32 * 0.3),
                f32s(&[2, 3], |i| 1.0 + i as f32 * 0.2),
            )
        } else {
            (a.clone(), b.clone())
        };
        let y = g.binary(op, ai, bi, shape2(2, 3));
        g.set_outputs(vec![y]);
        parity(g, &[("a", &av), ("b", &bv)]);
    }
}

#[test]
fn activations() {
    // Mixed-sign domain for the everywhere-defined activations.
    let xs = f32s(&[2, 4], |i| i as f32 * 0.5 - 2.0);
    for act in [
        Activation::Relu,
        Activation::Neg,
        Activation::Exp,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Abs,
        Activation::Sin,
        Activation::Cos,
        Activation::Silu,
    ] {
        let mut g = Graph::new("act");
        let x = g.input("x", shape2(2, 4));
        let y = g.activation(act, x, shape2(2, 4));
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
    // Positive domain for Log / Sqrt / Rsqrt.
    let xp = f32s(&[2, 4], |i| 0.25 + i as f32 * 0.5);
    for act in [Activation::Log, Activation::Sqrt, Activation::Rsqrt] {
        let mut g = Graph::new("act_pos");
        let x = g.input("x", shape2(2, 4));
        let y = g.activation(act, x, shape2(2, 4));
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xp)]);
    }
}

#[test]
fn reduce_ops() {
    let xs = f32s(&[3, 4], |i| (i as f32 * 0.37).sin() + 0.5);
    for op in [
        ReduceOp::Sum,
        ReduceOp::Mean,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::Prod,
    ] {
        let mut g = Graph::new("red");
        let x = g.input("x", shape2(3, 4));
        let y = g.reduce(x, op, vec![1], false, Shape::new(&[3], DType::F32));
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
    // Reduce all axes → scalar.
    let mut g = Graph::new("red_all");
    let x = g.input("x", shape2(3, 4));
    let y = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0, 1],
        false,
        Shape::scalar(DType::F32),
    );
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn softmax_last_axis() {
    let xs = f32s(&[2, 5], |i| i as f32 * 0.3 - 1.0);
    let mut g = Graph::new("sm");
    let x = g.input("x", shape2(2, 5));
    let y = g.softmax(x, -1, shape2(2, 5));
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn reverse_axis() {
    let xs = f32s(&[2, 4], |i| i as f32);
    let mut g = Graph::new("rev");
    let x = g.input("x", shape2(2, 4));
    let y = g.reverse(x, vec![1]);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

/// Backward through a non-ReLU activation exercises `ActivationBackward`.
#[test]
fn sigmoid_mlp_backward_matches_cpu() {
    let (in_dim, hidden, out_dim) = (3usize, 4usize, 2usize);
    let mut g = Graph::new("sig_mlp");
    let x = g.input("x", shape2(1, in_dim));
    let w1 = g.param("w1", shape2(in_dim, hidden));
    let b1 = g.param("b1", shape2(1, hidden));
    let w2 = g.param("w2", shape2(hidden, out_dim));
    let b2 = g.param("b2", shape2(1, out_dim));
    let h = g.matmul(x, w1, shape2(1, hidden));
    let h = g.add(h, b1);
    let h = g.activation(Activation::Sigmoid, h, shape2(1, hidden));
    let y = g.matmul(h, w2, shape2(1, out_dim));
    let y = g.add(y, b2);
    let t = g.input("target", shape2(1, out_dim));
    let d = g.sub(y, t);
    let sq = g.mul(d, d);
    let loss = g.sum(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);

    let bwd = rlx_autodiff::grad_with_loss(&g, &[w1, b1, w2, b2]);

    let xd = vec![0.5f32, -0.3, 0.8];
    let td = vec![0.2f32, 0.7];
    let w1d = f32s(&[in_dim, hidden], |i| 0.1 * i as f32 - 0.2);
    let b1d = vec![0.0f32; hidden];
    let w2d = f32s(&[hidden, out_dim], |i| 0.15 * i as f32 - 0.1);
    let b2d = vec![0.0f32; out_dim];

    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(bwd.clone());
    sess.set_param("w1", &w1d);
    sess.set_param("b1", &b1d);
    sess.set_param("w2", &w2d);
    sess.set_param("b2", &b2d);
    let reference = sess.run(&[("x", &xd), ("target", &td), ("d_output", &[1.0])]);

    let plan = rlx_webgl::build_plan(&bwd).expect("plan");
    let got = rlx_webgl::run_cpu(
        &plan,
        &[
            ("x", &xd),
            ("target", &td),
            ("d_output", &[1.0]),
            ("w1", &w1d),
            ("b1", &b1d),
            ("w2", &w2d),
            ("b2", &b2d),
        ],
    )
    .expect("run_cpu");

    assert_eq!(reference.len(), got.len());
    for (r, gv) in reference.iter().zip(&got) {
        for (a, b) in r.iter().zip(gv) {
            assert!((a - b).abs() <= 1e-4 * (1.0 + a.abs()), "ref={a} webgl={b}");
        }
    }
}

// --- Complex (C64) simulation on the f32-uniform arena ------------------------
// The planner must un-reject complex Cast (→ ComplexCast lane moves) and route
// complex Binary (→ BinaryC64) instead of the scalar-per-lane path. Ending each
// graph on a real (F32) output keeps the reference/webgl comparison unambiguous
// (both return N real f32), while still exercising real↔C64 casts + C64 arith.

fn shape_c64(r: usize, c: usize) -> Shape {
    Shape::new(&[r, c], DType::C64)
}

#[test]
fn complex_c64_mul_roundtrip() {
    // re(cast_f32( C64(a) * C64(b) )) == a*b   (imag parts are 0).
    let a = f32s(&[2, 3], |i| i as f32 - 2.5);
    let b = f32s(&[2, 3], |i| 1.5 - i as f32 * 0.4);
    let mut g = Graph::new("cmul");
    let ai = g.input("a", shape2(2, 3));
    let bi = g.input("b", shape2(2, 3));
    let ac = g.cast(ai, DType::C64);
    let bc = g.cast(bi, DType::C64);
    let prod = g.binary(BinaryOp::Mul, ac, bc, shape_c64(2, 3));
    let re = g.cast(prod, DType::F32);
    g.set_outputs(vec![re]);
    parity(g, &[("a", &a), ("b", &b)]);
}

#[test]
fn complex_c64_add_roundtrip() {
    let a = f32s(&[2, 3], |i| i as f32 - 2.5);
    let b = f32s(&[2, 3], |i| 1.5 - i as f32 * 0.4);
    let mut g = Graph::new("cadd");
    let ai = g.input("a", shape2(2, 3));
    let bi = g.input("b", shape2(2, 3));
    let ac = g.cast(ai, DType::C64);
    let bc = g.cast(bi, DType::C64);
    let sum = g.binary(BinaryOp::Add, ac, bc, shape_c64(2, 3));
    let re = g.cast(sum, DType::F32);
    g.set_outputs(vec![re]);
    parity(g, &[("a", &a), ("b", &b)]);
}

// --- Materialized complex Expand (lane-aware broadcast) -----------------------
// Ported from rlx-cuda/tests/complex_parity.rs::expand_complex_materialized. A
// complex `Op::Expand` returned directly is MATERIALIZED — it exercises the
// element-indexed broadcast gather. On the lane-based f32 arena a complex element
// spans 2 (C64) / 4 (C128) f32 lanes, so a naive element-per-output broadcast
// shatters the `[re, im]` pairing (writes N of the N*lane lanes from wrong source
// lanes). This gate pins the lane-aware planner fix (append an innermost lane
// axis so whole complex elements copy as a contiguous group). Golden = rlx-cpu;
// candidate = webgl's planner + native CPU executor. Comparison is in the shared
// native byte representation via the df64 boundary helpers.

use rlx_runtime::backend::{narrow_f32_to_bytes, widen_bytes_to_f32};

/// C64 native bytes from interleaved `[re, im]` f32 pairs (`comps.len() == 2N`).
fn c64_bytes(comps: &[f32]) -> Vec<u8> {
    comps.iter().flat_map(|x| x.to_le_bytes()).collect()
}
/// C128 native bytes from interleaved `[re, im]` f64 pairs (`comps.len() == 2N`).
fn c128_bytes(comps: &[f64]) -> Vec<u8> {
    comps.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// rlx-cpu golden: complex `Expand(in_dims → out_dims)`, NATIVE output bytes.
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
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
    c.run_typed(&[("x", in_bytes, dt)]).remove(0).0
}

/// webgl candidate: complex `Expand` through `build_plan` + `run_cpu`. Feeds f32
/// lanes (widened from native bytes) and re-narrows the output lanes to NATIVE
/// bytes, so the comparison is in the same representation rlx-cpu emits.
fn webgl_expand_bytes(
    in_dims: &[usize],
    out_dims: &[usize],
    dt: DType,
    in_bytes: &[u8],
) -> Vec<u8> {
    let lanes_in = widen_bytes_to_f32(in_bytes, dt);
    let mut g = Graph::new("cexp_webgl");
    let x = g.input("x", Shape::new(in_dims, dt));
    let tgt: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let y = g.add_node(
        Op::Expand { target_shape: tgt },
        vec![x],
        Shape::new(out_dims, dt),
    );
    g.set_outputs(vec![y]);
    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let lanes_out = rlx_webgl::run_cpu(&plan, &[("x", &lanes_in)])
        .expect("run_cpu")
        .remove(0);
    narrow_f32_to_bytes(&lanes_out, dt)
}

#[test]
fn expand_complex_materialized() {
    // C64[1,2] -> [3,2]: broadcast the OUTER dim. 2 complex elems [re, im].
    let c64 = [1.5f32, 2.5, -3.0, 4.0];
    let cpu = cpu_expand_bytes(&[1, 2], &[3, 2], DType::C64, &c64_bytes(&c64));
    let webgl = webgl_expand_bytes(&[1, 2], &[3, 2], DType::C64, &c64_bytes(&c64));
    assert_eq!(
        webgl, cpu,
        "C64 materialized Expand [1,2]->[3,2] webgl vs cpu mismatch"
    );

    // C128[2,1] -> [2,3]: broadcast the INNER dim (f32-exact df64 values).
    let c128 = [1.5f64, -2.5, 3.25, -4.75];
    let cpu = cpu_expand_bytes(&[2, 1], &[2, 3], DType::C128, &c128_bytes(&c128));
    let webgl = webgl_expand_bytes(&[2, 1], &[2, 3], DType::C128, &c128_bytes(&c128));
    assert_eq!(
        webgl, cpu,
        "C128 materialized Expand [2,1]->[2,3] webgl vs cpu mismatch"
    );
}

// --- Other element-indexed movement ops must be lane-aware too ----------------
// Expand was one of a CLASS: every element-indexed movement op (Transpose,
// Narrow, Concat, Gather, …) copies one f32 per ELEMENT and shatters complex
// unless made lane-aware. These gates (ported from rlx-wgpu's complex_parity)
// pin the webgl planner fix for the common single-/two-input movement ops.
// Golden = rlx-cpu native output bytes; candidate = webgl `build_plan` + native
// `run_cpu`, compared in the shared native representation via the df64 boundary.

/// rlx-cpu golden for a single-input op → NATIVE bytes.
fn cpu_op1_bytes(in_dims: &[usize], out_dims: &[usize], dt: DType, op: Op, xb: &[u8]) -> Vec<u8> {
    let mut g = Graph::new("cop1_ref");
    let x = g.input("x", Shape::new(in_dims, dt));
    let y = g.add_node(op, vec![x], Shape::new(out_dims, dt));
    g.set_outputs(vec![y]);
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
    c.run_typed(&[("x", xb, dt)]).remove(0).0
}
/// webgl candidate for a single-input op → NATIVE bytes (narrowed).
fn webgl_op1_bytes(in_dims: &[usize], out_dims: &[usize], dt: DType, op: Op, xb: &[u8]) -> Vec<u8> {
    let lanes_in = widen_bytes_to_f32(xb, dt);
    let mut g = Graph::new("cop1_webgl");
    let x = g.input("x", Shape::new(in_dims, dt));
    let y = g.add_node(op, vec![x], Shape::new(out_dims, dt));
    g.set_outputs(vec![y]);
    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let lanes_out = rlx_webgl::run_cpu(&plan, &[("x", &lanes_in)])
        .expect("run_cpu")
        .remove(0);
    narrow_f32_to_bytes(&lanes_out, dt)
}

#[test]
fn transpose_complex() {
    // C64[2,3] --perm[1,0]--> [3,2]. 6 complex elems; a naive per-f32 reindex
    // reads lane `i` of the WRONG element and shatters the [re, im] pairing.
    let c64: Vec<f32> = (0..12).map(|i| i as f32 + 0.5).collect();
    let op = Op::Transpose { perm: vec![1, 0] };
    let cpu = cpu_op1_bytes(&[2, 3], &[3, 2], DType::C64, op.clone(), &c64_bytes(&c64));
    let webgl = webgl_op1_bytes(&[2, 3], &[3, 2], DType::C64, op, &c64_bytes(&c64));
    assert_eq!(webgl, cpu, "C64 Transpose webgl vs cpu mismatch");
}

#[test]
fn narrow_complex() {
    // C64[4] --narrow axis0 [1..3)--> [2]. Keep complex elems 1,2.
    let c64 = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let op = Op::Narrow {
        axis: 0,
        start: 1,
        len: 2,
    };
    let cpu = cpu_op1_bytes(&[4], &[2], DType::C64, op.clone(), &c64_bytes(&c64));
    let webgl = webgl_op1_bytes(&[4], &[2], DType::C64, op, &c64_bytes(&c64));
    assert_eq!(webgl, cpu, "C64 Narrow webgl vs cpu mismatch");
}

#[test]
fn concat_complex() {
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
        let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
        c.run_typed(&[
            ("a", &c64_bytes(&a), DType::C64),
            ("b", &c64_bytes(&b), DType::C64),
        ])
        .remove(0)
        .0
    };
    let webgl = {
        let al = widen_bytes_to_f32(&c64_bytes(&a), DType::C64);
        let bl = widen_bytes_to_f32(&c64_bytes(&b), DType::C64);
        let mut g = Graph::new("ccat_webgl");
        let x = g.input("a", Shape::new(&[2], DType::C64));
        let y = g.input("b", Shape::new(&[3], DType::C64));
        let z = g.add_node(
            Op::Concat { axis: 0 },
            vec![x, y],
            Shape::new(&[5], DType::C64),
        );
        g.set_outputs(vec![z]);
        let plan = rlx_webgl::build_plan(&g).expect("plan");
        let out = rlx_webgl::run_cpu(&plan, &[("a", &al), ("b", &bl)])
            .expect("run_cpu")
            .remove(0);
        narrow_f32_to_bytes(&out, DType::C64)
    };
    assert_eq!(webgl, cpu, "C64 Concat webgl vs cpu mismatch");
}

#[test]
fn gather_complex() {
    // C64[4] table --gather axis0 by idx [2,0,3,1]--> C64[4]. Each complex
    // element is 2 f32 lanes [re, im]; a naive per-f32 gather reads lane `i` of
    // the WRONG element and shatters the pairing. Indices MUST be I64 — rlx-cpu's
    // exec_gather reads I32 as f32 (garbage), which would collapse the golden.
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
        let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
        c.run_typed(&[
            ("t", &c64_bytes(&table), DType::C64),
            ("ix", &idx_bytes, DType::I64),
        ])
        .remove(0)
        .0
    };
    let webgl = {
        let tl = widen_bytes_to_f32(&c64_bytes(&table), DType::C64);
        let il = widen_bytes_to_f32(&idx_bytes, DType::I64);
        let mut g = Graph::new("cgat_webgl");
        let t = g.input("t", Shape::new(&[4], DType::C64));
        let ix = g.input("ix", Shape::new(&[4], DType::I64));
        let z = g.add_node(
            Op::Gather { axis: 0 },
            vec![t, ix],
            Shape::new(&[4], DType::C64),
        );
        g.set_outputs(vec![z]);
        let plan = rlx_webgl::build_plan(&g).expect("plan");
        let out = rlx_webgl::run_cpu(&plan, &[("t", &tl), ("ix", &il)])
            .expect("run_cpu")
            .remove(0);
        narrow_f32_to_bytes(&out, DType::C64)
    };
    assert_eq!(webgl, cpu, "C64 Gather webgl vs cpu mismatch");
}
