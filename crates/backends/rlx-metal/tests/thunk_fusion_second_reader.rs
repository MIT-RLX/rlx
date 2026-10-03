// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A thunk fusion may not drop a producer whose value is read again later.
//!
//! Metal's thunk-level fusions (`fuse_gdn_gated_norm`, `fuse_l2_norm`) collapse
//! a pattern into one dispatch and `Nop` out the thunks that produced its
//! intermediates. Their liveness scan deliberately stops at the fused window —
//! a full-graph scan hits an opaque thunk and conservatively rejects, which
//! would disable the fusion on the decode graphs it exists for — so a reader
//! AFTER the window was invisible to it, and only a graph-output check stood in
//! the way.
//!
//! A forward graph has no such reader, which is why this went unnoticed. A
//! BACKWARD graph re-reads the recomputed forward values: Metal returned zeros
//! for `silu(gate)` and NaN out of the L2-normalize chain, so every Qwen3.5 GDN
//! parameter gradient came back NaN or zero while the forward stayed bit-exact
//! (`rlx-qwen35`'s `trunk_backward_device_parity` covers that end to end).
//!
//! These graphs reproduce the shape of the hazard directly: the fusable pattern
//! plus one more consumer of the intermediate, which must still see the real
//! value.

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn cpu_vs_metal(g: &Graph, feeds: &[(&str, &[f32])]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let run = |d: Device| {
        let mut s = Session::new(d).compile(g.clone());
        s.run(feeds)
    };
    (run(Device::Cpu), run(Device::Metal))
}

fn check(label: &str, g: &Graph, feeds: &[(&str, &[f32])]) {
    let (cpu, met) = cpu_vs_metal(g, feeds);
    for (i, (a, b)) in cpu.iter().zip(&met).enumerate() {
        let err = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
        assert!(
            b.iter().all(|v| v.is_finite()),
            "{label}: metal output {i} is not finite"
        );
        assert!(
            err / scale < 1e-5,
            "{label}: output {i} differs by {err:.3e} (rel {:.3e}) — a fusion \
             dropped a producer whose value is still read",
            err / scale
        );
    }
}

/// `out = rms_norm(x) * silu(z)`, with `silu(z)` read a second time.
#[test]
fn gated_norm_keeps_a_second_reader_of_silu() {
    let (rows, h) = (8usize, 16usize);
    let f = DType::F32;
    let n = rows * h;
    let mut g = Graph::new("gated_norm");
    let x = g.input("x", Shape::new(&[rows, h], f));
    let z = g.input("z", Shape::new(&[rows, h], f));
    let gamma = g.input("gamma", Shape::new(&[h], f));
    let beta = g.input("beta", Shape::new(&[h], f));
    let nrm = g.rms_norm(x, gamma, beta, 1e-6);
    let sz = g.activation(Activation::Silu, z, Shape::new(&[rows, h], f));
    let out = g.binary(BinaryOp::Mul, nrm, sz, Shape::new(&[rows, h], f));
    // The second reader: without it the fusion is legitimate.
    let again = g.binary(BinaryOp::Add, sz, out, Shape::new(&[rows, h], f));
    g.set_outputs(vec![out, again]);

    let xv: Vec<f32> = (0..n).map(|i| 0.3 * ((i % 11) as f32 - 5.0)).collect();
    let zv: Vec<f32> = (0..n).map(|i| 0.2 * ((i % 7) as f32 - 3.0)).collect();
    let gv: Vec<f32> = (0..h).map(|i| 0.9 + 0.01 * i as f32).collect();
    let bv = vec![0f32; h];
    check(
        "gated norm",
        &g,
        &[
            ("x", &xv[..]),
            ("z", &zv[..]),
            ("gamma", &gv[..]),
            ("beta", &bv[..]),
        ],
    );
}

/// `out = x / max(sqrt(sum(x*x)), eps)` — the L2-normalize chain — with the
/// denominator read a second time.
#[test]
fn l2_norm_keeps_a_second_reader_of_the_denominator() {
    let (rows, h) = (8usize, 4usize);
    let f = DType::F32;
    let n = rows * h;
    let mut g = Graph::new("l2");
    let x = g.input("x", Shape::new(&[rows, h], f));
    let sq = g.binary(BinaryOp::Mul, x, x, Shape::new(&[rows, h], f));
    let ss = g.add_node(
        rlx_ir::Op::Reduce {
            op: rlx_ir::op::ReduceOp::Sum,
            axes: vec![1],
            keep_dim: true,
        },
        vec![sq],
        Shape::new(&[rows, 1], f),
    );
    let rt = g.activation(Activation::Sqrt, ss, Shape::new(&[rows, 1], f));
    let eps = g.constant(1e-6, f);
    let den = g.binary(BinaryOp::Max, rt, eps, Shape::new(&[rows, 1], f));
    let denb = g.add_node(
        rlx_ir::Op::Expand {
            target_shape: vec![rows as i64, h as i64],
        },
        vec![den],
        Shape::new(&[rows, h], f),
    );
    let out = g.binary(BinaryOp::Div, x, denb, Shape::new(&[rows, h], f));
    // Second reader of the denominator, as the backward of a normalize has.
    let again = g.binary(BinaryOp::Mul, denb, out, Shape::new(&[rows, h], f));
    g.set_outputs(vec![out, again]);

    let xv: Vec<f32> = (0..n).map(|i| 0.25 * ((i % 9) as f32 - 4.0)).collect();
    check("l2 norm", &g, &[("x", &xv[..])]);
}
