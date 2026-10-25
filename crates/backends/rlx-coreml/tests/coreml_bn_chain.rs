// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An eval-mode BatchNorm written out elementwise after a convolution —
//! `conv → +b → (x − mean) / √(var + ε) · γ + β` — segfaulted CoreML inside
//! `libBNNS` while it initialised the program (LEAF's tokenizer at
//! `dim_cnn = 8`, 6 electrodes). `hybrid::recip_bn_div_after_conv` turns that
//! division into a reciprocal multiply. Before it, every case below except the
//! 65-electrode one crashed the test process (SIGSEGV); each is checked
//! against rlx-cpu.

use rlx_coreml::CoremlExecutable;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + salt) * 37 % 101) as f32) * 0.02 - 1.0)
        .collect()
}

struct Case {
    g: Graph,
    params: Vec<(String, Vec<f32>)>,
    x_len: usize,
}

fn param(
    g: &mut Graph,
    ps: &mut Vec<(String, Vec<f32>)>,
    name: &str,
    dims: &[usize],
    positive: bool,
) -> NodeId {
    let n: usize = dims.iter().product();
    let mut v = data(n, ps.len() * 13 + 1);
    if positive {
        v.iter_mut().for_each(|x| *x = x.abs() + 0.5);
    }
    ps.push((name.to_string(), v));
    g.param(name, Shape::new(dims, F))
}

/// LEAF's tokenizer: `[B, C, N·W]` → windows → temporal conv → spatial conv →
/// eval BN (elementwise) → GELU → max-pool → `[B, N, F·(W+1)/pool]`.
fn tokenizer(b: usize, c: usize, n: usize, f: usize) -> Case {
    let (w, pool) = (100, 10);
    let mut g = Graph::new("bn-chain");
    let mut ps = Vec::new();
    let x = g.input("x", Shape::new(&[b, c, n * w], F));
    let x = g.reshape_(x, vec![b as i64, c as i64, n as i64, w as i64]);
    let x = g.transpose_(x, vec![0, 2, 1, 3]);
    let x = g.reshape_(x, vec![(b * n) as i64, 1, c as i64, w as i64]);
    let w0 = param(&mut g, &mut ps, "w0", &[f, 1, 1, 40], false);
    let y = g.conv2d(x, w0, [1, 40], [1, 1], [0, 20], [1, 1], 1);
    let b0 = param(&mut g, &mut ps, "b0", &[f], false);
    let b0 = g.reshape_(b0, vec![1, f as i64, 1, 1]);
    let y = g.add(y, b0);
    let w1 = param(&mut g, &mut ps, "w1", &[f, f, c, 1], false);
    let y = g.conv2d(y, w1, [c, 1], [1, 1], [0, 0], [1, 1], 1);
    let b1 = param(&mut g, &mut ps, "b1", &[f], false);
    let b1 = g.reshape_(b1, vec![1, f as i64, 1, 1]);
    let y = g.add(y, b1);
    let r4 = |g: &mut Graph, t: NodeId| g.reshape_(t, vec![1, f as i64, 1, 1]);
    let rm = param(&mut g, &mut ps, "rm", &[f], false);
    let rv = param(&mut g, &mut ps, "rv", &[f], true);
    let bw = param(&mut g, &mut ps, "bw", &[f], false);
    let bb = param(&mut g, &mut ps, "bb", &[f], false);
    let (rm, rv, bw, bb) = (
        r4(&mut g, rm),
        r4(&mut g, rv),
        r4(&mut g, bw),
        r4(&mut g, bb),
    );
    let y = g.sub(y, rm);
    let eps = g.constant(1e-5, F);
    let den = g.add(rv, eps);
    let sd = g.sqrt(den);
    let y = g.div(y, sd);
    let y = g.mul(y, bw);
    let y = g.add(y, bb);
    let y = g.gelu(y);
    let y = g.pool2d(y, ReduceOp::Max, [1, pool], [1, pool], [0, 0]);
    let y = g.reshape_(y, vec![b as i64, n as i64, (f * ((w + 1) / pool)) as i64]);
    g.set_outputs(vec![y]);
    Case {
        g,
        params: ps,
        x_len: b * c * n * w,
    }
}

fn check(case: Case) {
    let x = data(case.x_len, 7);
    let mut cpu = Session::new(Device::Cpu).compile(case.g.clone());
    for (n, v) in &case.params {
        cpu.set_param(n, v);
    }
    cpu.finalize_params();
    let want = cpu.run(&[("x", &x)]).remove(0);
    let mut exe = CoremlExecutable::compile(case.g);
    for (n, v) in &case.params {
        exe.set_param(n, v);
    }
    exe.finalize().expect("coreml finalize");
    let got = exe.run(&[("x", &x)]).expect("coreml run").remove(0);
    assert_eq!(got.len(), want.len());
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    assert!(err <= 1e-4 * scale, "max_err {err} (scale {scale})");
}

#[test]
fn bn_chain_small_width() {
    // LEAF's train_parity config: 6 electrodes, 4 windows, 8 filters.
    check(tokenizer(1, 6, 4, 8));
}

#[test]
fn bn_chain_small_width_batched() {
    check(tokenizer(3, 6, 4, 8));
}

#[test]
fn bn_chain_shapes_that_crashed() {
    // (C, N, F) combinations measured to segfault without the rewrite.
    for (c, n, f) in [
        (1, 4, 8),
        (16, 4, 8),
        (32, 4, 8),
        (63, 4, 8),
        (64, 4, 8),
        (64, 20, 64),
        (6, 4, 64),
    ] {
        check(tokenizer(1, c, n, f));
    }
}

#[test]
fn bn_chain_release_width() {
    // The released LEAF config (65 electrodes, 64 filters) never crashed.
    check(tokenizer(1, 65, 20, 64));
}

/// The rewrite touches only the conv-fed `div` of the chain.
#[test]
fn rewrite_targets_only_the_conv_fed_div() {
    let count = |g: &Graph, op: BinaryOp| {
        g.nodes()
            .iter()
            .filter(|n| matches!(&n.op, Op::Binary(o) if *o == op))
            .count()
    };
    let case = tokenizer(1, 6, 4, 8);
    assert_eq!(count(&case.g, BinaryOp::Div), 1);
    let after = rlx_coreml::hybrid::recip_bn_div_after_conv(case.g);
    assert_eq!(count(&after, BinaryOp::Div), 0, "the BN div is rewritten");

    // A division with no convolution in front is left alone.
    let mut g = Graph::new("plain-div");
    let x = g.input("x", Shape::new(&[2, 3], F));
    let k = g.param("k", Shape::new(&[3], F));
    let m = g.param("m", Shape::new(&[3], F));
    let s = g.sub(x, m);
    let d = g.div(s, k);
    g.set_outputs(vec![d]);
    let after = rlx_coreml::hybrid::recip_bn_div_after_conv(g);
    assert_eq!(count(&after, BinaryOp::Div), 1, "non-conv div untouched");
}
