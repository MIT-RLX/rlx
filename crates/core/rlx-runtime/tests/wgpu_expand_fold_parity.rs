// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! wgpu reads a broadcast `Binary` operand from the tensor an `Expand` was fed
//! (`fold_expand_into_binary`) instead of from the materialised copy. Every
//! case here must equal rlx-cpu bit for bit: the fold changes which element is
//! read from where, never an operation, so any difference is an indexing error.
//!
//! The shapes are the ones a training step produces — a batch-norm's
//! `expand(gamma)`, autodiff's `expand(expand(g))` — plus the cases the fold
//! must leave alone: an `Expand` another op still reads, an `Expand` that is a
//! graph output, and a broadcast no single `(rep, len)` describes.

#![cfg(all(feature = "gpu", feature = "cpu"))]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;

fn expand(g: &mut Graph, x: NodeId, dims: &[usize]) -> NodeId {
    g.add_node(
        Op::Expand {
            target_shape: dims.iter().map(|&d| d as i64).collect(),
        },
        vec![x],
        Shape::new(dims, F),
    )
}

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + 1.0) * seed).sin() * 1.7)
        .collect()
}

fn same_bits(what: &str, build: impl Fn() -> Graph, inputs: &[(&str, Vec<f32>)]) {
    let Some(_gpu) = common::serialize_gpu() else {
        return;
    };
    let borrowed: Vec<(&str, &[f32])> = inputs.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let run = |d: Device| Session::new(d).compile(build()).run(&borrowed);
    let (gpu, cpu) = (run(Device::Gpu), run(Device::Cpu));
    assert_eq!(gpu.len(), cpu.len(), "{what}: output count");
    for (o, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        assert_eq!(g.len(), c.len(), "{what}: output {o} length");
        for (i, (a, b)) in g.iter().zip(c).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: output {o} element {i} — wgpu {a} vs cpu {b}"
            );
        }
    }
}

/// `(x * expand(gamma)) + expand(beta)` — the tail of a batch-norm.
#[test]
fn per_channel_scale_and_shift() {
    let full = [4, 8, 3, 33];
    same_bits(
        "bn affine",
        || {
            let mut g = Graph::new("bn_affine");
            let x = g.input("x", Shape::new(&full, F));
            let gamma = g.input("gamma", Shape::new(&[1, 8, 1, 1], F));
            let beta = g.input("beta", Shape::new(&[1, 8, 1, 1], F));
            let ge = expand(&mut g, gamma, &full);
            let be = expand(&mut g, beta, &full);
            let y = g.binary(BinaryOp::Mul, x, ge, Shape::new(&full, F));
            let y = g.binary(BinaryOp::Add, be, y, Shape::new(&full, F));
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("x", fill(4 * 8 * 3 * 33, 0.37)),
            ("gamma", fill(8, 1.3)),
            ("beta", fill(8, 2.9)),
        ],
    );
}

/// Autodiff's reduced-gradient broadcast: an identity `Expand` feeding a
/// widening one, read by a `Mul` against a full-size tensor.
#[test]
fn chained_expands_fold_to_their_source() {
    same_bits(
        "expand chain",
        || {
            let mut g = Graph::new("expand_chain");
            let r = g.input("r", Shape::new(&[32, 16, 1, 1], F));
            let x = g.input("x", Shape::new(&[32, 16, 1, 65], F));
            let e1 = expand(&mut g, r, &[32, 16, 1, 1]);
            let e2 = expand(&mut g, e1, &[32, 16, 1, 65]);
            let y = g.binary(BinaryOp::Mul, e2, x, Shape::new(&[32, 16, 1, 65], F));
            g.set_outputs(vec![y]);
            g
        },
        &[("r", fill(32 * 16, 0.71)), ("x", fill(32 * 16 * 65, 0.13))],
    );
}

/// The `Expand` is also read by a non-`Binary` op, or is itself an output: it
/// has to stay materialised, and the `Binary` beside it must still be right.
/// One remaining reader each — the count the dead-`Expand` sweep must not
/// mistake for zero.
#[test]
fn an_expand_with_other_readers_is_kept() {
    let full = [2, 5, 7];
    for as_output in [false, true] {
        same_bits(
            if as_output {
                "expand is an output"
            } else {
                "expand read by relu"
            },
            || {
                let mut g = Graph::new("shared_expand");
                let x = g.input("x", Shape::new(&full, F));
                let b = g.input("b", Shape::new(&[5, 1], F));
                let be = expand(&mut g, b, &full);
                let y = g.binary(BinaryOp::Sub, x, be, Shape::new(&full, F));
                let other = if as_output {
                    be
                } else {
                    g.activation(Activation::Relu, be, Shape::new(&full, F))
                };
                g.set_outputs(vec![y, other]);
                g
            },
            &[("x", fill(70, 0.5)), ("b", fill(5, 3.1))],
        );
    }
}

/// `[2,1,4,5]` against `[2,3,4,5]` keeps axes 0 and 2..: not one contiguous
/// block, so no `(rep, len)` reads it and the `Expand` must be materialised.
#[test]
fn a_non_contiguous_broadcast_stays_expanded() {
    let full = [2, 3, 4, 5];
    same_bits(
        "non-contiguous",
        || {
            let mut g = Graph::new("noncontig");
            let x = g.input("x", Shape::new(&full, F));
            let m = g.input("m", Shape::new(&[2, 1, 4, 5], F));
            let me = expand(&mut g, m, &full);
            let y = g.binary(BinaryOp::Div, x, me, Shape::new(&full, F));
            g.set_outputs(vec![y]);
            g
        },
        &[("x", fill(120, 0.29)), ("m", fill(40, 0.83))],
    );
}
