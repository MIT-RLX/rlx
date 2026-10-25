// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU `Concat` of 1- and 2-byte element types.
//!
//! The CPU `Concat` thunk moved 4-byte words whatever the dtype, so a bf16 or
//! u8 concat came out with half its elements missing and rows interleaved
//! (bf16 `[a | b]` read back as `[a1, a3, a5, b1, …, 0, 0]`). Nothing failed
//! loudly. The way it surfaced: `FuseSharedInputMatMul` concatenates the
//! weights of matmuls that share an input (an MLP's gate and up projections),
//! so a model holding bf16 weights computed garbage on CPU while every
//! single-matmul test passed.
//!
//! bf16 params stay packed at 2 B while bf16 activations are widened to 4 B,
//! so the fix has to convert between layouts, not just copy narrower bytes.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn bf16_params_concatenate_exactly() {
    let (k, n1, n2) = (4usize, 3usize, 5usize);
    let a: Vec<f32> = (0..k * n1).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..k * n2).map(|i| 100.0 + i as f32).collect();
    let mut g = Graph::new("concat_bf16");
    let pa = g.param("a", Shape::new(&[k, n1], DType::BF16));
    let pb = g.param("b", Shape::new(&[k, n2], DType::BF16));
    let c = g.add_node(
        Op::Concat { axis: 1 },
        vec![pa, pb],
        Shape::new(&[k, n1 + n2], DType::BF16),
    );
    let f = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![c],
        Shape::new(&[k, n1 + n2], DType::F32),
    );
    g.set_outputs(vec![f]);
    let mut cg = Session::new(Device::Cpu).compile(g);
    cg.set_param_typed("a", &bf16_bytes(&a), DType::BF16);
    cg.set_param_typed("b", &bf16_bytes(&b), DType::BF16);
    let got = cg.run(&[]).remove(0);
    let want: Vec<f32> = (0..k)
        .flat_map(|r| {
            a[r * n1..(r + 1) * n1]
                .iter()
                .chain(&b[r * n2..(r + 1) * n2])
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(got, want);
}

#[test]
fn u8_params_concatenate_exactly() {
    let mut g = Graph::new("concat_u8");
    let pa = g.param("a", Shape::new(&[2, 3], DType::U8));
    let pb = g.param("b", Shape::new(&[2, 5], DType::U8));
    let c = g.add_node(
        Op::Concat { axis: 1 },
        vec![pa, pb],
        Shape::new(&[2, 8], DType::U8),
    );
    let f = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![c],
        Shape::new(&[2, 8], DType::F32),
    );
    g.set_outputs(vec![f]);
    let mut cg = Session::new(Device::Cpu).compile(g);
    cg.set_param_typed("a", &[0, 1, 2, 3, 4, 5], DType::U8);
    cg.set_param_typed("b", &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19], DType::U8);
    let got = cg.run(&[]).remove(0);
    let want = [
        0., 1., 2., 10., 11., 12., 13., 14., 3., 4., 5., 15., 16., 17., 18., 19.,
    ];
    assert_eq!(got, want);
}

/// The shape that broke a bf16-resident MLP: two matmuls sharing an input,
/// fused into one over a concatenated bf16 weight.
#[test]
fn shared_input_bf16_matmuls_match_f32() {
    let (m, k, n) = (2usize, 64usize, 96usize);
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
        .collect();
    let w = |seed: usize| -> Vec<f32> {
        (0..k * n)
            .map(|i| {
                f32::from_bits(
                    ((((i * 5 + seed) % 17) as f32 - 8.0) * 0.05).to_bits() & 0xffff_0000,
                )
            })
            .collect()
    };
    let (w1, w2) = (w(0), w(3));
    let run = |dt: DType| {
        let mut g = Graph::new("shared_input");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let p1 = g.param("w1", Shape::new(&[k, n], dt));
        let p2 = g.param("w2", Shape::new(&[k, n], dt));
        let a = g.matmul(xi, p1, Shape::new(&[m, n], DType::F32));
        let b = g.matmul(xi, p2, Shape::new(&[m, n], DType::F32));
        let s = g.add_node(
            Op::Binary(rlx_ir::op::BinaryOp::Add),
            vec![a, b],
            Shape::new(&[m, n], DType::F32),
        );
        g.set_outputs(vec![s]);
        let mut cg = Session::new(Device::Cpu).compile(g);
        if dt == DType::BF16 {
            cg.set_param_typed("w1", &bf16_bytes(&w1), dt);
            cg.set_param_typed("w2", &bf16_bytes(&w2), dt);
        } else {
            cg.set_param("w1", &w1);
            cg.set_param("w2", &w2);
        }
        cg.run(&[("x", &x)]).remove(0)
    };
    let (f, h) = (run(DType::F32), run(DType::BF16));
    let err = f
        .iter()
        .zip(&h)
        .fold(0f32, |e, (a, b)| e.max((a - b).abs()));
    assert!(
        err < 1e-4,
        "bf16 shared-input matmuls differ from f32 by {err}"
    );
}
