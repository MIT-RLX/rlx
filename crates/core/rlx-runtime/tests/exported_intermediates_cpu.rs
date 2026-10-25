// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An intermediate exported as a graph output keeps its own value on CPU.
//!
//! Two rewrites dropped it. `FuseMatMulBiasAct` absorbed an exported matmul,
//! so that output returned `x·w + b`. And the thunk-level Narrow→Attention /
//! Narrow→Rope rewrites turn a Narrow into a Nop when its only *thunk* reader
//! is the consumer; a graph output is read by the caller, which they did not
//! count, so an exported Q/K/V slice came back all zeros. Both were found by
//! `rlx-stablehlo`'s IREE round trip, whose bisect exports every op.

use rlx_ir::op::{BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

#[test]
fn exported_matmul_is_not_the_biased_sum() {
    let mut g = Graph::new("mm_bias");
    let x = g.input("x", Shape::new(&[1, 2, 3], F));
    let w = g.param("w", Shape::new(&[3, 2], F));
    let b = g.param("b", Shape::new(&[2], F));
    let m = g.matmul(x, w, Shape::new(&[1, 2, 2], F));
    let y = g.binary(BinaryOp::Add, m, b, Shape::new(&[1, 2, 2], F));
    g.set_outputs(vec![y, m]);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("w", &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    c.set_param("b", &[10.0, 20.0]);
    let o = c.run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    assert_eq!(o[1], vec![4.0, 5.0, 10.0, 11.0], "x·w");
    assert_eq!(o[0], vec![14.0, 25.0, 20.0, 31.0], "x·w + b");
}

#[test]
fn exported_qkv_slice_survives_the_strided_attention_rewrite() {
    let mut g = Graph::new("qkv");
    let x = g.input("x", Shape::new(&[1, 4, 24], F));
    let mask = g.input("mask", Shape::new(&[1, 4], F));
    let s = Shape::new(&[1, 4, 8], F);
    let narrow = |g: &mut Graph, start| {
        g.add_node(
            Op::Narrow {
                axis: 2,
                start,
                len: 8,
            },
            vec![x],
            s.clone(),
        )
    };
    let q = narrow(&mut g, 0);
    let k = narrow(&mut g, 8);
    let v = narrow(&mut g, 16);
    let a = g.add_node(
        Op::Attention {
            num_heads: 2,
            head_dim: 4,
            v_head_dim: None,
            mask_kind: MaskKind::Custom,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v, mask],
        s.clone(),
    );
    g.set_outputs(vec![a, q]);
    let xs: Vec<f32> = (0..96).map(|i| i as f32 * 0.01).collect();
    let mut c = Session::new(Device::Cpu).compile(g);
    let o = c.run(&[("x", &xs), ("mask", &[1.0; 4])]);
    let want: Vec<f32> = (0..4)
        .flat_map(|r| (0..8).map(move |j| xs_at(r * 24 + j)))
        .collect();
    assert_eq!(o[1], want);
}

fn xs_at(i: usize) -> f32 {
    i as f32 * 0.01
}
