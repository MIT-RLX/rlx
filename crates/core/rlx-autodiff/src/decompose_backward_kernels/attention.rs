// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Primitive compositions for training `*Backward` ops (higher-order AD).

//! `attention` — extracted from the `decompose_backward_kernels` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AttentionBwdWrt, CmpOp, MaskKind, SteKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use super::*;

/// Build attention backward via reverse-mode on `sum(Attention(q,k,v) ⊙ dy)`.
pub fn compose_attention_backward(
    wrt: AttentionBwdWrt,
    q_shape: &Shape,
    k_shape: &Shape,
    v_shape: &Shape,
    dy_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
    mask_kind: MaskKind,
    mask_shape: Option<&Shape>,
) -> Graph {
    let mut sub = Graph::new("attn_bwd_decomp");
    let q = sub.input("q", q_shape.clone());
    let k = sub.input("k", k_shape.clone());
    let v = sub.input("v", v_shape.clone());
    let dy = sub.input("dy", dy_shape.clone());
    let mask_node = if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
        Some(
            sub.input(
                "mask",
                mask_shape
                    .cloned()
                    .expect("Custom/Bias attention decompose requires mask shape"),
            ),
        )
    } else {
        None
    };

    let (q, k, v, dy, q_shape) = reshape_attn_rank4(&mut sub, q, k, v, dy, num_heads, head_dim);
    let k_shape = sub.node(k).shape.clone();
    let dy_shape = sub.node(dy).shape.clone();
    let q_seq = q_shape.dim(2).unwrap_static();
    let k_seq = k_shape.dim(2).unwrap_static();

    let y = expand_attention_forward_primitives(
        &mut sub, q, k, v, num_heads, head_dim, &dy_shape, q_seq, k_seq, mask_kind, mask_node,
        mask_shape,
    );

    let prod = sub.mul(y, dy);
    let rank = dy_shape.rank();
    let loss = sub.sum(prod, (0..rank).collect(), false);
    sub.set_outputs(vec![loss]);

    let prep = prepare_graph_for_ad(sub);
    let wrt_id = match wrt {
        AttentionBwdWrt::Query => q,
        AttentionBwdWrt::Key => k,
        AttentionBwdWrt::Value => v,
    };
    let mut bwd = grad_with_loss(&prep, &[wrt_id]);
    crate::compose::internalize_d_output(&mut bwd);
    let grad_out = bwd.outputs[1];
    bwd.set_outputs(vec![grad_out]);
    bwd
}

/// Merge an attention-backward subgraph into `g` at the given bindings.
pub fn emit_attention_backward(
    g: &mut Graph,
    wrt: AttentionBwdWrt,
    inputs: &[NodeId],
    _out_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
    mask_kind: MaskKind,
) -> NodeId {
    let (q, k, v, dy, mask_in) = match inputs {
        [q, k, v, dy] => (*q, *k, *v, *dy, None),
        [q, k, v, dy, mask] => (*q, *k, *v, *dy, Some(*mask)),
        _ => panic!("AttentionBackward expects [q, k, v, dy] or [q, k, v, dy, mask]"),
    };
    let mask_shape = mask_in.map(|m| g.node(m).shape.clone());
    let sub = compose_attention_backward(
        wrt,
        &g.node(q).shape.clone(),
        &g.node(k).shape.clone(),
        &g.node(v).shape.clone(),
        &g.node(dy).shape.clone(),
        num_heads,
        head_dim,
        mask_kind,
        mask_shape.as_ref(),
    );
    let mut bind = std::collections::HashMap::from([
        ("q".to_string(), q),
        ("k".to_string(), k),
        ("v".to_string(), v),
        ("dy".to_string(), dy),
    ]);
    if let Some(mask) = mask_in {
        bind.insert("mask".to_string(), mask);
    }
    let id_map = merge_subgraph(g, &sub, &bind);
    id_map[&sub.outputs[0]]
}
