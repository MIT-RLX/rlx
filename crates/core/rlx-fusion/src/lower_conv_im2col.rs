// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower 2-D `Op::Conv` to `Op::Im2Col` + matrix products, for backends that
//! have a matmul and an im2col but no convolution kernel (the XDNA NPU: its
//! AIE array is a GEMM engine, and a convolution there *is* im2col → GEMM).
//!
//! With `x [N, C, H, W]`, weight `[O, C/g, kH, kW]`, `S = H_out·W_out` and
//! `K = kH·kW`, the im2col matrix is `col [N·S, C·K]` (row `n·S + s`, column
//! `c·K + k`):
//!
//! * `g = 1`: `col · reshape(w, [O, C·K])ᵀ` → `[N·S, O]`.
//! * depthwise (`C/g = 1`, `O = C·m`): no matmul — `col` viewed as
//!   `[N·S, C, 1, K]` times `w` as `[1, C, m, K]`, summed over `K`.
//! * other groups: per-group batched matmul, `[g, N·S, (C/g)·K] ·
//!   [g, (C/g)·K, O/g]`, channels regrouped group-major as `Conv` orders them.
//!
//! then `[N·S, O]` → `[N, O, H_out, W_out]`.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

fn dims(g: &Graph, x: NodeId) -> Vec<usize> {
    g.shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect()
}

fn i64s(v: &[usize]) -> Vec<i64> {
    v.iter().map(|&d| d as i64).collect()
}

/// Decompose one 2-D `Conv` (inputs already remapped into `g`). `None` when
/// the shapes are not static NCHW (left as `Conv`).
pub fn lower_conv_im2col(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    kernel_size: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    groups: usize,
) -> Option<NodeId> {
    if kernel_size.len() != 2 || g.shape(x).rank() != 4 || g.shape(w).rank() != 4 {
        return None;
    }
    if g.shape(x)
        .dims()
        .iter()
        .any(|d| !matches!(d, shape::Dim::Static(_)))
    {
        return None;
    }
    let xd = dims(g, x);
    let wd = dims(g, w);
    let (n, c, h, wi) = (xd[0], xd[1], xd[2], xd[3]);
    let (o, cg, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
    let groups = groups.max(1);
    if cg * groups != c || o % groups != 0 {
        return None;
    }
    let at = |v: &[usize], i: usize, d: usize| v.get(i).copied().unwrap_or(d);
    let (sh, sw) = (at(stride, 0, 1), at(stride, 1, 1));
    let (ph, pw) = (at(padding, 0, 0), at(padding, 1, 0));
    let (dh, dw) = (at(dilation, 0, 1), at(dilation, 1, 1));
    let ho = shape::conv2d_spatial_output(h, kh, sh, ph, dh);
    let wo = shape::conv2d_spatial_output(wi, kw, sw, pw, dw);
    let (s, k) = (ho * wo, kh * kw);
    let rows = n * s;

    let col_shape =
        shape::im2col_output_shape(g.shape(x), [kh, kw], [sh, sw], [ph, pw], [dh, dw]).ok()?;
    let col = g.add_node(
        Op::Im2Col {
            kernel_size: vec![kh, kw],
            stride: vec![sh, sw],
            padding: vec![ph, pw],
            dilation: vec![dh, dw],
        },
        vec![x],
        col_shape,
    );

    let y = if groups == 1 {
        // [rows, C·K] · [C·K, O]
        let wm = g.reshape_(w, i64s(&[o, c * k]));
        let wt = g.transpose_(wm, vec![1, 0]);
        g.mm(col, wt)
    } else if cg == 1 {
        // Depthwise: [rows, C, 1, K] · [1, C, m, K] → Σ_K → [rows, C·m].
        let m = o / c;
        let c4 = g.reshape_(col, i64s(&[rows, c, 1, k]));
        let w4 = g.reshape_(w, i64s(&[1, c, m, k]));
        let prod = g.mul(c4, w4);
        let summed = g.sum(prod, vec![3], false);
        g.reshape_(summed, i64s(&[rows, o]))
    } else {
        let og = o / groups;
        let c3 = g.reshape_(col, i64s(&[rows, groups, cg * k]));
        let cb = g.transpose_(c3, vec![1, 0, 2]); // [g, rows, Cg·K]
        let w3 = g.reshape_(w, i64s(&[groups, og, cg * k]));
        let wb = g.transpose_(w3, vec![0, 2, 1]); // [g, Cg·K, Og]
        let yb = g.mm(cb, wb); // [g, rows, Og]
        let yt = g.transpose_(yb, vec![1, 0, 2]);
        g.reshape_(yt, i64s(&[rows, o]))
    };
    let y4 = g.reshape_(y, i64s(&[n, ho, wo, o]));
    Some(g.transpose_(y4, vec![0, 3, 1, 2]))
}

/// Rewrite every static 2-D `Op::Conv` into im2col + matrix products.
pub struct LowerConvIm2col;

impl Pass for LowerConvIm2col {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Conv]
    }

    fn name(&self) -> &str {
        "lower_conv_im2col"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Conv { .. }))
        {
            return graph;
        }
        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            let lowered = if let Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } = &node.op
                && node.inputs.len() == 2
            {
                let x = id_map[&node.inputs[0]];
                let w = id_map[&node.inputs[1]];
                lower_conv_im2col(
                    &mut new_graph,
                    x,
                    w,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    *groups,
                )
            } else {
                None
            };
            let new_id = lowered.unwrap_or_else(|| {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
            });
            id_map.insert(node.id, new_id);
        }
        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}
