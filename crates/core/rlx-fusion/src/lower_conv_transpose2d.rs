// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::ConvTranspose2d` to a dilate + flip + ordinary `Op::Conv`.
//!
//! Only Metal and CUDA carry a native transposed-convolution kernel. Every
//! other backend rejects the op outright, which means a single `nn.
//! ConvTranspose2d` anywhere in a model costs the whole backend — Real-CUGAN's
//! U-Nets use four of them, so the entire anime-upscaler family was
//! Metal/CUDA-only for want of a decomposition.
//!
//! # The identity
//!
//! A transposed convolution is an ordinary convolution over a *dilated* and
//! *re-padded* input, with the kernel flipped and its channel axes swapped:
//!
//! ```text
//!   dilate(x, s)      insert s−1 zeros between adjacent pixels
//!   pad by k − 1 − p  (plus `output_padding` on the trailing edge only)
//!   conv with w', stride 1, pad 0
//!
//!   w' = reverse(transpose(w, [1, 0, 2, 3]), axes = [2, 3])
//! ```
//!
//! because `ConvTranspose2d` stores its weight as `[C_in, C_out/groups, kH, kW]`
//! — input channels first, the opposite of `Conv` — and a transposed
//! convolution correlates with the *mirrored* kernel.
//!
//! Output extent is unchanged by the rewrite:
//! `(H − 1)·s − 2p + d·(k − 1) + 1 + output_padding`.
//!
//! # `output_padding` is asymmetric
//!
//! It exists to disambiguate which of the `s` input sizes that map to the same
//! output size was meant, so it is added to the **bottom/right only**. Padding
//! it symmetrically produces a plausible, differently-sized image.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

/// Insert `s − 1` zeros between adjacent elements along `axis`.
///
/// Composed from `concat` + `reshape` rather than a scatter: the zero block is
/// a `full`, and interleaving is a reshape of the concatenated pair. The
/// trailing zeros (after the last real element) are then narrowed off, which is
/// what makes the result `(n − 1)·s + 1` rather than `n·s`.
fn dilate(g: &mut Graph, x: NodeId, axis: usize, s: usize) -> NodeId {
    if s <= 1 {
        return x;
    }
    let dims: Vec<usize> = g
        .shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let n = dims[axis];
    if n == 0 {
        return x;
    }
    let dtype = g.shape(x).dtype();

    // [..., n, ...] → [..., n, 1, ...] so a zero block can be concatenated on
    // the new axis, giving [..., n, s, ...] = the dilated run plus a tail.
    let mut expanded = dims.clone();
    expanded.insert(axis + 1, 1);
    let e: Vec<i64> = expanded.iter().map(|&d| d as i64).collect();
    let unit = g.reshape_(x, e);

    let mut zdims = expanded.clone();
    zdims[axis + 1] = s - 1;
    let zeros = g.full(&zdims, 0.0, dtype);
    let interleaved = g.concat_(vec![unit, zeros], axis + 1);

    let mut merged = dims.clone();
    merged[axis] = n * s;
    let m: Vec<i64> = merged.iter().map(|&d| d as i64).collect();
    let flat = g.reshape_(interleaved, m);

    // Drop the `s − 1` zeros that follow the final element.
    g.narrow_(flat, axis, 0, (n - 1) * s + 1)
}

/// Decompose one `ConvTranspose2d` (inputs already remapped into `g`).
#[allow(clippy::too_many_arguments)]
pub fn lower_conv_transpose2d(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    kernel_size: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    output_padding: &[usize],
    groups: usize,
) -> NodeId {
    let (kh, kw) = (kernel_size[0], kernel_size[1]);
    let (sh, sw) = (stride[0], stride[1]);
    let (ph, pw) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);
    let (oph, opw) = (
        output_padding.first().copied().unwrap_or(0),
        output_padding.get(1).copied().unwrap_or(0),
    );

    // Dilate the spatial axes of NCHW.
    let x = dilate(g, x, 2, sh);
    let x = dilate(g, x, 3, sw);

    // `k − 1 − p` on every side, then `output_padding` on the trailing edge.
    // A negative amount would mean the original padding exceeded the kernel
    // reach, which `ConvTranspose2d` does not permit.
    let before_h = dh * (kh - 1) - ph;
    let before_w = dw * (kw - 1) - pw;
    let x = g.pad_(
        x,
        vec![
            [0, 0],
            [0, 0],
            [before_h, before_h + oph],
            [before_w, before_w + opw],
        ],
        rlx_ir::op::PadMode::Constant(0.0),
    );

    // Weight layouts differ between the two ops, and for grouped convolutions
    // a plain axis swap is *not* enough:
    //
    //   ConvTranspose2d  [C_in,  C_out/g, kH, kW]   input channels lead
    //   Conv             [C_out, C_in/g,  kH, kW]   output channels lead
    //
    // `C_in` is what splits into groups, so it has to be unpacked first:
    // `[g, C_in/g, C_out/g, …]` → swap the two channel axes → repack as
    // `[C_out, C_in/g, …]`. At `g = 1` this collapses to the plain swap, but
    // writing only the plain swap makes every grouped model fail Conv's own
    // shape check — which is how this was caught.
    let wdims: Vec<usize> = g
        .shape(w)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let (cin, cout_g) = (wdims[0], wdims[1]);
    let wf = if groups == 1 {
        let wt = g.transpose_(w, vec![1, 0, 2, 3]);
        let wshape = g.shape(wt).clone();
        g.add_node(Op::Reverse { axes: vec![2, 3] }, vec![wt], wshape)
    } else {
        let split = g.reshape_(
            w,
            vec![
                groups as i64,
                (cin / groups) as i64,
                cout_g as i64,
                kh as i64,
                kw as i64,
            ],
        );
        let swapped = g.transpose_(split, vec![0, 2, 1, 3, 4]);
        let packed = g.reshape_(
            swapped,
            vec![
                (groups * cout_g) as i64,
                (cin / groups) as i64,
                kh as i64,
                kw as i64,
            ],
        );
        let wshape = g.shape(packed).clone();
        g.add_node(Op::Reverse { axes: vec![2, 3] }, vec![packed], wshape)
    };

    g.conv2d(x, wf, [kh, kw], [1, 1], [0, 0], [dh, dw], groups)
}

/// Rewrite every `Op::ConvTranspose2d` into primitives.
pub struct LowerConvTranspose2d;

impl Pass for LowerConvTranspose2d {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::ConvTranspose2d]
    }

    fn name(&self) -> &str {
        "lower_conv_transpose2d"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ConvTranspose2d { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::ConvTranspose2d {
                kernel_size,
                stride,
                padding,
                dilation,
                output_padding,
                groups,
            } = &node.op
            {
                let x = id_map[&node.inputs[0]];
                let w = id_map[&node.inputs[1]];
                lower_conv_transpose2d(
                    &mut new_graph,
                    x,
                    w,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    output_padding,
                    *groups,
                )
            } else {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}
