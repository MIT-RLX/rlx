// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::AxialRope2d` (SAM2 axial 2-D RoPE) to primitives. Semantic
//! oracle for every backend that does not rotate natively.
//!
//! Every output element is a fixed linear combination of exactly **two** input
//! elements — the element itself and its interleaved GptJ partner:
//!
//! ```text
//! out[i] = coefA[i]·x[i] + coefB[i]·x[pair[i]]
//! ```
//!
//! `coefA` / `coefB` (the `cos` / `±sin` of the axial angle) and `pair` (the
//! even↔odd partner index) are compile-time constants built from the *same*
//! frequency tables the CPU kernel (`apply_axial_rope2d`) uses, so this is a
//! bit-exact transcription of the reference loop — not an approximation. The
//! whole op collapses to one `gather` + two `mul` + one `add`.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::BinaryOp;
use rlx_ir::*;
use std::collections::HashMap;

/// Static dims of a shape (these ops are always statically shaped).
fn static_dims(s: &Shape) -> Vec<usize> {
    s.dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            _ => panic!("AxialRope2d lowering requires static dims"),
        })
        .collect()
}

/// Build the `(coefA, coefB, pair)` constant tables for one **single-batch**
/// `[n_tokens, num_heads*head_dim]` block (element-flattened order).
fn axial_tables(
    n_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    end_x: usize,
    theta: f32,
    repeat: usize,
) -> (Vec<f32>, Vec<f32>, Vec<i64>) {
    let half = head_dim / 2;
    let q4 = head_dim / 4;
    let hs = num_heads * head_dim;
    let total = n_tokens * hs;
    let mut coef_a = vec![0f32; total];
    let mut coef_b = vec![0f32; total];
    let mut pair = vec![0i64; total];

    let mut freq = vec![0f32; q4];
    for (i, f) in freq.iter_mut().enumerate() {
        *f = 1.0 / theta.powf((4 * i) as f32 / head_dim as f32);
    }

    for tok in 0..n_tokens {
        let pos = tok / repeat.max(1);
        let tx = (pos % end_x) as f32;
        let ty = (pos / end_x) as f32;
        for h in 0..num_heads {
            let base = tok * hs + h * head_dim;
            for d in 0..head_dim {
                let idx = base + d;
                // Which half (X vs Y), quarter-channel `c`, even/odd role.
                let (c, role, is_x) = if d < half {
                    (d / 2, d % 2, true)
                } else {
                    let dd = d - half;
                    (dd / 2, dd % 2, false)
                };
                let coord = if is_x { tx } else { ty };
                let ang = coord * freq[c];
                let co = ang.cos();
                let si = ang.sin();
                coef_a[idx] = co;
                if role == 0 {
                    // out[i0] = x0·cos − x1·sin ; partner is the odd sibling.
                    coef_b[idx] = -si;
                    pair[idx] = (idx + 1) as i64;
                } else {
                    // out[i1] = x0·sin + x1·cos ; partner is the even sibling.
                    coef_b[idx] = si;
                    pair[idx] = (idx - 1) as i64;
                }
            }
        }
    }
    (coef_a, coef_b, pair)
}

/// Decompose one `Op::AxialRope2d` (input `x` already remapped into `g`).
#[allow(clippy::too_many_arguments)]
pub fn lower_axial_rope2d(
    g: &mut Graph,
    x: NodeId,
    end_x: usize,
    end_y: usize,
    head_dim: usize,
    num_heads: usize,
    theta: f32,
    repeat_factor: usize,
) -> NodeId {
    let out_shape = g.shape(x).clone();
    let dims = static_dims(&out_shape);
    let numel: usize = dims.iter().product();
    let repeat = repeat_factor.max(1);
    let n_tokens = end_x * end_y * repeat;
    let hs = num_heads * head_dim;
    let block = n_tokens * hs; // one batch's worth of elements
    let batch = numel / block; // ≥ 1; tables tile across batch

    let (a1, b1, p1) = axial_tables(n_tokens, num_heads, head_dim, end_x, theta, repeat);

    // Tile per-batch tables; `pair` gets a per-batch offset into the flat array.
    let mut coef_a = Vec::with_capacity(numel);
    let mut coef_b = Vec::with_capacity(numel);
    let mut pair = Vec::with_capacity(numel);
    for b in 0..batch {
        coef_a.extend_from_slice(&a1);
        coef_b.extend_from_slice(&b1);
        let off = (b * block) as i64;
        pair.extend(p1.iter().map(|p| p + off));
    }

    let x_flat = g.reshape_(x, vec![numel as i64]);
    let f32sh = Shape::new(&[numel], DType::F32);
    let i64sh = Shape::new(&[numel], DType::I64);

    let a_bytes: Vec<u8> = coef_a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let a_node = g.add_node(Op::Constant { data: a_bytes }, vec![], f32sh.clone());
    let b_bytes: Vec<u8> = coef_b.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_node = g.add_node(Op::Constant { data: b_bytes }, vec![], f32sh.clone());
    let p_bytes: Vec<u8> = pair.iter().flat_map(|v| v.to_le_bytes()).collect();
    let p_node = g.add_node(Op::Constant { data: p_bytes }, vec![], i64sh);

    let xg = g.gather_(x_flat, p_node, 0);
    let t1 = g.binary(BinaryOp::Mul, x_flat, a_node, f32sh.clone());
    let t2 = g.binary(BinaryOp::Mul, xg, b_node, f32sh.clone());
    let out_flat = g.binary(BinaryOp::Add, t1, t2, f32sh);

    let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    g.reshape_(out_flat, dims_i64)
}

/// Rewrite every `Op::AxialRope2d` node into primitives.
pub struct LowerAxialRope2d;

impl Pass for LowerAxialRope2d {
    // Lifted from the scan `run` already performs: without these kinds
    // the pass rebuilds the graph node-for-node and returns it unchanged.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::AxialRope2d]
    }

    fn name(&self) -> &str {
        "lower_axial_rope2d"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::AxialRope2d { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::AxialRope2d {
                end_x,
                end_y,
                head_dim,
                num_heads,
                theta,
                repeat_factor,
            } = &node.op
            {
                let x = id_map[&node.inputs[0]];
                lower_axial_rope2d(
                    &mut new_graph,
                    x,
                    *end_x,
                    *end_y,
                    *head_dim,
                    *num_heads,
                    *theta,
                    *repeat_factor,
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
