// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Expand a batched-matmul operand that broadcasts along only some batch axes
//! to the output's batch dims.
//!
//! Batched GEMM kernels take each operand as the full batch or as one matrix
//! repeated (stride 0). `[2,1,M,K]·[1,3,K,N]` or `[2,3,M,K]·[3,K,N]` is neither:
//! the operand's matrix count is neither the output's nor 1. Backends without
//! a per-matrix index (wgpu, CUDA, ROCm) refused such shapes; expanded, the
//! operand is the full batch. An operand of lower rank whose batch the kernels
//! do take (`[8,K,N]` against `[1,8,M,N]`) gets unit axes prepended, since
//! they also want equal ranks. Shapes the kernels already take are untouched.

use std::collections::HashMap;

use rlx_ir::{Graph, NodeId, Op, OpKind, Shape};

use crate::Pass;

/// See the module docs.
pub struct ExpandPartialBatchBroadcast;

fn lead(s: &Shape) -> usize {
    (0..s.rank().saturating_sub(2))
        .map(|d| s.dim(d).unwrap_static())
        .product()
}

/// What an operand of a batched matmul needs before the kernels take it.
#[derive(Clone, Copy, PartialEq)]
enum Fix {
    None,
    /// Batch is the output's or one matrix, but at a lower rank
    /// (`[8,K,N]` against a `[1,8,M,N]` output): prepend unit axes.
    Rank,
    /// Broadcasts along only part of the batch: expand to the output's batch.
    Expand,
}

fn fixes(g: &Graph, n: &rlx_ir::Node) -> [Fix; 2] {
    if !matches!(n.op, Op::MatMul) || n.shape.rank() < 3 || n.inputs.len() != 2 {
        return [Fix::None; 2];
    }
    let out = lead(&n.shape);
    [0, 1].map(|i| {
        let s = &g.node(n.inputs[i]).shape;
        if s.rank() < 3 {
            Fix::None
        } else if lead(s) != out && lead(s) != 1 {
            Fix::Expand
        } else if s.rank() < n.shape.rank() {
            Fix::Rank
        } else {
            Fix::None
        }
    })
}

impl Pass for ExpandPartialBatchBroadcast {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::MatMul]
    }

    fn name(&self) -> &str {
        "expand_partial_batch_broadcast"
    }

    fn run(&self, g: Graph) -> Graph {
        if !g
            .nodes()
            .iter()
            .any(|n| fixes(&g, n).iter().any(|f| *f != Fix::None))
        {
            return g;
        }
        let mut out = Graph::new(g.name.clone());
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in g.nodes() {
            let mut ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            for (i, fix) in fixes(&g, node).into_iter().enumerate() {
                if fix == Fix::None {
                    continue;
                }
                let src = &g.node(node.inputs[i]).shape;
                let r = node.shape.rank();
                if fix == Fix::Rank {
                    let mut dims = vec![1usize; r - src.rank()];
                    dims.extend((0..src.rank()).map(|d| src.dim(d).unwrap_static()));
                    let new_shape: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
                    ins[i] = out.add_node(
                        Op::Reshape { new_shape },
                        vec![ins[i]],
                        Shape::new(&dims, src.dtype()),
                    );
                    continue;
                }
                let mut dims: Vec<usize> = (0..r - 2)
                    .map(|d| node.shape.dim(d).unwrap_static())
                    .collect();
                dims.push(src.dim(src.rank() - 2).unwrap_static());
                dims.push(src.dim(src.rank() - 1).unwrap_static());
                let target_shape: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
                ins[i] = out.add_node(
                    Op::Expand { target_shape },
                    vec![ins[i]],
                    Shape::new(&dims, src.dtype()),
                );
            }
            let nid = out.add_node(node.op.clone(), ins, node.shape.clone());
            id_map.insert(node.id, nid);
        }
        out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
        out
    }
}
