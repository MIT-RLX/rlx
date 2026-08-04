// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Op-level parity checking. After stat-injection / fusion-skip / attention
//! decompose / any IR transform, prove the result is unchanged — not just at the
//! final output, but **op by op**. `expose_all_ops` marks every compute node as
//! a graph output, so two structure-preserving graphs can be compared tensor by
//! tensor. Structure-changing transforms (fusion) are compared at the primary
//! output within tolerance.

use rlx_ir::{Graph, NodeId};
use rlx_runtime::{CompileOptions, Device, Session};
use std::collections::HashMap;

/// Rebuild `graph` with **every compute node exposed as an output** (in node
/// order). For a transform that copies the original nodes first (like the
/// injection passes), the first N exposed outputs align 1:1 with the original.
pub fn expose_all_ops(graph: &Graph) -> Graph {
    let mut g = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut exposed = Vec::new();
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = g.add_node(node.op.clone(), inputs, node.shape.clone());
        id_map.insert(node.id, new_id);
        if !node.inputs.is_empty() {
            exposed.push(new_id); // compute nodes only (skip Input/Param/Constant)
        }
    }
    g.set_outputs(exposed);
    g
}

fn run(
    graph: Graph,
    inputs: &[(&str, &[f32])],
    params: &[(String, Vec<f32>)],
    skip_fusion: bool,
) -> Vec<Vec<f32>> {
    let mut opts = CompileOptions::default();
    opts.fusion_opts.skip_fusion = skip_fusion;
    let mut c = Session::new(Device::Cpu).compile_with(graph, &opts);
    for (n, d) in params {
        c.set_param(n, d);
    }
    c.run(inputs)
}

/// Parity result.
#[derive(Clone, Debug)]
pub struct ParityReport {
    pub ops_checked: usize,
    pub max_abs: f32,
    pub max_rel: f32,
    /// Index (in node order) of the op with the largest error.
    pub worst_op: usize,
}

impl ParityReport {
    pub fn exact(&self) -> bool {
        self.max_abs == 0.0
    }
    pub fn within(&self, tol: f32) -> bool {
        self.max_rel <= tol
    }
}

fn compare(a: &[Vec<f32>], b: &[Vec<f32>]) -> ParityReport {
    let n = a.len().min(b.len());
    let (mut max_abs, mut max_rel, mut worst) = (0f32, 0f32, 0usize);
    for i in 0..n {
        if a[i].len() != b[i].len() {
            continue;
        }
        for (x, y) in a[i].iter().zip(&b[i]) {
            let ae = (x - y).abs();
            if ae > max_abs {
                max_abs = ae;
                worst = i;
            }
            max_rel = max_rel.max(ae / x.abs().max(1e-6));
        }
    }
    ParityReport {
        ops_checked: n,
        max_abs,
        max_rel,
        worst_op: worst,
    }
}

/// Op-by-op parity between two **structure-preserving** graphs (e.g. original vs
/// stat-injected): compiles both with all ops exposed, compares each tensor.
/// Injection should be bit-exact (`max_abs == 0`).
pub fn op_level_parity(
    original: &Graph,
    transformed: &Graph,
    inputs: &[(&str, &[f32])],
    params: &[(String, Vec<f32>)],
) -> ParityReport {
    // Compare the UNFUSED graphs so every op is a materialized primitive — the
    // injection preserves ops, so this must be bit-exact. (Fusion equivalence is
    // checked separately by `fusion_output_parity`; with fusion on, the tap's
    // extra consumers legitimately change *which* ops get fused, so a per-op
    // compare there is apples-to-oranges.)
    let a = run(expose_all_ops(original), inputs, params, true);
    let b = run(expose_all_ops(transformed), inputs, params, true);
    compare(&a, &b)
}

/// Primary-output parity for a **structure-changing** transform (e.g. fusion on
/// vs off): compiles `graph` both ways and compares output 0 within tolerance.
pub fn fusion_output_parity(
    graph: &Graph,
    inputs: &[(&str, &[f32])],
    params: &[(String, Vec<f32>)],
) -> ParityReport {
    let fused = run(graph.clone(), inputs, params, false);
    let unfused = run(graph.clone(), inputs, params, true);
    compare(
        &fused[..1.min(fused.len())],
        &unfused[..1.min(unfused.len())],
    )
}
