// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower SPD-manifold spectral ops (`Op::ReEig`, `Op::LogEig`) to primitives.
//!
//! The native ReEig / LogEig kernels do an **f64** LAPACK eigendecomposition and
//! only exist on CPU; every GPU backend "supports" them only via a CPU
//! host-fallback (`rlx_cpu::spd` against the mapped arena — no GPU eigen kernel).
//!
//! This pass rewrites **f32** ReEig / LogEig nodes into the graph-primitive
//! cyclic-Jacobi eigensolver from [`rlx_ir::ops::spd_eig`] (`Op::Scan`-bodied),
//! so SPDNet / TensorCSPNet / GraphCSPNet / TSMNet run natively on **every
//! backend** with no per-model changes. The **dtype gate is deliberate and makes
//! the pass zero-regression**: f32 is exactly the case GPUs want *and* the case
//! the native f64 kernels can't run (an f32 CPU SPD op panics), while f64 nodes
//! are left untouched so CPU LAPACK and the GPU f64 host-fallback are unaffected.
//! Hence the pass can run unconditionally in the rewrite pipeline (it is a no-op
//! unless an f32 ReEig/LogEig is present) — no backend `supported_ops()` edits.
//!
//! Both ops produce a packed `[2n²+n]` buffer `Y ∥ λ ∥ U` (the native layout, so
//! backward can reuse the eigendecomposition); the replacement emits the same
//! packed layout via [`rlx_ir::ops::spd_eig::spectral_packed`], so the downstream
//! `Narrow(0,0,n²)` that the manifold builder inserts selects `Y` unchanged.

use crate::pass::Pass;
use rlx_ir::ops::spd_eig::{bimap, spd_batch_norm_transport, spd_jacobi_sweeps, spectral_packed};
use rlx_ir::{DType, Graph, NodeId, Op};
use std::collections::HashMap;

/// True for an f32 SPD-manifold op (ReEig / LogEig / BiMap / SpdBatchNorm) —
/// the ops this pass rewrites to primitives. Gated on the first input's dtype.
fn is_f32_spd(graph: &Graph, node: &rlx_ir::Node) -> bool {
    matches!(
        node.op,
        Op::ReEig { .. } | Op::LogEig { .. } | Op::BiMap | Op::SpdBatchNorm { .. }
    ) && graph.shape(node.inputs[0]).dtype() == DType::F32
}

/// Rewrite every **f32** SPD-manifold op (`Op::ReEig` / `Op::LogEig` / `Op::BiMap`
/// / `Op::SpdBatchNorm`) into its graph-primitive subgraph (Jacobi eigensolver
/// for the spectral ops, matmuls for BiMap). Idempotent (no-op when no f32 SPD op
/// is present). Covers the full SPDNet-family forward op set so SPDNet /
/// TensorCSPNet / GraphCSPNet / TSMNet run natively on GPU.
pub struct LowerSpectral;

impl Pass for LowerSpectral {
    fn name(&self) -> &str {
        "lower_spectral"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph.nodes().iter().any(|n| is_f32_spd(&graph, n)) {
            return graph;
        }

        let sweeps = spd_jacobi_sweeps();
        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if is_f32_spd(&graph, node) {
                let inp: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                match &node.op {
                    Op::ReEig { eps } | Op::LogEig { eps } => {
                        let log = matches!(node.op, Op::LogEig { .. });
                        // Native op output is packed [2n²+n]; recover n from the input.
                        let n = new_graph.shape(inp[0]).dim(0).unwrap_static();
                        spectral_packed(&mut new_graph, inp[0], n, sweeps, *eps as f64, log)
                    }
                    Op::BiMap => bimap(&mut new_graph, inp[0], inp[1]),
                    Op::SpdBatchNorm { eps } => {
                        // inputs [x [batch,n,n], mean [n,n], g [n,n]]
                        let xs = new_graph.shape(inp[0]);
                        let (batch, n) = (xs.dim(0).unwrap_static(), xs.dim(1).unwrap_static());
                        spd_batch_norm_transport(
                            &mut new_graph,
                            inp[0],
                            inp[1],
                            inp[2],
                            n,
                            batch,
                            sweeps,
                            *eps as f64,
                        )
                    }
                    _ => unreachable!("is_f32_spd guarantees an SPD op"),
                }
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
