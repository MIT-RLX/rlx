// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Quantize/dequantize insertion pass.
//!
//! The IR-rewrite half of post-training quantization. Given a
//! per-tensor or per-channel calibration record (produced by running
//! forward on a sample batch, see `rlx_cpu::calibrate`), this pass
//! walks the graph and inserts `Op::Quantize → Op::Dequantize` pairs
//! immediately downstream of each tagged node. Consumers of the
//! original tap node are rewired to read the dequantized result, so
//! everything past the tap sees an INT8 round-tripped activation /
//! weight while the rest of the graph stays in fp32.
//!
//! Why a Q/DQ pair instead of switching the whole subgraph to INT8?
//! For PTQ this is the standard "fake-quant" pattern — the IR stays
//! coherent in fp32, but each tap loses one quant step of precision
//! to simulate the on-device int8 path. Real INT8-arithmetic kernels
//! (`Op::DequantMatMul`, etc.) replace specific Q/DQ-bracketed regions
//! later in the pipeline; this pass just produces the canonical form.
//!
//! Scope intentionally narrow: insert-only, no measurement. The
//! caller is responsible for filling `CalibrationRecord` from
//! whatever execution path it has access to.

use rlx_ir::{Graph, Node, NodeId, Op, Shape};
use std::collections::HashMap;

/// One calibrated quant entry per tap. `axis = None` is per-tensor;
/// `axis = Some(d)` is per-channel along axis `d`, in which case
/// `scales` and `zero_points` must each have length `tap.shape.dim(d)`.
#[derive(Debug, Clone)]
pub struct CalibrationEntry {
    pub axis: Option<usize>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<i32>,
}

impl CalibrationEntry {
    /// Convenience constructor for the per-tensor symmetric case.
    pub fn per_tensor(scale: f32) -> Self {
        Self {
            axis: None,
            scales: vec![scale],
            zero_points: vec![0],
        }
    }

    /// Per-channel symmetric (`zp = 0`) along `axis`.
    pub fn per_channel(axis: usize, scales: Vec<f32>) -> Self {
        let n = scales.len();
        Self {
            axis: Some(axis),
            scales,
            zero_points: vec![0; n],
        }
    }
}

/// Map of tap NodeId → calibrated quant params.
pub type CalibrationRecord = HashMap<NodeId, CalibrationEntry>;

/// Insert `Quantize → Dequantize` pairs at every tap in `record`.
/// Returns a graph where each tagged node is followed by a
/// `Quantize → Dequantize` pair, and every consumer of the original
/// tap reads from the dequantized output instead.
///
/// One-pass build: when we copy a consumer node, we rewrite any input
/// edge that refers to a tap so it points at the tap's DQ instead.
/// The Q and DQ nodes themselves are exempt (we identify them via
/// their `Op::Quantize` / `Op::Dequantize` discriminants — the tap's
/// raw value still flows in to the Quantize).
pub fn insert_q_dq(graph: Graph, record: &CalibrationRecord) -> Graph {
    let mut out = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    // For each old-graph tap NodeId, the NodeId of its dequantized
    // replacement in `out`. Consumers of the tap rewrite their inputs
    // to read from this id instead of the raw tap.
    let mut tap_dq: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        // Translate `node.inputs` for the *new* graph, rerouting any
        // tap reference to the tap's DQ.
        let new_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|inp| {
                // The Q node we'll insert next iteration is the only
                // legal raw-tap consumer; everything else routes through
                // DQ. Since we haven't placed the Q yet (it's inserted
                // *after* the tap node it wraps), the only nodes we
                // consider "Q" here are nodes we ourselves emit below.
                // No risk of self-reference: we route via tap_dq only
                // when it's already populated — i.e. for nodes that
                // come after their producer was tapped.
                tap_dq.get(inp).copied().unwrap_or(id_map[inp])
            })
            .collect();

        let new_id = out.add_node(node.op.clone(), new_inputs, node.shape.clone());
        id_map.insert(node.id, new_id);

        if let Some(entry) = record.get(&node.id) {
            let q = insert_quantize(new_id, node, entry, &mut out);
            let dq = insert_dequantize(q, node, entry, &mut out);
            tap_dq.insert(node.id, dq);
        }
    }

    // Outputs: if a tap is also a graph output, return the DQ.
    let new_outputs: Vec<NodeId> = graph
        .outputs
        .iter()
        .map(|&id| tap_dq.get(&id).copied().unwrap_or(id_map[&id]))
        .collect();
    out.set_outputs(new_outputs);
    out
}

fn insert_quantize(
    src: NodeId,
    src_node: &Node,
    entry: &CalibrationEntry,
    out: &mut Graph,
) -> NodeId {
    let q_shape: Shape = src_node.shape.clone().with_dtype(rlx_ir::DType::I8);
    out.add_node(
        Op::Quantize {
            axis: entry.axis,
            scales: entry.scales.clone(),
            zero_points: entry.zero_points.clone(),
        },
        vec![src],
        q_shape,
    )
}

fn insert_dequantize(
    q: NodeId,
    src_node: &Node,
    entry: &CalibrationEntry,
    out: &mut Graph,
) -> NodeId {
    let dq_shape: Shape = src_node.shape.clone().with_dtype(rlx_ir::DType::F32);
    out.add_node(
        Op::Dequantize {
            axis: entry.axis,
            scales: entry.scales.clone(),
            zero_points: entry.zero_points.clone(),
        },
        vec![q],
        dq_shape,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::*;
    use rlx_ir::*;

    #[test]
    fn inserts_q_dq_pair_after_tap() {
        let f = DType::F32;
        let mut g = Graph::new("ptq_demo");
        let x = g.input("x", Shape::new(&[4, 8], f));
        let y = g.activation(Activation::Relu, x, Shape::new(&[4, 8], f));
        let z = g.binary(BinaryOp::Add, y, y, Shape::new(&[4, 8], f));
        g.set_outputs(vec![z]);

        // Tag `y` for per-tensor quantization.
        let mut record = CalibrationRecord::new();
        record.insert(y, CalibrationEntry::per_tensor(0.05));

        let g2 = insert_q_dq(g, &record);

        // Expect: a Quantize and a Dequantize node now exist.
        assert!(
            g2.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Quantize { .. }))
        );
        assert!(
            g2.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Dequantize { .. }))
        );

        // The Add node's inputs should now reference the Dequantize
        // output, not the Relu output. Find the Add and check.
        let add = g2
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Binary(BinaryOp::Add)))
            .expect("add node");
        for &in_id in &add.inputs {
            let in_op = &g2.node(in_id).op;
            assert!(
                matches!(in_op, Op::Dequantize { .. }),
                "Add input should be Dequantize, got {in_op:?}"
            );
        }
    }

    #[test]
    fn untagged_nodes_pass_through_unchanged() {
        let f = DType::F32;
        let mut g = Graph::new("no_taps");
        let x = g.input("x", Shape::new(&[4], f));
        let y = g.activation(Activation::Relu, x, Shape::new(&[4], f));
        g.set_outputs(vec![y]);

        let n_before = g.len();
        let g2 = insert_q_dq(g, &CalibrationRecord::new());
        assert_eq!(g2.len(), n_before);
    }
}
