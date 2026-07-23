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

//! Claim-then-expand for fused / region ops the CPU thunk path cannot run
//! natively.
//!
//! CPU claims `FusedConvBiasAct`, `PartitionedConv`, `FusedTransformerLayer`,
//! `TransformRegion`, and `BatchElementwiseRegion` for legalize / coverage
//! (and so fusion may emit them). The catch-all compile arm is `Thunk::Nop`,
//! so those nodes must be rewritten to primitives the existing thunks already
//! execute — same pattern as Metal's `lower_cpu_nop_fused_for_metal` and
//! OneAPI's `expand_cpu_nop_fused` + `DecomposeFusionRegions`.
//!
//! `Op::If` / `Op::While` are claimed too; `LowerControlFlow` (fusion pipeline
//! + `CpuBackend::compile`) expands them before thunks.

use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::pass::Pass as _;
use std::collections::HashMap;

/// Expand claimed fused/region forms that would otherwise become `Thunk::Nop`.
///
/// Call after fusion / LIR→graph and before `compile_thunks` / memory plan.
pub fn prepare_graph_for_thunks(graph: Graph) -> Graph {
    let graph = expand_cpu_nop_fused(graph);
    // TransformRegion / BatchElementwiseRegion → ResizeNearest2x + ElementwiseRegion
    // (+ Concat). Native FK keep is off for CPU by default; this still covers
    // `RLX_NATIVE_FK_REGIONS=1` and hand-built region IR.
    rlx_opt::rlx_fusion::DecomposeFusionRegions.run(graph)
}

/// Expand fused ops whose CPU compile arm is `Thunk::Nop`.
pub fn expand_cpu_nop_fused(g: Graph) -> Graph {
    let needs = g.nodes().iter().any(|n| {
        matches!(
            n.op,
            Op::FusedConvBiasAct { .. }
                | Op::PartitionedConv { .. }
                | Op::FusedTransformerLayer { .. }
        )
    });
    if !needs {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();
    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::FusedConvBiasAct { .. }
            | Op::PartitionedConv { .. }
            | Op::FusedTransformerLayer { .. } => {
                inline_unfused(&mut out, &node.op, &new_inputs, &node.shape)
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

fn inline_unfused(out: &mut Graph, op: &Op, inputs: &[NodeId], shape: &rlx_ir::Shape) -> NodeId {
    let mut mini = Graph::new("cpu_unfuse");
    let mut mini_ins = Vec::with_capacity(inputs.len());
    for (i, &src) in inputs.iter().enumerate() {
        let sh = out.node(src).shape.clone();
        mini_ins.push(mini.append_node(
            Op::Input {
                name: format!("in{i}"),
            },
            vec![],
            sh,
            None,
        ));
    }
    let out_id = mini.append_node(op.clone(), mini_ins, shape.clone(), None);
    mini.set_outputs(vec![out_id]);
    let expanded = rlx_opt::unfuse_fused_for_autodiff(mini);
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for n in expanded.nodes() {
        if let Op::Input { name } = &n.op {
            if let Some(rest) = name.strip_prefix("in") {
                if let Ok(i) = rest.parse::<usize>() {
                    map.insert(n.id, inputs[i]);
                    continue;
                }
            }
        }
        let mapped: Vec<NodeId> = n.inputs.iter().map(|id| map[id]).collect();
        let nid = out.add_node(n.op.clone(), mapped, n.shape.clone());
        map.insert(n.id, nid);
    }
    map[&expanded.outputs[0]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::{Activation, ChainOperand, ChainStep, RegionPrologue, TransformStep};
    use rlx_ir::{DType, Shape};

    fn const_f32(g: &mut Graph, xs: &[f32], dims: &[usize]) -> NodeId {
        let mut bytes = Vec::with_capacity(xs.len() * 4);
        for x in xs {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        g.add_node(
            Op::Constant { data: bytes },
            vec![],
            Shape::new(dims, DType::F32),
        )
    }

    #[test]
    fn expand_removes_partitioned_conv() {
        let mut g = Graph::new("pc");
        let x = const_f32(&mut g, &[1.0, 2.0, 3.0, 4.0], &[4]);
        let ir = const_f32(&mut g, &[0.5, 0.25], &[2]);
        let y = g.partitioned_conv(x, ir, 4);
        g.set_outputs(vec![y]);
        assert!(
            g.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::PartitionedConv { .. }))
        );
        let out = prepare_graph_for_thunks(g);
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::PartitionedConv { .. })),
            "PartitionedConv must expand before thunks"
        );
    }

    #[test]
    fn expand_removes_transform_and_batch_regions() {
        let mut g = Graph::new("fk");
        let x = const_f32(&mut g, &[1.0; 4], &[1, 1, 2, 2]);
        let tr = g.add_node(
            Op::TransformRegion {
                steps: vec![TransformStep::ResizeNearest2x(ChainOperand::Input(0))],
                num_inputs: 1,
            },
            vec![x],
            Shape::new(&[1, 1, 4, 4], DType::F32),
        );
        let a = const_f32(&mut g, &[1.0; 4], &[1, 1, 2, 2]);
        let b = const_f32(&mut g, &[2.0; 4], &[1, 1, 2, 2]);
        let batch = g.add_node(
            Op::BatchElementwiseRegion {
                chain: vec![ChainStep::Activation(
                    Activation::Relu,
                    ChainOperand::Input(0),
                )],
                num_batch_inputs: 2,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            vec![a, b],
            Shape::new(&[2, 1, 2, 2], DType::F32),
        );
        g.set_outputs(vec![tr, batch]);
        let out = prepare_graph_for_thunks(g);
        assert!(
            !out.nodes().iter().any(|n| matches!(
                n.op,
                Op::TransformRegion { .. } | Op::BatchElementwiseRegion { .. }
            )),
            "FK regions must decompose before thunks"
        );
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ResizeNearest2x)),
            "TransformRegion should become ResizeNearest2x"
        );
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ElementwiseRegion { .. })),
            "BatchElementwiseRegion should become ElementwiseRegion slices"
        );
    }

    #[test]
    fn expand_removes_fused_conv_bias_act() {
        let mut g = Graph::new("fcba");
        // Minimal shapes: NCHW 1x1x2x2, 1x1x1x1 weight, bias [1].
        let x = const_f32(&mut g, &[1.0; 4], &[1, 1, 2, 2]);
        let w = const_f32(&mut g, &[1.0], &[1, 1, 1, 1]);
        let b = const_f32(&mut g, &[0.5], &[1]);
        let y = g.add_node(
            Op::FusedConvBiasAct {
                kernel_size: vec![1, 1],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: 1,
                activation: Some(Activation::Relu),
                has_residual: false,
            },
            vec![x, w, b],
            Shape::new(&[1, 1, 2, 2], DType::F32),
        );
        g.set_outputs(vec![y]);
        let out = prepare_graph_for_thunks(g);
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedConvBiasAct { .. })),
            "FusedConvBiasAct must expand before thunks"
        );
    }
}
