// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

use rlx_ir::{Graph, NodeId, Op, OpKind};
use rlx_opt::pass::Pass as _;
use std::collections::HashMap;

/// OpKinds rlx-cpu **claims but has no thunk arm for**. The catch-all compile
/// arm is `Thunk::Nop`, so one of these reaching `compile_thunks` produces a
/// zeroed output rather than an error.
///
/// This list is not documentation — [`expand_cpu_nop_fused`] dispatches on it,
/// so it cannot drift from what actually gets expanded. It is also the input to
/// `rlx-runtime/tests/host_fallback_never_nops.rs`, which is the point: **every
/// backend routing an op here to the CPU host fallback gets silent zeros**, and
/// that composition is invisible to a per-backend test. Vulkan's
/// `PartitionedConv` shipped exactly that way — claimed, never lowered, routed
/// to a host path that Nops it, reproducing identically on three drivers
/// because the op never ran at all.
pub const NO_THUNK_ARM: &[OpKind] = &[
    // Rewritten to primitives by `expand_cpu_nop_fused`, below.
    OpKind::FusedConvBiasAct,
    OpKind::PartitionedConv,
    OpKind::FusedTransformerLayer,
    // Rewritten by `DecomposeFusionRegions` in `prepare_graph_for_thunks`.
    OpKind::TransformRegion,
    OpKind::BatchElementwiseRegion,
];

/// The subset of [`NO_THUNK_ARM`] that [`expand_cpu_nop_fused`] itself inlines.
/// The rest are handled by `DecomposeFusionRegions`.
const INLINED_HERE: &[OpKind] = &[
    OpKind::FusedConvBiasAct,
    OpKind::PartitionedConv,
    OpKind::FusedTransformerLayer,
];

/// Expand claimed fused/region forms that would otherwise become `Thunk::Nop`.
///
/// Call after fusion / LIR→graph and before `compile_thunks` / memory plan.
pub fn prepare_graph_for_thunks(graph: Graph) -> Graph {
    let graph = expand_cpu_nop_fused(graph);
    // I32 / U32 stay integers; f32-computing ops get casts around them.
    let graph = crate::int_storage::legalize_int_storage(graph);
    // TransformRegion / BatchElementwiseRegion → ResizeNearest2x + ElementwiseRegion
    // (+ Concat). Native FK keep is off for CPU by default; this still covers
    // `RLX_NATIVE_FK_REGIONS=1` and hand-built region IR.
    rlx_opt::rlx_fusion::DecomposeFusionRegions.run(graph)
}

/// Expand fused ops whose CPU compile arm is `Thunk::Nop`.
pub fn expand_cpu_nop_fused(g: Graph) -> Graph {
    let needs = g
        .nodes()
        .iter()
        .any(|n| INLINED_HERE.contains(&n.op.kind()));
    if !needs {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();
    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id =
            if let Some(id) = fold_conv3d_bias(&mut out, &node.op, &new_inputs, &node.shape) {
                id
            } else if INLINED_HERE.contains(&node.op.kind()) {
                inline_unfused(&mut out, &node.op, &new_inputs, &node.shape)
            } else {
                out.add_node(node.op.clone(), new_inputs, node.shape.clone())
            };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// A 3-D `FusedConvBiasAct` (no residual) as a **3-input** `Op::Conv3d`
/// whose thunk adds the bias in its store, then its activation, if any.
///
/// Expanded the generic way it is `conv → Reshape(bias) → Expand → Add`, and
/// the `Expand` of a parameter is precomputed and held for the whole run: one
/// output-sized tensor per convolution. A 10-layer MeshNet over a 164×164×224
/// crop planned a 2.4 GB arena that way, against ~0.25 GB of activations.
/// The wgpu backend folds the same way (`fold_conv_bias_into_conv3d`).
///
/// Backend-private, as there: `Op::Conv3d` is two inputs in the IR contract,
/// and this graph exists only between here and `compile_thunks`.
fn fold_conv3d_bias(
    out: &mut Graph,
    op: &Op,
    inputs: &[NodeId],
    shape: &rlx_ir::Shape,
) -> Option<NodeId> {
    let Op::FusedConvBiasAct {
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        activation,
        has_residual: false,
    } = op
    else {
        return None;
    };
    if kernel_size.len() != 3 || inputs.len() != 3 {
        return None;
    }
    let a3 = |v: &Vec<usize>| [v[0], v[1], v[2]];
    let conv = out.add_node(
        Op::Conv3d {
            stride: a3(stride),
            padding: a3(padding),
            dilation: a3(dilation),
            groups: *groups,
        },
        inputs.to_vec(),
        shape.clone(),
    );
    Some(match activation {
        Some(a) => out.add_node(Op::Activation(*a), vec![conv], shape.clone()),
        None => conv,
    })
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
    fn a_3d_conv_bias_folds_into_the_conv() {
        let mut g = Graph::new("fcba3");
        let x = const_f32(&mut g, &[1.0; 8], &[1, 1, 2, 2, 2]);
        let w = const_f32(&mut g, &[2.0], &[1, 1, 1, 1, 1]);
        let b = const_f32(&mut g, &[0.5], &[1]);
        let y = g.add_node(
            Op::FusedConvBiasAct {
                kernel_size: vec![1, 1, 1],
                stride: vec![1, 1, 1],
                padding: vec![0, 0, 0],
                dilation: vec![1, 1, 1],
                groups: 1,
                activation: Some(Activation::Relu),
                has_residual: false,
            },
            vec![x, w, b],
            Shape::new(&[1, 1, 2, 2, 2], DType::F32),
        );
        g.set_outputs(vec![y]);
        let out = prepare_graph_for_thunks(g);
        let conv = out
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Conv3d { .. }))
            .expect("a Conv3d");
        assert_eq!(conv.inputs.len(), 3, "the bias rides on the conv");
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Expand { .. })),
            "no bias-sized Expand"
        );
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Activation(Activation::Relu)))
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
