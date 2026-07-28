// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! IR passes shared by [`crate::TpuExecutable::compile`] and orchestrated HLO segments.

use rlx_ir::Graph;
use rlx_ir::OpKind;
use rlx_opt::pass::Pass as _;

/// Training-bwd kinds TPU already lowers (or keeps host) — leave intact when
/// expanding [`OpKind::AttentionBackward`] via autodiff decompose.
const PRESERVE_BACKWARD_EXCEPT_ATTENTION: &[OpKind] = &[
    OpKind::ReluBackward,
    OpKind::ActivationBackward,
    OpKind::LayerNormBackwardInput,
    OpKind::LayerNormBackwardGamma,
    OpKind::RmsNormBackwardInput,
    OpKind::RmsNormBackwardGamma,
    OpKind::RmsNormBackwardBeta,
    OpKind::GroupNormBackwardInput,
    OpKind::GroupNormBackwardGamma,
    OpKind::GroupNormBackwardBeta,
    OpKind::RopeBackward,
    OpKind::Conv2dBackwardInput,
    OpKind::Conv2dBackwardWeight,
    OpKind::MaxPool2dBackward,
    OpKind::CumsumBackward,
    OpKind::GatherBackward,
    OpKind::SoftmaxCrossEntropyBackward,
    OpKind::FakeQuantizeBackward,
    OpKind::ScanBackward,
    OpKind::ScanBackwardXs,
    OpKind::AdaLayerNormBackward,
    OpKind::GatedResidualBackward,
];

/// Expand `AttentionBackward` to MatMul/Softmax/… via autodiff decompose.
fn expand_attention_backward(graph: Graph) -> Graph {
    let needs = graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op, rlx_ir::Op::AttentionBackward { .. }));
    if !needs {
        return graph;
    }
    rlx_opt::rlx_autodiff::decompose_backward_ops_except(graph, PRESERVE_BACKWARD_EXCEPT_ATTENTION)
}

/// Run the TPU pre-HLO pipeline (DCE, tier-2 fusions, elementwise regions, FKL,
/// compose-to-primitives for claimed HLO-friendly ops, unfuse).
pub fn prepare_graph_for_hlo(graph: Graph) -> Graph {
    let graph = rlx_opt::DeadCodeElimination.run(graph);
    let graph = rlx_opt::ConstantFolding.run(graph);
    let graph = rlx_opt::FuseResidualLN.run(graph);
    let graph = rlx_opt::FuseResidualRmsNorm.run(graph);
    let graph = rlx_opt::FuseRmsNormReshape.run(graph);
    let graph = rlx_opt::FuseMatMulBiasAct.run(graph);
    let graph = rlx_opt::LegalizeBroadcast.run(graph);
    let graph = rlx_opt::MarkElementwiseRegions.run(graph);
    let graph = crate::fk_pipeline::apply_fk_passes(graph);
    // Claimed ops that compose from primitives already in the HLO walker.
    let graph = rlx_fusion::LowerFma.run(graph);
    let graph = rlx_fusion::LowerGroupNorm.run(graph);
    let graph = rlx_fusion::LowerBatchNormInference.run(graph);
    let graph = rlx_fusion::LowerSoftmaxCrossEntropy.run(graph);
    let graph = rlx_fusion::LowerBackwardOps.run(graph);
    // Expand AttentionBackward → MatMul/Softmax/… primitives already in
    // the HLO walker. Preserve every other training-bwd kind that TPU
    // lowers natively (or keeps host-segmented).
    let graph = expand_attention_backward(graph);
    // f32 BiMap / ReEig / LogEig / SpdBatchNorm → Jacobi Scan + matmuls.
    let graph = rlx_fusion::LowerSpectral.run(graph);
    // Unroll short / budgeted Scans (incl. SPD Jacobi) into HLO primitives;
    // residual Scan stays host-segmented.
    let graph = rlx_fusion::maybe_unroll_scans_budget(graph, 4096);
    let graph = rlx_fusion::unfuse_recurrent_ops(graph);
    // FusedConvBiasAct / PartitionedConv / GatedDeltaNet → primitives
    // already in the HLO walker (keep FusedMatMulBiasAct / ResidualLN native).
    let graph = expand_compose_hosts(graph);
    crate::unfuse::unfuse(graph)
}

/// Expand claimed compose forms that have no dedicated HLO arm but
/// `rlx_fusion::unfuse_fused_for_autodiff` already decomposes.
fn expand_compose_hosts(g: Graph) -> Graph {
    use rlx_ir::{NodeId, Op};
    use std::collections::HashMap;

    let needs = g.nodes().iter().any(|n| {
        matches!(
            n.op,
            Op::FusedConvBiasAct { .. } | Op::PartitionedConv { .. } | Op::GatedDeltaNet { .. }
        )
    });
    if !needs {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::FusedConvBiasAct { .. } | Op::PartitionedConv { .. } | Op::GatedDeltaNet { .. } => {
                inline_unfused_compose(&mut out, &node.op, &new_inputs, &node.shape)
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

fn inline_unfused_compose(
    out: &mut Graph,
    op: &rlx_ir::Op,
    inputs: &[rlx_ir::NodeId],
    shape: &rlx_ir::Shape,
) -> rlx_ir::NodeId {
    use rlx_ir::{NodeId, Op};
    use std::collections::HashMap;

    let mut mini = Graph::new("tpu_unfuse");
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
    let expanded = rlx_fusion::unfuse_fused_for_autodiff(mini);
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
