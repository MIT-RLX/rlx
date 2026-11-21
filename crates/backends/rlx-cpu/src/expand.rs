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
    let graph = fuse_strided_softmax(graph);
    let graph = fold_conv3d_bias_add(graph);
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

/// `perm` swaps `axis` with the last axis and moves nothing else.
fn swap_with_last(perm: &[usize]) -> Option<usize> {
    let last = perm.len().checked_sub(1)?;
    let axis = perm
        .iter()
        .enumerate()
        .find(|&(i, &p)| i != p)
        .map(|(i, _)| i)?;
    let ok = axis < last
        && perm[axis] == last
        && perm[last] == axis
        && perm
            .iter()
            .enumerate()
            .all(|(i, &p)| i == axis || i == last || i == p);
    ok.then_some(axis)
}

/// Undo `LowerSoftmaxAxis` where the CPU kernel does not need it.
///
/// That pass turns a softmax over a non-last axis into `Transpose(axis <-> last)
/// -> Softmax(last) -> Transpose back`, because most backends only reduce
/// contiguous runs. The CPU thunk normalises along any axis by stride
/// (`softmax_strided`), so the two transposes are pure cost: on a segmentation
/// network's `[1, C, D, H, W]` channel softmax they are two full copies of the
/// output, which measured larger than every convolution layer combined but one.
///
/// Rewritten back to `Softmax { axis }` only for the exact shape that pass emits
/// (single-use transposes, the same swap both sides, f32), and before memory
/// planning, so buffer lifetimes are computed for the graph that runs.
pub fn fuse_strided_softmax(g: Graph) -> Graph {
    let mut uses: HashMap<NodeId, usize> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            *uses.entry(i).or_insert(0) += 1;
        }
    }
    let outputs: std::collections::HashSet<NodeId> = g.outputs.iter().copied().collect();
    let single = |id: NodeId| uses.get(&id) == Some(&1) && !outputs.contains(&id);
    let mut replace: HashMap<NodeId, (NodeId, usize)> = HashMap::new();
    let mut skip: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for t2 in g.nodes() {
        let Op::Transpose { perm } = &t2.op else {
            continue;
        };
        let Some(axis) = swap_with_last(perm) else {
            continue;
        };
        let Some(&s_id) = t2.inputs.first() else {
            continue;
        };
        let s = g.node(s_id);
        let rank = s.shape.rank() as i32;
        let Op::Softmax { axis: s_axis } = s.op else {
            continue;
        };
        if !(s_axis == rank - 1 || s_axis == -1) {
            continue;
        }
        let Some(&t1_id) = s.inputs.first() else {
            continue;
        };
        let t1 = g.node(t1_id);
        let Op::Transpose { perm: p1 } = &t1.op else {
            continue;
        };
        let Some(&x_id) = t1.inputs.first() else {
            continue;
        };
        let x = g.node(x_id);
        let f32_all = [&x.shape, &t1.shape, &s.shape, &t2.shape]
            .iter()
            .all(|sh| sh.dtype() == rlx_ir::DType::F32);
        if p1 != perm
            || !single(t1_id)
            || !single(s_id)
            || !f32_all
            || x.shape.dims() != t2.shape.dims()
        {
            continue;
        }
        replace.insert(t2.id, (x_id, axis));
        skip.insert(t1_id);
        skip.insert(s_id);
    }
    if replace.is_empty() {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        if skip.contains(&node.id) {
            continue;
        }
        let new_id = match replace.get(&node.id) {
            Some(&(x, axis)) => out.add_node(
                Op::Softmax { axis: axis as i32 },
                vec![id_map[&x]],
                node.shape.clone(),
            ),
            None => {
                let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                out.add_node(node.op.clone(), ins, node.shape.clone())
            }
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// The rank-1 `[c]` f32 tensor behind a chain of `Reshape` / `Expand` (at least one): a
/// per-channel bias as a graph builder spells it before broadcasting it over NCDHW.
fn per_channel_source(g: &Graph, mut id: NodeId, c: usize) -> Option<NodeId> {
    let mut peeled = 0;
    while peeled < 4 {
        match &g.node(id).op {
            Op::Reshape { .. } | Op::Expand { .. } => {
                id = g.node(id).inputs[0];
                peeled += 1;
            }
            _ => break,
        }
    }
    let sh = &g.node(id).shape;
    (peeled > 0
        && sh.rank() == 1
        && sh.dims()[0] == rlx_ir::Dim::Static(c)
        && sh.dtype() == rlx_ir::DType::F32)
        .then_some(id)
}

/// `Conv3d → Add(bias broadcast from [C_out])` as the **3-input** `Op::Conv3d` whose thunk adds
/// the bias in its store, as [`fold_conv3d_bias`] does for a fused op.
///
/// Every 3-D network FreeSurfer runs (SynthSeg, SynthStrip, SynthMorph, the sclimbic and
/// hypothalamic U-Nets) builds its convolutions this way; `FuseConvBiasAct` only matches 2-D
/// convolutions, so each layer paid an extra full pass over its output for the add, and the
/// broadcast operand besides. Matched only for a single-use, 2-input, f32 convolution whose
/// add has the convolution's shape, before memory planning.
pub fn fold_conv3d_bias_add(g: Graph) -> Graph {
    let mut uses: HashMap<NodeId, usize> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            *uses.entry(i).or_insert(0) += 1;
        }
    }
    let outputs: std::collections::HashSet<NodeId> = g.outputs.iter().copied().collect();
    // add id -> (conv id, bias source)
    let mut fold: HashMap<NodeId, (NodeId, NodeId)> = HashMap::new();
    let mut folded_convs: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for add in g.nodes() {
        if !matches!(add.op, Op::Binary(rlx_ir::op::BinaryOp::Add)) || add.inputs.len() != 2 {
            continue;
        }
        for (ci, bi) in [(0, 1), (1, 0)] {
            let conv = g.node(add.inputs[ci]);
            if !matches!(conv.op, Op::Conv3d { .. })
                || conv.inputs.len() != 2
                || uses.get(&conv.id) != Some(&1)
                || outputs.contains(&conv.id)
                || conv.shape.dims() != add.shape.dims()
                || conv.shape.dtype() != rlx_ir::DType::F32
                || conv.shape.rank() != 5
            {
                continue;
            }
            let rlx_ir::Dim::Static(c_out) = conv.shape.dims()[1] else {
                continue;
            };
            if let Some(bias) = per_channel_source(&g, add.inputs[bi], c_out) {
                fold.insert(add.id, (conv.id, bias));
                folded_convs.insert(conv.id);
                break;
            }
        }
    }
    if fold.is_empty() {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        if folded_convs.contains(&node.id) {
            continue;
        }
        let new_id = match fold.get(&node.id) {
            Some(&(conv_id, bias)) => {
                let conv = g.node(conv_id);
                let ins = vec![
                    id_map[&conv.inputs[0]],
                    id_map[&conv.inputs[1]],
                    id_map[&bias],
                ];
                out.add_node(conv.op.clone(), ins, node.shape.clone())
            }
            None => {
                let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                out.add_node(node.op.clone(), ins, node.shape.clone())
            }
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

#[cfg(test)]
mod conv3d_bias_tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    #[test]
    fn a_broadcast_bias_add_folds_into_the_convolution() {
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[1, 2, 4, 4, 4], DType::F32));
        let w = g.param("w", Shape::new(&[3, 2, 3, 3, 3], DType::F32));
        let b = g.param("b", Shape::new(&[3], DType::F32));
        let y = g.conv3d(x, w, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1);
        let br = g.add_node(
            Op::Reshape {
                new_shape: vec![1, 3, 1, 1, 1],
            },
            vec![b],
            Shape::new(&[1, 3, 1, 1, 1], DType::F32),
        );
        let z = g.binary(
            rlx_ir::op::BinaryOp::Add,
            y,
            br,
            Shape::new(&[1, 3, 4, 4, 4], DType::F32),
        );
        g.set_outputs(vec![z]);
        let f = fold_conv3d_bias_add(g);
        let convs: Vec<_> = f
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Conv3d { .. }))
            .collect();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].inputs.len(), 3);
        assert!(!f.nodes().iter().any(|n| matches!(n.op, Op::Binary(_))));
        assert_eq!(f.outputs, vec![convs[0].id]);
    }

    #[test]
    fn a_convolution_read_twice_is_left_alone() {
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[1, 2, 4, 4, 4], DType::F32));
        let w = g.param("w", Shape::new(&[3, 2, 3, 3, 3], DType::F32));
        let b = g.param("b", Shape::new(&[3], DType::F32));
        let y = g.conv3d(x, w, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1);
        let br = g.add_node(
            Op::Reshape {
                new_shape: vec![1, 3, 1, 1, 1],
            },
            vec![b],
            Shape::new(&[1, 3, 1, 1, 1], DType::F32),
        );
        let z = g.binary(
            rlx_ir::op::BinaryOp::Add,
            y,
            br,
            Shape::new(&[1, 3, 4, 4, 4], DType::F32),
        );
        g.set_outputs(vec![z, y]);
        let f = fold_conv3d_bias_add(g);
        assert!(f.nodes().iter().any(|n| matches!(n.op, Op::Binary(_))));
    }
}

#[cfg(test)]
mod strided_softmax_tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    #[test]
    fn the_transpose_sandwich_folds_back_into_one_softmax() {
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[1, 3, 4, 5], DType::F32));
        let s = g.softmax(x, 1, Shape::new(&[1, 3, 4, 5], DType::F32));
        g.set_outputs(vec![s]);
        let lowered = rlx_opt::rlx_fusion::LowerSoftmaxAxis.run(g);
        assert!(
            lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Transpose { .. }))
        );
        let fused = fuse_strided_softmax(lowered);
        assert!(
            !fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Transpose { .. }))
        );
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Softmax { axis: 1 }))
        );
    }

    #[test]
    fn only_a_single_axis_swap_is_recognised() {
        assert_eq!(swap_with_last(&[0, 4, 2, 3, 1]), Some(1));
        assert_eq!(swap_with_last(&[0, 1, 2, 3]), None);
        assert_eq!(swap_with_last(&[1, 2, 0]), None);
    }
}
