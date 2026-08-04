// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Auto-rewriter — decompose unsupported ops into primitives the backend claims.
//!
//! When [`legalize_for_backend`] fails, this module applies structural lowers
//! and fused-op unfuse passes until the graph legalizes or no progress is made.

use std::collections::{HashMap, HashSet};

use rlx_fusion::control_flow::{LowerControlFlow, LowerScan};
use rlx_fusion::fusion::UnfuseElementwiseRegions;
use rlx_fusion::lower_axial_rope2d::LowerAxialRope2d;
use rlx_fusion::lower_backward_ops::LowerBackwardOps;
use rlx_fusion::lower_cumulative::LowerCumulative;
use rlx_fusion::lower_dot_general::LowerDotGeneral;
use rlx_fusion::lower_fake_quantize::LowerFakeQuantize;
use rlx_fusion::lower_fma::LowerFma;
use rlx_fusion::lower_histogram::LowerHistogram;
use rlx_fusion::lower_logical_kernels;
use rlx_fusion::lower_loss_ops::LowerSoftmaxCrossEntropy;
use rlx_fusion::lower_pad::LowerPad;
use rlx_fusion::lower_reduce_axes::LowerNonLastAxisReduce;
use rlx_fusion::lower_scaled_grouped_matmul::LowerScaledGroupedMatMul;
use rlx_fusion::lower_slice::LowerSlice;
use rlx_fusion::lower_spectral::LowerSpectral;
use rlx_fusion::lower_spline_activation::LowerSplineActivation;
use rlx_fusion::lower_spline_backward::LowerSplineActivationBackward;
use rlx_fusion::lower_structural::LowerStructural;
use rlx_fusion::lower_synth_matmul::LowerSynthMatMul;
use rlx_fusion::lower_synth_matmul_backward::LowerSynthMatMulBackward;
use rlx_fusion::lower_synth_reconstruct::LowerSynthReconstruct;
use rlx_fusion::lower_vae_ops::{LowerBatchNormInference, LowerGroupNorm, LowerResizeNearest2x};
use rlx_fusion::pass::Pass;
use rlx_fusion::unfuse::unfuse_fused_for_autodiff;
use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
use rlx_ir::{Graph, LowerContext, Node, NodeId, Op, OpKind, lookup_op};

use crate::legalize::legalize_for_backend;

// Kinds that trigger the `unfuse` pass (which holds their decomposition arm).
// Not all are "fused" — `LoraMatMul`/`FakeQuantize`/`AxialRope2d` are plain ops
// whose decomposition lives in `unfuse` and would otherwise never fire on a
// backend that lacks a native kernel (the op would hard-fail at legalization).
const FUSED_KINDS: &[OpKind] = &[
    OpKind::FusedMatMulBiasAct,
    OpKind::FusedConvBiasAct,
    OpKind::FusedSwiGLU,
    OpKind::FusedResidualLN,
    OpKind::FusedResidualRmsNorm,
    OpKind::FusedAttentionBlock,
    OpKind::FusedTransformerLayer,
    OpKind::GatedDeltaNet,
    OpKind::Lstm,
    OpKind::Gru,
    OpKind::Rnn,
    OpKind::Mamba2,
    OpKind::SelectiveScan,
    OpKind::LoraMatMul,
    OpKind::PartitionedConv,
    OpKind::AdaLayerNorm,
    OpKind::GatedResidual,
];

fn unsupported_kinds(graph: &Graph, supported: &[OpKind]) -> HashSet<OpKind> {
    legalize_for_backend(graph, supported)
        .err()
        .map(|bad| bad.into_iter().map(|(_, k)| k).collect())
        .unwrap_or_default()
}

fn needs_unfuse(kinds: &HashSet<OpKind>) -> bool {
    kinds.iter().any(|k| FUSED_KINDS.contains(k))
}

#[cfg(feature = "training")]
fn needs_backward_decompose(bad: &HashSet<OpKind>) -> bool {
    use OpKind::*;
    // Every `*Backward` kind that `rlx_autodiff::decompose_backward_ops_except`
    // can lower to primitives (mirror `contains_training_backward_except`).
    // This only ever fires for a kind the target backend does NOT claim (it's in
    // `bad`), so backends with a native kernel — which list the kind in their
    // `supported_ops()` and so keep it out of `bad` — are unaffected; for the
    // rest it turns an unsupported-op error into a working decomposition.
    bad.iter().any(|k| {
        matches!(
            k,
            Conv2dBackwardInput
                | Conv2dBackwardWeight
                | MaxPool2dBackward
                | LayerNormBackwardInput
                | LayerNormBackwardGamma
                | RmsNormBackwardInput
                | RmsNormBackwardGamma
                | RmsNormBackwardBeta
                | GroupNormBackwardInput
                | GroupNormBackwardGamma
                | GroupNormBackwardBeta
                | RopeBackward
                | AttentionBackward
                | CumsumBackward
                | GatherBackward
                | BatchNormInferenceBackwardInput
                | BatchNormInferenceBackwardGamma
                | BatchNormInferenceBackwardBeta
                | FakeQuantizeBackward
                | SoftmaxCrossEntropyBackward
                | ReluBackward
                | ActivationBackward
                | ScanBackward
                | ScanBackwardXs
                | AdaLayerNormBackward
                | GatedResidualBackward
        )
    })
}

/// Copy a node into `out`, preserving its debug `name` and `origin`.
fn copy_node(out: &mut Graph, node: &Node, inputs: Vec<NodeId>) -> NodeId {
    let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
    let n = out.node_mut(id);
    n.name = node.name.clone();
    n.origin = node.origin.clone();
    id
}

/// Decompose registered custom ops that provide an
/// [`OpExtension::lower`](rlx_ir::OpExtension::lower) rule into primitive
/// subgraphs.
///
/// This is the middle extensibility tier for [`Op::Custom`]: an op with **no**
/// `lower` override stays opaque and is executed by its per-backend kernel (see
/// `rlx-cpu/src/op_registry.rs`); an op **with** a `lower` override becomes
/// primitives here, so it fuses and runs on **every** backend with no kernel and
/// no edit to the closed core `Op` enum. Registering `lower` is opt-in — an op
/// that ships a hand-tuned native kernel simply doesn't implement it.
///
/// Rebuilds the graph node-by-node (the standard `HashMap<NodeId, NodeId>`
/// remap); a no-op when the graph contains no custom ops. Runs early in the
/// backend rewrite so the decomposition is visible to fusion.
pub fn lower_custom_ops(graph: Graph) -> Graph {
    if !graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::Custom { .. }))
    {
        return graph;
    }
    let mut out = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::Custom { name, .. } => {
                // `lower` emits the decomposition directly into `out` against the
                // already-remapped inputs and returns its output node. `None`
                // (the trait default, or a decline) keeps the op opaque.
                let lowered = lookup_op(name).and_then(|ext| {
                    let mut ctx = LowerContext {
                        inputs: &new_inputs,
                        out: &mut out,
                    };
                    ext.lower(node, &mut ctx)
                });
                match lowered {
                    Some(id) => id,
                    None => copy_node(&mut out, node, new_inputs),
                }
            }
            _ => copy_node(&mut out, node, new_inputs),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(graph.outputs.iter().map(|o| id_map[o]).collect());
    out
}

/// Rewrite `graph` toward `supported` op kinds. Idempotent when already legal.
pub fn rewrite_for_backend(graph: Graph, supported: &[OpKind]) -> Graph {
    rewrite_for_backend_with_config(graph, supported, KernelDispatchConfig::default())
}

/// Like [`rewrite_for_backend`] but applies logical-kernel common lowers first.
pub fn rewrite_for_backend_with_dispatch(
    graph: Graph,
    supported: &[OpKind],
    dispatch: KernelDispatchPolicy,
) -> Graph {
    rewrite_for_backend_with_config(graph, supported, KernelDispatchConfig::new(dispatch))
}

/// Full dispatch control (policy + per-`OpKind` overrides).
pub fn rewrite_for_backend_with_config(
    mut graph: Graph,
    supported: &[OpKind],
    config: KernelDispatchConfig,
) -> Graph {
    graph = lower_logical_kernels(graph, supported, config);

    // Lower f32 SPD spectral ops (ReEig/LogEig) to the graph-primitive Jacobi
    // eigensolver on EVERY backend — a no-op unless an f32 spectral op is
    // present. f64 nodes are left for the native LAPACK / host-fallback path, so
    // this never regresses CPU or the f64 GPU host-fallback (see LowerSpectral).
    // Runs before the `supported.is_empty()` early-out because an f32 SPD op has
    // no native kernel anywhere and must always be decomposed.
    graph = LowerSpectral.run(graph);

    // Decompose registered custom ops that opt into a `lower` rule, on EVERY
    // backend (including empty-supported codegen targets below) — a no-op unless
    // the graph carries such an op. Runs before the `supported.is_empty()`
    // early-out so decomposition reaches the standalone/codegen paths too.
    graph = lower_custom_ops(graph);

    if supported.is_empty() {
        return graph;
    }

    for _ in 0..16 {
        if legalize_for_backend(&graph, supported).is_ok() {
            return graph;
        }
        let bad = unsupported_kinds(&graph, supported);
        if bad.is_empty() {
            break;
        }

        let mut changed = false;

        if bad.contains(&OpKind::GroupNorm) {
            graph = LowerGroupNorm.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::BatchNormInference) {
            graph = LowerBatchNormInference.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::ResizeNearest2x) {
            graph = LowerResizeNearest2x.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::Pad) {
            // Every backend except Metal/CUDA (which claim `OpKind::Pad`) lowers
            // pad to full/narrow/reverse/expand/concat here.
            graph = LowerPad.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::Slice) {
            // Non-native backends lower strided slice to narrow/reverse/gather.
            graph = LowerSlice.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::AxialRope2d) {
            // Non-native backends lower SAM2 axial 2-D RoPE to a constant-table
            // `gather` + `mul`/`add` (bit-exact vs the CPU kernel).
            graph = LowerAxialRope2d.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::FakeQuantize) {
            // Non-native backends lower PerBatch fake-quant to
            // `round`/`clamp`/`mul` (one canonical `Round` → consistent ties).
            graph = LowerFakeQuantize.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::Clamp)
            || bad.contains(&OpKind::Tile)
            || bad.contains(&OpKind::Trilu)
        {
            // Clamp/Tile/Trilu decompose to max/min, concat, mul-by-mask.
            graph = LowerStructural.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::CumProd) || bad.contains(&OpKind::CumMax) {
            // CumProd/CumMax decompose to a masked reduce over an inserted
            // query axis (CPU/Metal/CUDA claim them natively).
            graph = LowerCumulative.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::Histogram) {
            // Only CPU claims Histogram natively; everywhere else it decomposes
            // to Compare + mul + Reduce::Sum + Concat.
            graph = LowerHistogram.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::DotGeneral) {
            graph = LowerDotGeneral.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::ScaledGroupedMatMul) {
            // Backends without a native FP4-grouped kernel decompose to
            // ScaledDequantize (both operands) + Transpose + GroupedMatMul
            // (+ per-expert bias gather/add). Runs on every backend that
            // supports the portable GroupedMatMul segmented GEMM.
            graph = LowerScaledGroupedMatMul.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::SynthMatMul) {
            // Backends without a native codebook-synthesis kernel decompose to
            // Cast + Reshape + Gather (reconstruct the dense weight) + Transpose
            // + MatMul. Runs on every backend with the portable MatMul.
            graph = LowerSynthMatMul.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::SynthMatMulBackward) {
            // Backends without the fused synth-backward kernel decompose it to the
            // same primitives the generic VJP used to emit (Gather + MatMul +
            // Transpose + ScatterAdd) — bit-identical, runs everywhere.
            graph = LowerSynthMatMulBackward.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::SynthReconstruct) {
            graph = LowerSynthReconstruct.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::SplineActivation) {
            // Backends without a native KAN spline kernel decompose to the RBF
            // basis expansion (Reshape/Expand/Sub/Mul/Exp) + ReduceSum. All-f32,
            // so this runs on GPU backends too.
            graph = LowerSplineActivation.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::SplineActivationBackwardX)
            || bad.contains(&OpKind::SplineActivationBackwardCoeff)
        {
            // Backends without the fused KAN spline-backward kernels decompose to
            // the same RBF basis + contraction the generic VJP used to emit.
            graph = LowerSplineActivationBackward.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::Fma) {
            // Backends without a native single-rounding FMA fall back to
            // mul+add (two roundings — loses the compensated-arithmetic benefit).
            graph = LowerFma.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::If) || bad.contains(&OpKind::While) {
            graph = LowerControlFlow.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::ReEig) || bad.contains(&OpKind::LogEig) {
            // A backend that doesn't even claim the f64 host-fallback for ReEig/
            // LogEig: lower f32 nodes to the Jacobi eigensolver (f64 nodes remain
            // and will surface as a clear legalize error — f64 SPD needs CPU).
            graph = LowerSpectral.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::Scan) {
            // Backends without a native scan (HLO/codegen: TPU, QNN, Cerebras,
            // WebGL) get Op::Scan unrolled into inlined body replicas. Backends
            // that list `Scan` in their supported set keep it and host-fallback.
            graph = LowerScan.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::ElementwiseRegion) {
            graph = UnfuseElementwiseRegions::FOR_CPU.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::SoftmaxCrossEntropy)
            || bad.contains(&OpKind::SoftmaxCrossEntropyWithLogits)
            || bad.contains(&OpKind::SoftmaxCrossEntropyBackward)
        {
            graph = LowerSoftmaxCrossEntropy.run(graph);
            changed = true;
        }
        #[cfg(feature = "training")]
        if needs_backward_decompose(&bad) {
            graph = rlx_autodiff::decompose_backward_ops_except(graph, supported);
            changed = true;
        }
        if bad.contains(&OpKind::ReluBackward)
            || bad.contains(&OpKind::ActivationBackward)
            || bad.contains(&OpKind::BatchNormInferenceBackwardInput)
            || bad.contains(&OpKind::BatchNormInferenceBackwardGamma)
            || bad.contains(&OpKind::BatchNormInferenceBackwardBeta)
        {
            graph = LowerBackwardOps.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::Reduce)
            || graph.nodes().iter().any(|n| {
                matches!(&n.op, Op::Reduce { axes, .. } if {
                    let rank = graph.shape(n.inputs[0]).rank();
                    axes.len() > 1 || (rank > 0 && axes.as_slice() != [rank - 1])
                })
            })
        {
            graph = LowerNonLastAxisReduce.run(graph);
            changed = true;
        }
        if needs_unfuse(&bad) {
            graph = unfuse_fused_for_autodiff(graph);
            changed = true;
        }

        if !changed {
            break;
        }
    }
    graph
}

/// Legalize, rewriting unsupported ops first when possible.
pub fn legalize_or_rewrite_for_backend(
    graph: Graph,
    supported: &[OpKind],
) -> Result<Graph, Vec<(rlx_ir::NodeId, OpKind)>> {
    legalize_or_rewrite_for_backend_with_config(graph, supported, KernelDispatchConfig::default())
}

/// Legalize with explicit logical-kernel dispatch policy.
pub fn legalize_or_rewrite_for_backend_with_dispatch(
    graph: Graph,
    supported: &[OpKind],
    dispatch: KernelDispatchPolicy,
) -> Result<Graph, Vec<(rlx_ir::NodeId, OpKind)>> {
    legalize_or_rewrite_for_backend_with_config(
        graph,
        supported,
        KernelDispatchConfig::new(dispatch),
    )
}

/// Legalize with full [`KernelDispatchConfig`].
pub fn legalize_or_rewrite_for_backend_with_config(
    graph: Graph,
    supported: &[OpKind],
    config: KernelDispatchConfig,
) -> Result<Graph, Vec<(rlx_ir::NodeId, OpKind)>> {
    if supported.is_empty() {
        return Ok(graph);
    }
    let graph = rewrite_for_backend_with_config(graph, supported, config);
    legalize_for_backend(&graph, supported).map(|()| graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::*;

    #[test]
    fn lower_custom_op_decomposes_to_primitives_for_restricted_backend() {
        use rlx_ir::op::BinaryOp;
        use rlx_ir::{LowerContext, Node, NodeId, OpExtension, register_op};
        use std::sync::Arc;

        // A downstream-style custom op that opts into decomposition: `y = x + x`.
        struct DoubleOp;
        impl OpExtension for DoubleOp {
            fn name(&self) -> &str {
                "test_double_lower"
            }
            fn num_inputs(&self) -> usize {
                1
            }
            fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
                inputs[0].clone()
            }
            fn lower(&self, _node: &Node, ctx: &mut LowerContext) -> Option<NodeId> {
                let x = ctx.inputs[0];
                let shape = ctx.out.node(x).shape.clone();
                Some(
                    ctx.out
                        .add_node(Op::Binary(BinaryOp::Add), vec![x, x], shape),
                )
            }
        }
        register_op(Arc::new(DoubleOp));

        let f = DType::F32;
        let mut g = Graph::new("cust");
        let x = g.input("x", Shape::new(&[4], f));
        let y = g.custom_op("test_double_lower", vec![], vec![x]);
        g.set_outputs(vec![y]);

        // A restricted backend that does NOT claim `Custom` — before lowering the
        // graph is illegal for it; after, it decomposes to Input + Binary.
        let prims = &[OpKind::Input, OpKind::Binary];
        assert!(legalize_for_backend(&g, prims).is_err());

        let lowered = rewrite_for_backend(g, prims);
        assert!(
            legalize_for_backend(&lowered, prims).is_ok(),
            "custom op should legalize after lowering"
        );
        assert!(
            !lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Custom { .. })),
            "the Op::Custom node must be gone"
        );
        assert!(
            lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Binary(BinaryOp::Add))),
            "expected the decomposed Add primitive"
        );
    }

    #[test]
    fn rewrite_lowers_sce_for_cuda_primitives() {
        let f = DType::F32;
        let mut g = Graph::new("sce");
        let logits = g.input("logits", Shape::new(&[4, 4], f));
        let labels = g.input("labels", Shape::new(&[4], f));
        let loss = g.softmax_cross_entropy_with_logits(logits, labels);
        g.set_outputs(vec![loss]);

        let cuda_like = &[
            OpKind::Input,
            OpKind::Constant,
            OpKind::Reduce,
            OpKind::Binary,
            OpKind::Expand,
            OpKind::Activation,
            OpKind::Reshape,
            OpKind::Compare,
            OpKind::Where,
            OpKind::Concat,
            OpKind::Softmax,
        ];
        assert!(legalize_for_backend(&g, cuda_like).is_err());
        let lowered = rewrite_for_backend(g, cuda_like);
        assert!(legalize_for_backend(&lowered, cuda_like).is_ok());
    }

    #[test]
    fn unfuses_fused_matmul_for_minimal_cpu_set() {
        let f = DType::F32;
        let mut g = Graph::new("fused");
        let x = g.input("x", Shape::new(&[2, 8], f));
        let w = g.param("w", Shape::new(&[8, 4], f));
        let b = g.param("b", Shape::new(&[4], f));
        let out = g.fused_matmul_bias_act(x, w, b, None, Shape::new(&[2, 4], f));
        g.set_outputs(vec![out]);

        let supported = &[
            OpKind::Input,
            OpKind::Param,
            OpKind::MatMul,
            OpKind::Binary,
            OpKind::Expand,
            OpKind::Activation,
        ];
        assert!(legalize_for_backend(&g, supported).is_err());

        let rewritten = rewrite_for_backend(g, supported);
        assert!(legalize_for_backend(&rewritten, supported).is_ok());
        assert!(rewritten.nodes().iter().any(|n| matches!(n.op, Op::MatMul)));
        assert!(
            rewritten
                .nodes()
                .iter()
                .all(|n| !matches!(n.op, Op::FusedMatMulBiasAct { .. }))
        );
    }

    #[test]
    fn rewrite_lowers_group_norm_for_minimal_set() {
        let f = DType::F32;
        let mut g = Graph::new("gn");
        let x = g.input("x", Shape::new(&[1, 4, 2, 2], f));
        let gamma = g.param("g", Shape::new(&[4], f));
        let beta = g.param("b", Shape::new(&[4], f));
        let out = g.add_node(
            Op::GroupNorm {
                num_groups: 2,
                eps: 1e-6,
            },
            vec![x, gamma, beta],
            Shape::new(&[1, 4, 2, 2], f),
        );
        g.set_outputs(vec![out]);

        let supported = &[
            OpKind::Input,
            OpKind::Param,
            OpKind::Constant,
            OpKind::Reshape,
            OpKind::Reduce,
            OpKind::Binary,
            OpKind::Expand,
            OpKind::Activation,
            OpKind::Concat,
        ];
        let rewritten = rewrite_for_backend(g, supported);
        assert!(legalize_for_backend(&rewritten, supported).is_ok());
        assert!(
            !rewritten
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::GroupNorm { .. }))
        );
    }

    #[test]
    fn rewrite_lowers_relu_backward_for_metal_primitives() {
        use rlx_ir::{DType, Graph, Shape};

        let f = DType::F32;
        let mut g = Graph::new("rb");
        let x = g.input("x", Shape::new(&[4], f));
        let dy = g.input("dy", Shape::new(&[4], f));
        let dx = g.relu_backward(x, dy);
        g.set_outputs(vec![dx]);

        let metal_supported = &[
            OpKind::Input,
            OpKind::Constant,
            OpKind::Expand,
            OpKind::Compare,
            OpKind::Where,
            OpKind::Binary,
            OpKind::Activation,
        ];
        assert!(legalize_for_backend(&g, metal_supported).is_err());
        let lowered = rewrite_for_backend(g, metal_supported);
        assert!(legalize_for_backend(&lowered, metal_supported).is_ok());
    }

    #[test]
    fn legalize_or_rewrite_returns_graph_on_success() {
        let g = {
            let f = DType::F32;
            let mut g = Graph::new("ok");
            let x = g.input("x", Shape::new(&[2], f));
            let y = g.input("y", Shape::new(&[2], f));
            let s = g.add(x, y);
            g.set_outputs(vec![s]);
            g
        };
        let supported = &[OpKind::Input, OpKind::Binary];
        let out = legalize_or_rewrite_for_backend(g, supported).expect("legal");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn logical_kernel_lowers_gaussian_splat_when_not_supported() {
        use rlx_ir::ops::splat::{GaussianSplatInputs, GaussianSplatRenderParams};

        let f = DType::F32;
        let mut g = Graph::new("splat");
        let n = 2usize;
        let positions = g.input("pos", Shape::new(&[n * 3], f));
        let scales = g.input("sc", Shape::new(&[n * 3], f));
        let rotations = g.input("rot", Shape::new(&[n * 4], f));
        let opacities = g.input("op", Shape::new(&[n], f));
        let colors = g.input("col", Shape::new(&[n * 3], f));
        let sh = g.input("sh", Shape::new(&[n * 3], f));
        let meta = g.input("meta", Shape::new(&[23], f));
        let out = g.gaussian_splat_render(
            GaussianSplatInputs {
                positions,
                scales,
                rotations,
                opacities,
                colors,
                sh_coeffs: sh,
                meta,
            },
            GaussianSplatRenderParams {
                width: 4,
                height: 4,
                ..Default::default()
            },
        );
        g.set_outputs(vec![out]);

        let primitive = &[
            OpKind::Input,
            OpKind::Param,
            OpKind::Constant,
            OpKind::Reshape,
            OpKind::Reduce,
            OpKind::Binary,
            OpKind::Expand,
            OpKind::Concat,
        ];
        let rewritten = rewrite_for_backend_with_config(
            g,
            primitive,
            KernelDispatchConfig::new(KernelDispatchPolicy::PreferNative),
        );
        assert!(legalize_for_backend(&rewritten, primitive).is_ok());
        assert!(
            !rewritten
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::GaussianSplatRender { .. }))
        );
    }

    #[test]
    fn logical_kernel_lowers_gaussian_splat_backward_when_not_supported() {
        use rlx_ir::ops::splat::{
            GaussianSplatBackwardParams, GaussianSplatInputs, GaussianSplatRenderParams,
        };

        let f = DType::F32;
        let mut g = Graph::new("splat_bwd");
        let n = 2usize;
        let positions = g.input("pos", Shape::new(&[n * 3], f));
        let scales = g.input("sc", Shape::new(&[n * 3], f));
        let rotations = g.input("rot", Shape::new(&[n * 4], f));
        let opacities = g.input("op", Shape::new(&[n], f));
        let colors = g.input("col", Shape::new(&[n * 3], f));
        let sh = g.input("sh", Shape::new(&[n * 3], f));
        let meta = g.input("meta", Shape::new(&[23], f));
        let d_loss = g.input("dloss", Shape::new(&[16 * 4], f));
        let inputs = GaussianSplatInputs {
            positions,
            scales,
            rotations,
            opacities,
            colors,
            sh_coeffs: sh,
            meta,
        };
        let bwd = GaussianSplatBackwardParams {
            render: GaussianSplatRenderParams {
                width: 4,
                height: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let packed = g.gaussian_splat_render_backward(inputs, d_loss, bwd);
        g.set_outputs(vec![packed]);

        let primitive = &[
            OpKind::Input,
            OpKind::Constant,
            OpKind::Reshape,
            OpKind::Reduce,
            OpKind::Binary,
            OpKind::Expand,
            OpKind::Concat,
            OpKind::Narrow,
        ];
        let rewritten = rewrite_for_backend_with_config(
            g,
            primitive,
            KernelDispatchConfig::new(KernelDispatchPolicy::PreferNative),
        );
        assert!(legalize_for_backend(&rewritten, primitive).is_ok());
        assert!(
            !rewritten
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::GaussianSplatRenderBackward { .. }))
        );
    }

    #[test]
    fn force_common_kinds_overrides_full_supported_set() {
        use rlx_ir::ops::splat::{GaussianSplatInputs, GaussianSplatRenderParams};

        let f = DType::F32;
        let mut g = Graph::new("force_common");
        let n = 1usize;
        let positions = g.input("pos", Shape::new(&[n * 3], f));
        let scales = g.input("sc", Shape::new(&[n * 3], f));
        let rotations = g.input("rot", Shape::new(&[n * 4], f));
        let opacities = g.input("op", Shape::new(&[n], f));
        let colors = g.input("col", Shape::new(&[n * 3], f));
        let sh = g.input("sh", Shape::new(&[n * 3], f));
        let meta = g.input("meta", Shape::new(&[23], f));
        let out = g.gaussian_splat_render(
            GaussianSplatInputs {
                positions,
                scales,
                rotations,
                opacities,
                colors,
                sh_coeffs: sh,
                meta,
            },
            GaussianSplatRenderParams {
                width: 2,
                height: 2,
                ..Default::default()
            },
        );
        g.set_outputs(vec![out]);

        let full = &[
            OpKind::GaussianSplatRender,
            OpKind::Input,
            OpKind::Reshape,
            OpKind::Reduce,
        ];
        let config = KernelDispatchConfig {
            policy: KernelDispatchPolicy::PreferNative,
            force_common_kinds: &[OpKind::GaussianSplatRender],
            force_native_kinds: &[],
        };
        let rewritten = rewrite_for_backend_with_config(g, full, config);
        assert!(
            !rewritten
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::GaussianSplatRender { .. }))
        );
    }

    #[test]
    fn compile_pipeline_lowers_splat_with_force_common_kinds() {
        use crate::compiler::CompilePipeline;
        use crate::fusion_pipeline::FusionTarget;
        use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
        use rlx_ir::ops::splat::{GaussianSplatInputs, GaussianSplatRenderParams};
        use rlx_ir::{Graph, MirModule};

        let f = DType::F32;
        let mut g = Graph::new("pipe");
        let n = 2usize;
        let positions = g.input("pos", Shape::new(&[n * 3], f));
        let scales = g.input("sc", Shape::new(&[n * 3], f));
        let rotations = g.input("rot", Shape::new(&[n * 4], f));
        let opacities = g.input("op", Shape::new(&[n], f));
        let colors = g.input("col", Shape::new(&[n * 3], f));
        let sh = g.input("sh", Shape::new(&[n * 3], f));
        let meta = g.input("meta", Shape::new(&[23], f));
        let out = g.gaussian_splat_render(
            GaussianSplatInputs {
                positions,
                scales,
                rotations,
                opacities,
                colors,
                sh_coeffs: sh,
                meta,
            },
            GaussianSplatRenderParams {
                width: 4,
                height: 4,
                ..Default::default()
            },
        );
        g.set_outputs(vec![out]);

        let mut pipe = CompilePipeline::new(FusionTarget::Cpu);
        pipe.kernel_dispatch = KernelDispatchConfig {
            policy: KernelDispatchPolicy::PreferNative,
            force_common_kinds: &[OpKind::GaussianSplatRender],
            force_native_kinds: &[],
        };
        let config = KernelDispatchConfig {
            policy: KernelDispatchPolicy::PreferNative,
            force_common_kinds: &[OpKind::GaussianSplatRender],
            force_native_kinds: &[],
        };
        let lowered = rewrite_for_backend_with_config(g.clone(), &[], config);
        assert!(
            !lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::GaussianSplatRender { .. })),
            "empty supported + force_common: {:?}",
            lowered
                .nodes()
                .iter()
                .map(|n| format!("{:?}", n.op.kind()))
                .collect::<Vec<_>>()
        );
        let lowered_full = rewrite_for_backend_with_config(
            g,
            &[
                OpKind::GaussianSplatRender,
                OpKind::Input,
                OpKind::Reshape,
                OpKind::Reduce,
            ],
            config,
        );
        assert!(
            !lowered_full
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::GaussianSplatRender { .. }))
        );

        let (mir, _) = pipe.optimize_with_report(MirModule::from_graph(lowered));
        assert!(!mir.as_graph().nodes().iter().any(|n| {
            matches!(
                n.op,
                Op::GaussianSplatRender { .. } | Op::GaussianSplatRenderBackward { .. }
            )
        }));
    }
}
