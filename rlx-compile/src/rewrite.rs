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

//! Auto-rewriter — decompose unsupported ops into primitives the backend claims.
//!
//! When [`legalize_for_backend`] fails, this module applies structural lowers
//! and fused-op unfuse passes until the graph legalizes or no progress is made.

use std::collections::HashSet;

use rlx_fusion::control_flow::LowerControlFlow;
use rlx_fusion::fusion::UnfuseElementwiseRegions;
use rlx_fusion::lower_dot_general::LowerDotGeneral;
use rlx_fusion::lower_logical_kernels;
use rlx_fusion::lower_vae_ops::{LowerGroupNorm, LowerResizeNearest2x};
use rlx_fusion::pass::Pass;
use rlx_fusion::unfuse::unfuse_fused_for_autodiff;
use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
use rlx_ir::{Graph, OpKind};

use crate::legalize::legalize_for_backend;

const FUSED_KINDS: &[OpKind] = &[
    OpKind::FusedMatMulBiasAct,
    OpKind::FusedSwiGLU,
    OpKind::FusedResidualLN,
    OpKind::FusedResidualRmsNorm,
    OpKind::FusedAttentionBlock,
    OpKind::FusedTransformerLayer,
    OpKind::GatedDeltaNet,
    OpKind::SelectiveScan,
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
        if bad.contains(&OpKind::ResizeNearest2x) {
            graph = LowerResizeNearest2x.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::DotGeneral) {
            graph = LowerDotGeneral.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::If) || bad.contains(&OpKind::While) {
            graph = LowerControlFlow.run(graph);
            changed = true;
        }
        if bad.contains(&OpKind::ElementwiseRegion) {
            graph = UnfuseElementwiseRegions.run(graph);
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
