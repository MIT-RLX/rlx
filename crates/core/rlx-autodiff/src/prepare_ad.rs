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

//! MIR preparation for autodiff — canonical pre-passes shared by reverse-
//! and forward-mode AD.
//!
//! The HIR → MIR → LIR pipeline lowers fusion-friendly blocks to MIR
//! (`FusionPolicy::Direct`) or primitive chains (`FusionPolicy::Fusable` /
//! [`FusionPolicy::for_autodiff`]). Autodiff always runs on **MIR**
//! ([`Graph`]); this module rewrites fused / control-flow / scan shapes
//! into primitives the VJP table covers.
//!
//! Typical training flow:
//!
//! ```text
//! GraphModule (HIR) ──lower──▶ MirModule ──prepare_graph_for_ad──▶ MirModule
//!                                      └──grad_with_loss──▶ backward Graph
//! Compile forward (fused) and backward (AD + cleanup) via
//! [`rlx_compile::CompilePipeline::compile_training`] when the `training`
//! feature is enabled on `rlx-compile`; backward params alias the forward
//! weight layout instead of duplicating arena storage.
//! ```

use rlx_ir::hir::LowerError;
use rlx_ir::mir::MirModule;
use rlx_ir::{Graph, GraphModule, GraphStage, NodeId};

use rlx_fusion::pass::Pass;

pub use crate::autodiff::{convert_scans_for_ad, inline_custom_fn_for_autodiff};
pub use rlx_fusion::unfuse_fused_for_autodiff;

/// Error from [`grad_with_loss_module`] / [`jvp_module`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutodiffError {
    /// Autodiff requires MIR (or HIR to lower first). LIR carries a buffer
    /// plan that does not apply to a freshly built gradient graph.
    WrongStage {
        got: GraphStage,
        hint: &'static str,
    },
    Lower(LowerError),
}

impl std::fmt::Display for AutodiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongStage { got, hint } => {
                write!(f, "autodiff: cannot run on {got:?} stage — {hint}")
            }
            Self::Lower(e) => write!(f, "HIR lower failed: {e}"),
        }
    }
}

impl std::error::Error for AutodiffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lower(e) => Some(e),
            _ => None,
        }
    }
}

/// Canonical MIR pre-passes before reverse- or forward-mode AD.
///
/// Order:
/// 0. [`DecomposeFusionRegions`](rlx_fusion::DecomposeFusionRegions) — FKL batch/transform → primitives
/// 1. [`UnfuseElementwiseRegions`](rlx_fusion::UnfuseElementwiseRegions)
/// 2. [`legalize_multi_axis_reduce`](crate::legalize_reduce::legalize_multi_axis_reduce)
/// 3. [`rlx_fusion::unfuse_fused_for_autodiff`] — tier-2 fused ops → primitives
/// 4. [`LowerDotGeneral`](rlx_fusion::LowerDotGeneral)
/// 5. [`control_flow::inline_if`]
/// 6. [`control_flow::unroll_while`]
/// 7. [`inline_custom_fn_for_autodiff`]
/// 8. [`convert_scans_for_ad`]
///
/// **Why `legalize_multi_axis_reduce` runs early** (step 2, before
/// `unfuse_fused_for_autodiff`). The unfuse walker assumes every input
/// is already mapped by the time its consumer is visited; a user-built
/// `Op::Reduce { axes: [0, 1], .. }` in a custom loss head can leave
/// a dangling `id_map[i]` lookup inside the unfuser. Decomposing
/// multi-axis Reduce into a single-axis chain first keeps the walk in
/// strict topo order and is a no-op for graphs that never used
/// multi-axis reduce.
pub fn prepare_graph_for_ad(g: Graph) -> Graph {
    use rlx_fusion::pass::Pass as _;
    let g = rlx_fusion::DecomposeFusionRegions.run(g);
    let g = rlx_fusion::UnfuseElementwiseRegions::FOR_CPU.run(g);
    let g = crate::legalize_reduce::legalize_multi_axis_reduce(g);
    let g = rlx_fusion::unfuse_fused_for_autodiff(g);
    let g = rlx_fusion::LowerDotGeneral.run(g);
    let g = rlx_fusion::control_flow::inline_if(g);
    let g = rlx_fusion::control_flow::unroll_while(g);
    let g = inline_custom_fn_for_autodiff(g);
    let g = convert_scans_for_ad(g);
    crate::fuse_splat::fuse_decomposed_gaussian_splat(g)
}

/// [`Pass`] wrapper for [`prepare_graph_for_ad`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PrepareForAutodiff;

impl Pass for PrepareForAutodiff {
    fn name(&self) -> &str {
        "prepare_for_autodiff"
    }

    fn run(&self, graph: Graph) -> Graph {
        prepare_graph_for_ad(graph)
    }
}

/// Return MIR suitable for inspection or a custom AD walk.
pub fn prepare_mir_for_ad(mir: MirModule) -> MirModule {
    MirModule::from_graph(prepare_graph_for_ad(mir.into_graph()))
}

/// Lower HIR if needed, then run [`prepare_graph_for_ad`].
pub fn prepare_module_for_ad(module: GraphModule) -> Result<GraphModule, AutodiffError> {
    let mir = module_into_mir(module)?;
    Ok(MirModule::from_graph(prepare_graph_for_ad(mir.into_graph())).into())
}

/// Reverse-mode AD on a [`GraphModule`] at HIR or MIR stage.
pub fn grad_with_loss_module(module: GraphModule, wrt: &[NodeId]) -> Result<Graph, AutodiffError> {
    let mir = module_into_mir(module)?;
    Ok(crate::autodiff::grad_with_loss(mir.as_graph(), wrt))
}

/// Forward-mode AD on a [`GraphModule`] at HIR or MIR stage.
pub fn jvp_module(module: GraphModule, tangent_for: &[NodeId]) -> Result<Graph, AutodiffError> {
    let mir = module_into_mir(module)?;
    Ok(crate::autodiff_fwd::jvp(mir.as_graph(), tangent_for))
}

/// Hessian-vector product module wrapper (`wrt` params get tangent inputs).
pub fn hvp_module(module: GraphModule, wrt: &[NodeId]) -> Result<Graph, AutodiffError> {
    let mir = module_into_mir(module)?;
    Ok(crate::autodiff_fwd::hvp(mir.as_graph(), wrt))
}

/// Higher-order reverse-mode AD on a [`GraphModule`].
pub fn nth_order_grad_module(
    module: GraphModule,
    wrt_name: &str,
    order: usize,
) -> Result<Graph, AutodiffError> {
    let mir = module_into_mir(module)?;
    Ok(crate::higher_order::nth_order_grad(
        mir.as_graph(),
        wrt_name,
        order,
    ))
}

fn module_into_mir(module: GraphModule) -> Result<MirModule, AutodiffError> {
    match module.stage() {
        GraphStage::Lir => Err(AutodiffError::WrongStage {
            got: GraphStage::Lir,
            hint: "use the embedded `mir` from LIR or rebuild from HIR/MIR before AD",
        }),
        GraphStage::Hir => module.into_mir().map_err(AutodiffError::Lower),
        GraphStage::Mir => module.into_mir().map_err(AutodiffError::Lower),
    }
}

/// MIR extensions for the training pipeline.
pub trait MirAutodiffExt {
    /// Run [`prepare_graph_for_ad`] and return primitive MIR.
    fn prepare_for_autodiff(self) -> MirModule;

    /// [`crate::autodiff::grad_with_loss`] on this module's graph.
    fn grad_with_loss(&self, wrt: &[NodeId]) -> Graph;
}

impl MirAutodiffExt for MirModule {
    fn prepare_for_autodiff(self) -> MirModule {
        prepare_mir_for_ad(self)
    }

    fn grad_with_loss(&self, wrt: &[NodeId]) -> Graph {
        crate::autodiff::grad_with_loss(self.as_graph(), wrt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::Op;
    use rlx_ir::{DType, Shape};

    fn f32_shape(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn hir_direct_linear_grad_module() {
        let module = GraphModule::define("layer", |m| {
            let x = m.input("x", f32_shape(&[2, 8]));
            let w = m.param("w", f32_shape(&[8, 8]));
            let b = m.param("b", f32_shape(&[8]));
            m.linear(x, w, Some(b), None, f32_shape(&[2, 8]))
        });
        let mir = module.into_mir().expect("lower");
        assert!(
            mir.as_graph()
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedMatMulBiasAct { .. })),
            "Direct HIR should lower to FusedMatMulBiasAct"
        );

        let w = mir
            .as_graph()
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Param { name } if name == "w"))
            .map(|n| n.id)
            .expect("param w");
        let bwd = grad_with_loss_module(GraphModule::from_mir(mir), &[w]).expect("grad");
        assert!(
            !bwd.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedMatMulBiasAct { .. })),
            "backward graph should not retain fused ops"
        );
        assert!(bwd.outputs.len() >= 2);
    }

    #[test]
    fn prepare_for_autodiff_decomposes_batch_elementwise_region() {
        use rlx_fusion::fk_fusion::{FuseBatchPreprocess, MarkBatchSliceRegions};
        use rlx_fusion::fk_graphs::batch_narrow_relu_primitive_graph;
        let g = batch_narrow_relu_primitive_graph("batch_ad", 2, 3, 8, 8);
        let fused = FuseBatchPreprocess.run(MarkBatchSliceRegions.run(g));
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
        );
        let prep = prepare_graph_for_ad(fused);
        assert!(
            !prep
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. })),
            "AD prep should decompose BatchElementwiseRegion"
        );
    }

    #[test]
    fn prepare_for_autodiff_pass_matches_fn() {
        let mut g = Graph::new("t");
        let x = g.input("x", f32_shape(&[4]));
        g.set_outputs(vec![x]);
        let via_pass = PrepareForAutodiff.run(g.clone());
        let via_fn = prepare_graph_for_ad(g);
        assert_eq!(via_pass.len(), via_fn.len());
    }
}
