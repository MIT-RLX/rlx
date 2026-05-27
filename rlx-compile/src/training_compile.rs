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

//! Training compile — fused forward optimization + backward cleanup with
//! shared weight layout.
//!
//! Forward MIR runs the full fusion pipeline (fused kernels where the
//! backend supports them). Backward MIR is produced by
//! [`rlx_autodiff::grad_with_loss`] on the **optimized** forward graph and
//! only runs cleanup passes (no forward-only fusion that would reintroduce
//! ops AD cannot differentiate without unfusing).
//!
//! Parameters are planned once on the forward graph; the backward plan
//! aliases param nodes onto those offsets via [`SharedWeightLayout`] so
//! weights are not duplicated in the backward activation arena.

use rlx_ir::mir::MirModule;
use rlx_ir::{GraphModule, GraphStage, NodeId};

use crate::DeadCodeElimination;
use crate::compiler::lir_buffer_plan_from_memory;
use crate::compiler::{CompilePipeline, CompileResult};
use crate::memory::{
    MemoryPlanOptions, SharedWeightLayout, plan_memory_backward, plan_memory_with_options,
};
use rlx_fusion::fusion_report::FusionReport;
use rlx_fusion::pass::{Pass, run_passes};
use rlx_ir::lir::LirModule;

/// Forward + backward LIR with a single shared weight region.
#[derive(Debug, Clone)]
pub struct TrainingCompileResult {
    pub forward: CompileResult,
    pub backward: CompileResult,
    pub weights: SharedWeightLayout,
}

impl CompilePipeline {
    /// HIR/MIR → forward LIR (fused) + backward LIR (AD + cleanup), shared weights.
    pub fn compile_training(
        &self,
        module: GraphModule,
        wrt: &[NodeId],
    ) -> Result<TrainingCompileResult, TrainingCompileError> {
        let mir = match module.stage() {
            GraphStage::Hir => Self::lower_hir(
                module
                    .into_hir()
                    .expect("GraphModule stage() / into_hir mismatch"),
            )?,
            GraphStage::Mir => module.into_mir()?,
            GraphStage::Lir => {
                return Err(TrainingCompileError::WrongStage {
                    hint: "compile forward/backward from HIR or MIR, not LIR",
                });
            }
        };
        Ok(self.compile_training_mir(mir, wrt))
    }

    /// MIR → forward LIR (fused) + backward LIR (AD + cleanup), shared weights.
    pub fn compile_training_mir(&self, mir: MirModule, wrt: &[NodeId]) -> TrainingCompileResult {
        let (fwd_mir, fusion) = self.optimize_with_report(mir);
        let fwd_graph = fwd_mir.as_graph().clone();
        let fwd_plan = plan_memory_with_options(
            &fwd_graph,
            self.arena_alignment,
            MemoryPlanOptions::inference(),
        );
        let weights = SharedWeightLayout::from_forward(&fwd_graph, &fwd_plan);
        let fwd_lir = LirModule::new(
            fwd_mir.clone(),
            lir_buffer_plan_from_memory(&fwd_graph, &fwd_plan, self.arena_alignment),
        );

        let bwd_graph = rlx_autodiff::grad_with_loss(fwd_mir.as_graph(), wrt);
        let bwd_mir = self.optimize_backward(MirModule::from_graph(bwd_graph));
        let bwd_graph = bwd_mir.as_graph().clone();
        let bwd_plan = plan_memory_backward(&bwd_graph, self.arena_alignment, &weights);
        let bwd_lir = LirModule::new(
            bwd_mir,
            lir_buffer_plan_from_memory(&bwd_graph, &bwd_plan, self.arena_alignment),
        );

        if self.assert_fusion_clean && !fusion.missed.is_empty() {
            panic!(
                "fusion contract violated: {} missed patterns\n{fusion}",
                fusion.missed.len()
            );
        }

        TrainingCompileResult {
            forward: CompileResult {
                lir: fwd_lir,
                fusion,
            },
            backward: CompileResult {
                lir: bwd_lir,
                fusion: FusionReport::default(),
            },
            weights,
        }
    }

    /// Cleanup passes for backward MIR — no forward-only fusion.
    pub fn optimize_backward(&self, mir: MirModule) -> MirModule {
        let passes = backward_cleanup_passes();
        let graph = run_passes(mir.into_graph(), &passes, false);
        MirModule::from_graph(self.legalize_after_fusion(graph))
    }
}

/// Error from [`CompilePipeline::compile_training`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingCompileError {
    WrongStage { hint: &'static str },
    Lower(rlx_ir::hir::LowerError),
}

impl std::fmt::Display for TrainingCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongStage { hint } => write!(f, "training compile: {hint}"),
            Self::Lower(e) => write!(f, "HIR lower failed: {e}"),
        }
    }
}

impl std::error::Error for TrainingCompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lower(e) => Some(e),
            _ => None,
        }
    }
}

impl From<rlx_ir::hir::LowerError> for TrainingCompileError {
    fn from(e: rlx_ir::hir::LowerError) -> Self {
        Self::Lower(e)
    }
}

/// Passes safe on backward MIR after AD (no forward-only fusion).
pub fn backward_cleanup_passes() -> Vec<&'static dyn Pass> {
    // No forward-only fusion; skip `LowerControlFlow` (AD graphs are already
    // primitive and the pass can disturb reduce output shapes).
    vec![&DeadCodeElimination, &rlx_fusion::LowerDotGeneral]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::ReduceOp;
    use rlx_ir::{DType, Graph, Op, Shape};

    fn f32_shape(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn training_compile_shares_param_offsets() {
        let mut fwd = Graph::new("layer");
        let x = fwd.input("x", f32_shape(&[2, 8]));
        let w = fwd.param("w", f32_shape(&[8, 8]));
        let mm = fwd.matmul(x, w, f32_shape(&[2, 8]));
        let loss = fwd.reduce(
            mm,
            ReduceOp::Sum,
            vec![0, 1],
            false,
            Shape::new(&[], DType::F32),
        );
        fwd.set_outputs(vec![loss]);

        let fwd_plan = plan_memory_with_options(&fwd, 64, MemoryPlanOptions::inference());
        let weights = SharedWeightLayout::from_forward(&fwd, &fwd_plan);

        let bwd = rlx_autodiff::grad_with_loss(&fwd, &[w]);
        let bwd_plan = plan_memory_backward(&bwd, 64, &weights);

        let bwd_w = bwd
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Param { name } if name == "w"))
            .map(|n| n.id)
            .expect("bwd w");
        assert_eq!(
            bwd_plan.assignments[&bwd_w].offset, fwd_plan.assignments[&w].offset,
            "backward param should alias forward weight offset"
        );
        assert!(bwd_plan.arena_size >= weights.arena_size);
    }

    #[test]
    fn backward_cleanup_does_not_emit_fused_matmul() {
        let mut g = Graph::new("t");
        let x = g.input("x", f32_shape(&[2, 4]));
        let w = g.param("w", f32_shape(&[4, 4]));
        let mm = g.matmul(x, w, f32_shape(&[2, 4]));
        g.set_outputs(vec![mm]);
        let pipe = CompilePipeline::new(crate::fusion_pipeline::FusionTarget::Cpu);
        let (fwd, _) = pipe.optimize_with_report(MirModule::from_graph(g.clone()));
        let bwd = rlx_autodiff::grad_with_loss(fwd.as_graph(), &[w]);
        let bwd_opt = pipe.optimize_backward(MirModule::from_graph(bwd));
        assert!(
            !bwd_opt
                .as_graph()
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedMatMulBiasAct { .. })),
            "backward cleanup must not introduce forward fusion"
        );
    }
}
