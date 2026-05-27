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

//! HIR → MIR → LIR compiler pipeline.
//!
//! Explicit staging for the RLX compiler:
//!
//! ```text
//! HIR (blocks)  ──lower──▶  MIR (tensor DAG)  ──opt──▶  MIR  ──plan──▶  LIR
//! ```
//!
//! Backends consume [`CompileResult`] / [`LirModule`] (optimized MIR +
//! buffer plan + fusion report) and lower to device-specific thunks.

use rlx_ir::dynamic::collect_dynamic_symbols;
use rlx_ir::hir::HirModule;
use rlx_ir::lir::{LirBufferPlan, LirBufferSlot, LirIoManifest, LirModule, LirViewAlias};
use rlx_ir::mir::MirModule;
use rlx_ir::phase::derive_phases;
use rlx_ir::{Graph, GraphModule, GraphStage};

use crate::DeadCodeElimination;
use crate::debug_assert_graph;
use crate::fusion_pipeline::{
    FusionOptions, FusionTarget, fusion_limits_for_target, fusion_passes_for_supported,
    supported_for_target,
};
use crate::legalize::{format_legalize_error, legalize_for_backend};
use crate::memory::{self, MemoryPlan};
use crate::rewrite::rewrite_for_backend_with_config;
use rlx_fusion::fusion_report::FusionReport;
use rlx_fusion::pass::run_passes;
use rlx_fusion::{clip_elementwise_regions, with_fusion_limits};
use rlx_ir::OpKind;
use rlx_ir::logical_kernel::KernelDispatchConfig;

/// End-to-end compiler output: optimized LIR + fusion diagnostics.
#[derive(Debug, Clone)]
pub struct CompileResult {
    pub lir: LirModule,
    pub fusion: FusionReport,
}

impl CompileResult {
    pub fn has_dynamic_dims(&self) -> bool {
        self.lir.has_dynamic_dims()
    }

    pub fn dynamic_symbols(&self) -> &[u32] {
        self.lir.dynamic_symbols()
    }

    /// Re-plan buffers after binding symbolic dims to concrete sizes.
    pub fn specialize(&self, pipeline: &CompilePipeline, binding: &rlx_ir::DimBinding) -> Self {
        Self {
            lir: pipeline.specialize_lir(&self.lir, binding),
            fusion: self.fusion.clone(),
        }
    }
}

/// End-to-end compiler pipeline configuration.
#[derive(Debug, Clone, Copy)]
pub struct CompilePipeline {
    pub target: FusionTarget,
    pub opts: FusionOptions,
    pub arena_alignment: usize,
    /// When true, [`compile_hir`] / [`compile_graph`] panic if fusion
    /// diagnostics report missed block-level patterns.
    pub assert_fusion_clean: bool,
    /// Backend op claim set. When `Some` and non-empty, fusion passes
    /// are gated on these kinds and the optimized graph is legalized
    /// afterward. When `None`, [`supported_for_target`] is used.
    pub supported_ops: Option<&'static [OpKind]>,
    /// Native vs common IR lowering for logical kernels (see `rlx_ir::logical_kernel`).
    pub kernel_dispatch: KernelDispatchConfig,
}

impl Default for CompilePipeline {
    fn default() -> Self {
        Self {
            target: FusionTarget::Cpu,
            opts: FusionOptions::for_cpu(),
            arena_alignment: 64,
            assert_fusion_clean: false,
            supported_ops: None,
            kernel_dispatch: KernelDispatchConfig::from_env(),
        }
    }
}

impl CompilePipeline {
    pub fn new(target: FusionTarget) -> Self {
        let mut opts = match target {
            FusionTarget::Cpu => FusionOptions::for_cpu(),
            FusionTarget::Metal => FusionOptions::from_metal_env(),
            _ => FusionOptions::default(),
        };
        opts.fusion_limits = fusion_limits_for_target(target);
        Self {
            target,
            opts,
            ..Self::default()
        }
    }

    pub fn with_assert_fusion_clean(mut self, assert: bool) -> Self {
        self.assert_fusion_clean = assert;
        self
    }

    /// HIR → MIR (block lowering only).
    pub fn lower_hir(hir: HirModule) -> Result<MirModule, rlx_ir::hir::LowerError> {
        let mir = hir.lower_to_mir()?;
        debug_assert_graph!(mir.as_graph(), "hir→mir");
        Ok(mir)
    }

    /// Optional cleanup before fusion (DCE + control-flow lowering).
    pub fn preprocess_mir(mir: MirModule) -> MirModule {
        use rlx_fusion::pass::Pass as _;
        let graph = rlx_fusion::control_flow::LowerControlFlow.run(mir.into_graph());
        let graph = DeadCodeElimination.run(graph);
        MirModule::from_graph(graph)
    }

    pub fn with_supported_ops(mut self, ops: &'static [OpKind]) -> Self {
        self.supported_ops = Some(ops);
        self
    }

    pub fn with_kernel_dispatch(
        mut self,
        policy: rlx_ir::logical_kernel::KernelDispatchPolicy,
    ) -> Self {
        self.kernel_dispatch.policy = policy;
        self
    }

    pub fn with_kernel_dispatch_config(mut self, config: KernelDispatchConfig) -> Self {
        self.kernel_dispatch = config;
        self
    }

    fn effective_supported(&self) -> &'static [OpKind] {
        self.supported_ops
            .unwrap_or_else(|| supported_for_target(self.target))
    }

    fn backend_name(&self) -> &'static str {
        match self.target {
            FusionTarget::Cpu => "cpu",
            FusionTarget::Metal => "metal",
            FusionTarget::Mlx => "mlx",
            FusionTarget::Wgpu => "wgpu",
            FusionTarget::Cuda => "cuda",
            FusionTarget::Rocm => "rocm",
            FusionTarget::Tpu => "tpu",
        }
    }

    /// Run fusion + cleanup passes on MIR, returning fusion diagnostics.
    pub fn optimize_with_report(&self, mir: MirModule) -> (MirModule, FusionReport) {
        let before = mir.as_graph().clone();
        let passes = fusion_passes_for_supported(self.effective_supported(), self.opts);
        let limits = self.opts.fusion_limits;
        let graph = with_fusion_limits(limits, || run_passes(mir.into_graph(), &passes, false));
        let graph = clip_elementwise_regions(graph, limits);
        debug_assert_graph!(&graph, "fusion");
        let graph = self.legalize_after_fusion(graph);
        debug_assert_graph!(&graph, "legalize");
        let mir = MirModule::from_graph(graph);
        let fusion = FusionReport::analyze(&before, mir.as_graph());
        (mir, fusion)
    }

    /// Rewrite / legalize fused IR against the backend op claim set.
    /// Runs when [`supported_ops`](Self::supported_ops) is set (including
    /// auto-wiring from [`Backend::supported_ops`] in [`crate::stages::pipeline_for`]).
    pub(crate) fn legalize_after_fusion(&self, graph: Graph) -> Graph {
        let Some(supported) = self.supported_ops else {
            if self.kernel_dispatch.force_common_kinds.is_empty()
                && self.kernel_dispatch.policy
                    == rlx_ir::logical_kernel::KernelDispatchPolicy::PreferNative
            {
                return graph;
            }
            return rewrite_for_backend_with_config(graph, &[], self.kernel_dispatch);
        };
        if supported.is_empty() {
            return graph;
        }
        let graph = rewrite_for_backend_with_config(graph, supported, self.kernel_dispatch);
        if let Err(errors) = legalize_for_backend(&graph, supported) {
            panic!("{}", format_legalize_error(self.backend_name(), &errors));
        }
        graph
    }

    /// Run fusion + cleanup passes on MIR.
    pub fn optimize(&self, mir: MirModule) -> MirModule {
        self.optimize_with_report(mir).0
    }

    /// MIR → LIR (memory plan + schedule + phases + I/O manifest).
    pub fn plan_lir(&self, mir: MirModule) -> LirModule {
        self.plan_lir_with_options(mir, memory::MemoryPlanOptions::default())
    }

    /// MIR → LIR with explicit boundary allocation policy.
    pub fn plan_lir_with_options(
        &self,
        mir: MirModule,
        opts: memory::MemoryPlanOptions,
    ) -> LirModule {
        let graph = mir.as_graph().clone();
        let plan = memory::plan_memory_with_options(&graph, self.arena_alignment, opts);
        LirModule::new(
            mir,
            lir_buffer_plan_from_memory(&graph, &plan, self.arena_alignment),
        )
    }

    /// Bind symbolic dims and re-run buffer planning on specialized MIR.
    pub fn specialize_lir(&self, lir: &LirModule, binding: &rlx_ir::DimBinding) -> LirModule {
        use rlx_ir::dynamic::{
            bind_graph, sync_concat_shapes, sync_graph_shapes, sync_narrow_ops, sync_reshape_ops,
        };
        let mut bound = bind_graph(lir.as_graph(), binding);
        sync_reshape_ops(&mut bound);
        sync_concat_shapes(&mut bound);
        sync_narrow_ops(&mut bound);
        sync_graph_shapes(&mut bound);
        debug_assert_graph!(&bound, "specialize");
        self.plan_lir(MirModule::from_graph(bound))
    }

    fn finish(&self, mir: MirModule, fusion: FusionReport) -> CompileResult {
        debug_assert_graph!(mir.as_graph(), "pre-lir");
        if self.assert_fusion_clean && !fusion.missed.is_empty() {
            panic!(
                "fusion contract violated: {} missed patterns\n{fusion}",
                fusion.missed.len()
            );
        }
        CompileResult {
            lir: self.plan_lir(mir),
            fusion,
        }
    }

    /// HIR → LIR in one call with fusion report.
    pub fn compile_hir(&self, hir: HirModule) -> Result<CompileResult, rlx_ir::hir::LowerError> {
        if rlx_ir::env::var("RLX_IR_DUMP").is_some() {
            let name = hir.name.clone();
            let dump = crate::inspect::inspect_pipeline(self, hir.clone())?;
            crate::inspect::maybe_dump_pipeline(&dump, &name);
        }
        let mir = Self::lower_hir(hir)?;
        let (mir, fusion) = self.optimize_with_report(mir);
        Ok(self.finish(mir, fusion))
    }

    /// Legacy MIR entry: optimize + plan with fusion report.
    pub fn compile_mir(&self, mir: MirModule) -> CompileResult {
        let (mir, fusion) = self.optimize_with_report(mir);
        self.finish(mir, fusion)
    }

    /// Legacy entry: optimize an existing graph and plan buffers.
    pub fn compile_graph(&self, graph: Graph) -> CompileResult {
        self.compile_mir(MirModule::from_graph(graph))
    }

    /// Unified entry for [`GraphModule`] at any pipeline stage.
    pub fn compile_module(
        &self,
        module: GraphModule,
    ) -> Result<CompileResult, rlx_ir::hir::LowerError> {
        match module.stage() {
            GraphStage::Hir => {
                let hir = module
                    .into_hir()
                    .expect("GraphModule stage() / into_hir mismatch");
                self.compile_hir(hir)
            }
            GraphStage::Mir => {
                let mir = module.into_mir()?;
                Ok(self.compile_mir(mir))
            }
            GraphStage::Lir => Ok(CompileResult {
                lir: module
                    .into_lir()
                    .expect("GraphModule stage() / into_lir mismatch"),
                fusion: FusionReport::default(),
            }),
        }
    }
}

impl From<&MemoryPlan> for LirBufferPlan {
    fn from(plan: &MemoryPlan) -> Self {
        LirBufferPlan {
            arena_size: plan.arena_size,
            assignments: plan
                .assignments
                .iter()
                .map(|(id, slot)| {
                    (
                        *id,
                        LirBufferSlot {
                            offset: slot.offset,
                            size: slot.size,
                        },
                    )
                })
                .collect(),
            schedule: plan.schedule.clone(),
            ..Default::default()
        }
    }
}

impl From<&LirBufferPlan> for MemoryPlan {
    fn from(plan: &LirBufferPlan) -> Self {
        MemoryPlan {
            arena_size: plan.arena_size,
            assignments: plan
                .assignments
                .iter()
                .map(|(id, slot)| {
                    (
                        *id,
                        memory::BufferSlot {
                            offset: slot.offset,
                            size: slot.size,
                        },
                    )
                })
                .collect(),
            schedule: plan.schedule.clone(),
        }
    }
}

pub(crate) fn lir_buffer_plan_from_memory(
    graph: &Graph,
    plan: &MemoryPlan,
    alignment: usize,
) -> LirBufferPlan {
    let view_aliases = memory::collect_view_aliases(graph)
        .into_iter()
        .map(|(id, (root, byte_offset))| (id, LirViewAlias { root, byte_offset }))
        .collect();
    LirBufferPlan {
        arena_size: plan.arena_size,
        assignments: plan
            .assignments
            .iter()
            .map(|(id, slot)| {
                (
                    *id,
                    LirBufferSlot {
                        offset: slot.offset,
                        size: slot.size,
                    },
                )
            })
            .collect(),
        schedule: plan.schedule.clone(),
        view_aliases,
        phases: derive_phases(graph),
        io: LirIoManifest::collect(graph),
        alignment,
        dynamic_symbols: collect_dynamic_symbols(graph),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;
    use rlx_ir::Op;
    use rlx_ir::Shape;
    use rlx_ir::hir::FusionPolicy;

    fn f32_shape(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn pipeline_hir_to_lir() {
        let mut hir = HirModule::new("layer");
        let x = hir.input("x", f32_shape(&[2, 128]));
        let w = hir.param("w", f32_shape(&[128, 128]));
        let b = hir.param("b", f32_shape(&[128]));
        let h = hir.linear(x, w, Some(b), None, f32_shape(&[2, 128]));
        hir.outputs = vec![h];

        let pipe = CompilePipeline::new(FusionTarget::Cpu);
        let result = pipe.compile_hir(hir).expect("compile");
        assert!(result.lir.mir.len() <= 5);
        assert!(result.lir.arena_size() > 0);
        assert!(result.lir.buffers.bytes_saved() <= result.lir.buffers.total_unshared_bytes());
        assert!(result.fusion.fused_matmul_bias_act >= 1 || result.lir.mir.len() <= 5);
    }

    #[test]
    fn direct_hir_swiglu_emits_fused_op() {
        let mut hir = HirModule::new("ffn");
        let x = hir.input("x", f32_shape(&[4, 768]));
        let up_w = hir.param("up", f32_shape(&[768, 2048]));
        let gate_w = hir.param("gate", f32_shape(&[768, 2048]));
        let down_w = hir.param("down", f32_shape(&[2048, 768]));
        let out = hir.swiglu_ffn(x, up_w, gate_w, down_w, f32_shape(&[4, 768]));
        hir.outputs = vec![out];

        let pipe = CompilePipeline::new(FusionTarget::Cpu);
        let result = pipe.compile_hir(hir).expect("compile");
        let g = result.lir.mir.as_graph();
        assert!(
            g.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedSwiGLU { .. })),
            "direct HIR SwiGLU should lower to FusedSwiGLU"
        );
        assert!(result.fusion.missed_matmul_bias_act() == 0 || result.fusion.fused_swiglu >= 1);
    }

    #[test]
    fn compile_module_from_graph_define() {
        let module = GraphModule::define("ffn", |m| {
            let x = m.input("x", f32_shape(&[2, 64]));
            let w = m.param("w", f32_shape(&[64, 64]));
            m.linear(x, w, None, None, f32_shape(&[2, 64]))
        });
        assert_eq!(module.stage(), GraphStage::Hir);

        let pipe = CompilePipeline::new(FusionTarget::Cpu);
        let result = pipe.compile_module(module).expect("compile_module");
        assert!(result.lir.arena_size() > 0);
    }

    #[test]
    fn fusable_policy_leaves_room_for_passes() {
        let mut hir = HirModule::new("ffn").with_fusion_policy(FusionPolicy::Fusable);
        let x = hir.input("x", f32_shape(&[4, 768]));
        let up_w = hir.param("up", f32_shape(&[768, 2048]));
        let gate_w = hir.param("gate", f32_shape(&[768, 2048]));
        let down_w = hir.param("down", f32_shape(&[2048, 768]));
        let out = hir.swiglu_ffn(x, up_w, gate_w, down_w, f32_shape(&[4, 768]));
        hir.outputs = vec![out];

        let mir = CompilePipeline::lower_hir(hir).expect("lower");
        let g = mir.as_graph();
        assert!(g.nodes().iter().any(|n| matches!(n.op, Op::MatMul)));
        assert_eq!(g.len(), 9);

        let pipe = CompilePipeline::new(FusionTarget::Cpu);
        let result = pipe.compile_mir(mir);
        assert!(result.fusion.fused_swiglu >= 1);
    }

    #[test]
    fn lir_plan_includes_phases_io_and_fingerprint() {
        use rlx_ir::phase::Phase;

        let mut hir = HirModule::new("stream");
        let x = hir.input("x", f32_shape(&[1, 8]));
        let w = hir.param("w", f32_shape(&[8, 4]));
        let mm = hir.linear(x, w, None, None, f32_shape(&[1, 4]));
        hir.set_outputs(vec![mm]);

        let result = CompilePipeline::new(FusionTarget::Cpu)
            .compile_hir(hir)
            .expect("compile");
        assert!(!result.lir.buffers.phases.is_empty());
        let input_id = result.lir.buffers.io.inputs[0].1;
        assert_eq!(
            result.lir.buffers.phases.get(input_id),
            Some(Phase::Prologue)
        );
        assert_eq!(result.lir.buffers.io.inputs.len(), 1);
        assert_eq!(result.lir.fingerprint(), result.lir.fingerprint());
        assert_eq!(result.lir.buffers.alignment, 64);
    }

    #[test]
    fn dynamic_graph_compiles_and_specializes() {
        use rlx_ir::DimBinding;
        use rlx_ir::infer::GraphExt as _;
        use rlx_ir::sym;

        let mut g = Graph::new("dyn");
        let x = g.input("x", Shape::batch_seq_2d(sym::BATCH, sym::SEQ, DType::F32));
        let w = g.param("w", Shape::new(&[4, 8], DType::F32));
        let y = g.mm(x, w);
        g.set_outputs(vec![y]);

        let pipe = CompilePipeline::new(FusionTarget::Cpu);
        let result = pipe.compile_graph(g);
        assert!(result.has_dynamic_dims());
        assert!(result.lir.buffers.dynamic_symbols.contains(&sym::SEQ));

        let bound = result.specialize(&pipe, &DimBinding::batch_seq(2, 16));
        assert!(bound.lir.is_fully_static());
        assert!(bound.lir.arena_size() > 0);
    }
}
