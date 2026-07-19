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
use rlx_ir::{Graph, GraphModule, GraphStage, NodeId};

use crate::DeadCodeElimination;
use crate::debug_assert_graph;
use crate::fusion_pipeline::{
    FusionOptions, FusionTarget, fusion_limits_for_target, fusion_passes_for_supported,
    supported_for_target,
};
use crate::fusion_target::with_fusion_target;
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
    /// Override the legalize-error backend label (e.g. `"vulkan"` /
    /// `"oneapi"` when [`FusionTarget`] is still [`FusionTarget::Wgpu`]
    /// for shared fusion patterns). When `None`, derived from `target`.
    pub backend_label: Option<&'static str>,
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
            backend_label: None,
            kernel_dispatch: KernelDispatchConfig::from_env(),
        }
    }
}

fn lstm_y_shape(x: &rlx_ir::Shape, hidden_size: usize, bidirectional: bool) -> rlx_ir::Shape {
    let dirs = if bidirectional { 2 } else { 1 };
    if x.rank() == 3 {
        let seq = x.dim(0).unwrap_static();
        let batch = x.dim(1).unwrap_static().max(1);
        return rlx_ir::Shape::new(&[seq, dirs, batch, hidden_size], x.dtype());
    }
    rlx_ir::Shape::new(&[1, dirs, 1, hidden_size], x.dtype())
}

/// `sync_graph_shapes` can collapse `[seq,1,C]` LSTM inputs to `[1,1,C]`; restore seq.
fn fix_import_lstm_x_shape(x: &rlx_ir::Shape) -> rlx_ir::Shape {
    if x.rank() != 3 {
        return x.clone();
    }
    let d0 = x.dim(0).unwrap_static();
    let d1 = x.dim(1).unwrap_static();
    let d2 = x.dim(2).unwrap_static();
    if d0 == 1 && d1 <= 1 && (d2 == 640 || d2 == 512) {
        let seq = std::env::var("RLX_ONNX_SEQUENCE_LENGTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128);
        return rlx_ir::Shape::new(&[seq, d1.max(1), d2], x.dtype());
    }
    x.clone()
}

fn fix_lstm_output_shapes(graph: &mut Graph) {
    use rlx_ir::Op;
    let ids: Vec<NodeId> = graph.nodes().iter().map(|n| n.id).collect();
    for id in ids {
        let node = graph.node(id).clone();
        let Op::Custom { name, attrs, .. } = &node.op else {
            continue;
        };
        if !name.contains("LSTM") {
            continue;
        }
        let hidden_size = if attrs.len() >= 4 {
            u32::from_le_bytes(attrs[0..4].try_into().unwrap()) as usize
        } else {
            256
        };
        let bidirectional = attrs.len() > 4 && attrs[4] != 0;
        let x_id = node.inputs[0];
        let x = fix_import_lstm_x_shape(&graph.node(x_id).shape);
        graph.node_mut(x_id).shape = x.clone();
        graph.node_mut(id).shape = lstm_y_shape(&x, hidden_size, bidirectional);
    }
}

/// `sync_graph_shapes` can collapse `[1, seq, C]` activations to `[1, 1, C]`
/// when seq>1; restore from `RLX_ONNX_SEQUENCE_LENGTH` and propagate once.
///
/// Only runs when `RLX_ONNX_SEQUENCE_LENGTH` is set explicitly — decode graphs such as
/// Qwen3 talker use legitimate `[1, 1, H]` hidden states and must not be expanded.
fn fix_import_sequence_axis(graph: &mut Graph) {
    let Ok(seq_str) = std::env::var("RLX_ONNX_SEQUENCE_LENGTH") else {
        return;
    };
    let seq: usize = match seq_str.parse() {
        Ok(s) if s > 1 => s,
        _ => return,
    };
    for id in graph.nodes().iter().map(|n| n.id).collect::<Vec<_>>() {
        let node = graph.node(id);
        if node.shape.rank() != 3 {
            continue;
        }
        let dims: Vec<_> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        if dims[0] == 1 && dims[1] == 1 && dims[2] >= 64 {
            graph.node_mut(id).shape = rlx_ir::Shape::new(&[1, seq, dims[2]], node.shape.dtype());
        }
    }
    for id in graph.topo_order().collect::<Vec<_>>() {
        let node = graph.node(id).clone();
        if let Some(shape) = rlx_ir::infer_shape::infer_output_shape(graph, &node) {
            graph.node_mut(id).shape = shape;
        }
    }
}

impl CompilePipeline {
    pub fn new(target: FusionTarget) -> Self {
        let mut opts = match target {
            FusionTarget::Cpu => FusionOptions::for_cpu(),
            FusionTarget::Metal => FusionOptions::for_metal(),
            FusionTarget::Wgpu => FusionOptions::for_wgpu(),
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
        let mut mir = hir.lower_to_mir()?;
        rlx_ir::dynamic::sync_graph_shapes(mir.as_graph_mut());
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

    pub fn with_backend_label(mut self, label: &'static str) -> Self {
        self.backend_label = Some(label);
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
        if let Some(label) = self.backend_label {
            return label;
        }
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
        let need_fusion_diff = self.assert_fusion_clean || rlx_ir::env::flag("RLX_FUSION_REPORT");
        let before = need_fusion_diff.then(|| mir.as_graph().clone());
        let passes =
            fusion_passes_for_supported(self.effective_supported(), self.opts, self.target);
        let limits = self.opts.fusion_limits;
        let graph = with_fusion_target(self.target, || {
            with_fusion_limits(limits, || run_passes(mir.into_graph(), &passes, false))
        });
        let graph = clip_elementwise_regions(graph, limits);
        debug_assert_graph!(&graph, "fusion");
        // Downstream-registered fusion / rewrite passes (empty by default). Run
        // after the built-in fusion pipeline so core invariants hold, but before
        // legalize so their output is still lowered/legalized for the backend.
        let graph = rlx_fusion::pass::run_registered_ir_passes(graph);
        let mut graph = self.legalize_after_fusion(graph);
        rlx_ir::dynamic::sync_graph_shapes(&mut graph);
        fix_import_sequence_axis(&mut graph);
        fix_lstm_output_shapes(&mut graph);
        debug_assert_graph!(&graph, "legalize");
        let mir = MirModule::from_graph(graph);
        // Static NaN-source lint (opt-in): report provable compile-time
        // non-finite values with provenance before the graph ever runs.
        if rlx_ir::env::flag("RLX_LINT_NUMERICS") {
            for lint in crate::numeric_lint::lint_numerics(mir.as_graph()) {
                eprintln!("rlx numeric-lint: {lint}");
            }
        }
        let fusion = if let Some(ref before) = before {
            rlx_fusion::FusionReport::analyze(before, mir.as_graph())
        } else {
            rlx_fusion::FusionReport::scan(mir.as_graph())
        };
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
        let graph = mir.as_graph();
        let plan = memory::plan_memory_with_options(graph, self.arena_alignment, opts);
        let buffers = lir_buffer_plan_from_memory(graph, &plan, self.arena_alignment);
        LirModule::new(mir, buffers)
    }

    /// Bind symbolic dims and re-run buffer planning on specialized MIR.
    pub fn specialize_lir(&self, lir: &LirModule, binding: &rlx_ir::DimBinding) -> LirModule {
        use rlx_ir::dynamic::{
            bind_graph, sync_concat_shapes, sync_expand_ops, sync_graph_shapes, sync_narrow_ops,
            sync_reshape_ops,
        };
        let mut bound = bind_graph(lir.as_graph(), binding);
        sync_reshape_ops(&mut bound);
        sync_concat_shapes(&mut bound);
        sync_narrow_ops(&mut bound);
        sync_expand_ops(&mut bound);
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
        let dbg = rlx_ir::env::var("RLX_PHASE_TIMING").is_some();
        #[cfg(not(target_arch = "wasm32"))]
        let t = if dbg {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mir = Self::lower_hir(hir)?;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(t) = t {
            eprintln!("[phase]   lower_hir = {}ms", t.elapsed().as_millis());
        }
        #[cfg(not(target_arch = "wasm32"))]
        let t = if dbg {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let (mir, fusion) = self.optimize_with_report(mir);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(t) = t {
            eprintln!("[phase]   optimize = {}ms", t.elapsed().as_millis());
        }
        #[cfg(not(target_arch = "wasm32"))]
        let t = if dbg {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let r = self.finish(mir, fusion);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(t) = t {
            eprintln!("[phase]   finish(plan+lir) = {}ms", t.elapsed().as_millis());
        }
        Ok(r)
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
    fn numeric_lint_catches_const_div_by_zero_through_pipeline() {
        // A constant `1.0 / 0.0` must be reported by the static lint whether
        // const-folding bakes it into an inf Constant or leaves the Div in
        // place — both are provable NaN sources with provenance.
        let mut g = Graph::new("bad");
        let one = g.add_node(
            Op::Constant {
                data: 1.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            f32_shape(&[1]),
        );
        let zero = g.add_node(
            Op::Constant {
                data: 0.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            f32_shape(&[1]),
        );
        let d = g.binary(rlx_ir::op::BinaryOp::Div, one, zero, f32_shape(&[1]));
        g.set_outputs(vec![d]);

        let result = CompilePipeline::new(FusionTarget::Cpu).compile_graph(g);
        let lints = crate::numeric_lint::lint_numerics(result.lir.mir.as_graph());
        assert!(
            !lints.is_empty(),
            "compiled graph should surface the div-by-zero as a numeric lint"
        );
    }

    #[test]
    fn decode_hidden_shape_not_expanded_without_env() {
        // Qwen3 talker decode uses [1, 1, H] hidden states; must not be expanded to
        // [1, RLX_ONNX_SEQUENCE_LENGTH, H] unless that env is set explicitly.
        let mut g = Graph::new("decode_out");
        let x = g.input("x", f32_shape(&[1, 1, 1024]));
        g.set_outputs(vec![x]);
        let pipe = CompilePipeline::new(FusionTarget::Cpu);
        let result = pipe.compile_graph(g);
        let out = result
            .lir
            .mir
            .as_graph()
            .node(result.lir.mir.as_graph().outputs[0]);
        assert_eq!(out.shape.dims()[1].unwrap_static(), 1);
        assert_eq!(out.shape.num_elements(), Some(1024));
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
