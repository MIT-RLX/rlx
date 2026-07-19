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

//! ModelFlow builder and built output.

use std::collections::HashMap;

use anyhow::Result;
use rlx_ir::hir::HirModule;
use rlx_ir::{Graph, GraphModule, GraphStage, HirNodeId, Shape, hir_to_graph};

use crate::context::{FlowCtx, FlowState};
use crate::execution::ModelExecutionConfig;
use crate::extension::FlowExtensionPlan;
use crate::profile::CompileProfile;
use crate::stage::FlowStage;
use crate::value::FlowValue;
use crate::weight::WeightSource;

/// Block assembly-line builder — tier-0 model author surface.
#[derive(Debug)]
pub struct ModelFlow {
    name: String,
    pub(crate) profile: CompileProfile,
    /// Graph inputs declared before block stages (do not participate in tensor flow).
    inputs: Vec<(String, Shape)>,
    pub(crate) stages: Vec<FlowStage>,
    output_names: Vec<String>,
    extra_outputs: Vec<HirNodeId>,
    extension_plan: FlowExtensionPlan,
}

impl ModelFlow {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            profile: CompileProfile::default(),
            inputs: Vec::new(),
            stages: Vec::new(),
            output_names: vec!["output".into()],
            extra_outputs: Vec::new(),
            extension_plan: FlowExtensionPlan::default(),
        }
    }

    /// HIR extensions to apply after assemble, before compile (retroactive plugins).
    pub fn with_extensions(mut self, plan: FlowExtensionPlan) -> Self {
        self.extension_plan = plan;
        self
    }

    /// Declare a graph input. The first input starts the tensor flow; later
    /// inputs are side declarations only (e.g. `last_token_idx`).
    pub fn input(mut self, name: impl Into<String>, shape: Shape) -> Self {
        self.inputs.push((name.into(), shape));
        self
    }

    pub fn with_profile(mut self, profile: CompileProfile) -> Self {
        self.profile = profile;
        self
    }

    pub fn profile(&self) -> &CompileProfile {
        &self.profile
    }

    pub fn stage(mut self, stage: FlowStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Append a downstream-defined [`LayerStage`](crate::LayerStage) block.
    ///
    /// The **extension front door**: a model crate outside `rlx-flow` implements
    /// `LayerStage` for its own architecture block and drops it straight into a
    /// flow — no [`FlowStage`] enum variant, no core edit. The block composes
    /// primitives through [`FlowCtx`](crate::FlowCtx)'s emission surface, so it
    /// fuses and hits the fast path on every backend.
    ///
    /// (Distinct from [`layer`](Self::layer), which builds a named
    /// [`LayerStack`](crate::LayerStack) from a closure of built-in blocks.)
    pub fn layer_stage(mut self, stage: impl crate::LayerStage + 'static) -> Self {
        self.stages.push(FlowStage::dynamic(stage));
        self
    }

    pub fn output(mut self, name: impl Into<String>) -> Self {
        self.output_names = vec![name.into()];
        self
    }

    pub fn outputs(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.output_names = names.into_iter().map(Into::into).collect();
        self
    }

    /// Append side outputs (e.g. per-layer KV taps) after the primary output.
    pub fn with_extra_outputs(mut self, ids: Vec<HirNodeId>) -> Self {
        self.extra_outputs = ids;
        self
    }

    /// Build from a composable recipe, then optionally patch before compile.
    pub fn from_recipe(recipe: &impl crate::recipe::ModelRecipe) -> Self {
        recipe.assemble()
    }

    pub fn build(self, weights: &mut dyn WeightSource) -> Result<BuiltModel> {
        let mut module =
            GraphModule::hir(&self.name).with_fusion_policy(self.profile.fusion_policy());
        let mut params = HashMap::new();
        let mut state = FlowState::default();
        let mut ctx = FlowCtx {
            module,
            params: &mut params,
            weights,
            profile: &self.profile,
            state: &mut state,
        };

        let mut value: Option<FlowValue> = None;
        for (i, (name, shape)) in self.inputs.iter().enumerate() {
            let id = ctx.input(name, shape.clone());
            ctx.state.inputs.insert(name.clone(), (id, shape.clone()));
            if i == 0 {
                value = Some(ctx.wrap(id, shape.clone()));
            }
        }
        for stage in &self.stages {
            value = stage.emit(&mut ctx, value)?;
        }

        let primary = value.ok_or_else(|| anyhow::anyhow!("ModelFlow produced no output"))?;
        let mut outputs = vec![primary.id];
        outputs.extend(self.extra_outputs);
        // Auxiliary outputs published by stages via `ctx.publish_side_output`
        // (KV taps, aux heads) — appended after the primary + explicit extras so
        // existing output indices are preserved.
        let side_names: Vec<String> = ctx
            .state
            .side_outputs
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        outputs.extend(ctx.state.side_outputs.iter().map(|(_, id)| *id));

        ctx.module.set_outputs(outputs);
        module = ctx.module;
        if let Some(hir) = module.as_hir_mut() {
            self.extension_plan.apply(hir);
        }

        let mut output_names = self.output_names;
        output_names.extend(side_names);

        Ok(BuiltModel {
            module,
            params,
            typed_params: Vec::new(),
            profile: self.profile,
            output_names,
            primary_shape: primary.shape,
        })
    }

    /// Compatibility shim: older callers passed GGUF packed matmul params.
    ///
    /// The current flow builder ignores packed params; packed lowering lives in model crates.
    pub fn build_with(
        self,
        weights: &mut dyn WeightSource,
        _gguf_packed: Option<&crate::GgufPackedParams>,
    ) -> Result<BuiltModel> {
        self.build(weights)
    }
}

/// Result of assembling a model flow.
#[derive(Debug, Clone)]
pub struct BuiltModel {
    pub module: GraphModule,
    pub params: HashMap<String, Vec<f32>>,
    /// Packed U8 params (GGUF quant blobs) attached after compile via `set_param_typed`.
    pub typed_params: Vec<(String, Vec<u8>, rlx_ir::DType)>,
    pub profile: CompileProfile,
    output_names: Vec<String>,
    primary_shape: Shape,
}

impl BuiltModel {
    /// Attach variant + execution preset (shader-component bundle).
    pub fn with_execution_config(mut self, config: &ModelExecutionConfig) -> Self {
        self.profile = config.compile_profile();
        self
    }

    /// Wrap a legacy HIR builder product as tier-0 flow output (migration bridge).
    pub fn from_hir(hir: HirModule, params: HashMap<String, Vec<f32>>) -> anyhow::Result<Self> {
        let primary = hir
            .outputs
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("from_hir: module has no outputs"))?;
        let primary_shape = hir.node(primary).shape.clone();
        Ok(Self {
            module: GraphModule::from_hir(hir),
            params,
            typed_params: Vec::new(),
            profile: CompileProfile::default(),
            output_names: vec!["output".into()],
            primary_shape,
        })
    }

    /// Wrap a legacy MIR graph builder product as tier-0 flow output (migration bridge).
    pub fn from_graph(graph: Graph, params: HashMap<String, Vec<f32>>) -> anyhow::Result<Self> {
        let primary = graph
            .outputs
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("from_graph: graph has no outputs"))?;
        let primary_shape = graph.node(primary).shape.clone();
        Ok(Self {
            module: GraphModule::from_graph(graph),
            params,
            typed_params: Vec::new(),
            profile: CompileProfile::default(),
            output_names: vec!["output".into()],
            primary_shape,
        })
    }

    pub fn profile(&self) -> &CompileProfile {
        &self.profile
    }

    pub fn params(&self) -> &HashMap<String, Vec<f32>> {
        &self.params
    }

    pub fn primary_shape(&self) -> &Shape {
        &self.primary_shape
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// `(Graph, params)` for legacy compile paths. Moves `params` (no clone).
    pub fn into_graph_parts(mut self) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
        let params = std::mem::take(&mut self.params);
        let graph = self.into_graph()?;
        Ok((graph, params))
    }

    pub fn into_graph_module(self) -> GraphModule {
        self.module
    }

    pub fn into_hir(self) -> Option<HirModule> {
        self.module.into_hir()
    }

    pub fn into_graph(self) -> Result<Graph> {
        if self.module.stage() == GraphStage::Hir {
            let hir = self
                .module
                .into_hir()
                .ok_or_else(|| anyhow::anyhow!("expected HIR stage"))?;
            hir_to_graph(hir).map_err(Into::into)
        } else {
            self.module.into_graph().map_err(Into::into)
        }
    }

    pub fn lower(self) -> Result<GraphModule> {
        self.module.lower().map_err(Into::into)
    }

    /// Append side outputs after the primary output node.
    pub fn with_extra_hir_outputs(mut self, extra: impl IntoIterator<Item = HirNodeId>) -> Self {
        let primary = self.module.as_hir().expect("HIR stage").outputs[0];
        let mut outputs = vec![primary];
        outputs.extend(extra);
        self.module.set_outputs(outputs);
        self
    }

    /// Split into HIR module + param map (common compile path).
    pub fn into_parts(mut self) -> Result<(HirModule, HashMap<String, Vec<f32>>)> {
        // Move the param map out instead of cloning it — `into_hir` consumes
        // `self` but only reads `self.module`, so the original would just be
        // dropped. For a multi-GB weight map this avoids a full duplicate on
        // every compile.
        let params = std::mem::take(&mut self.params);
        let hir = self
            .into_hir()
            .ok_or_else(|| anyhow::anyhow!("expected HIR stage"))?;
        Ok((hir, params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::LayerStack;
    use crate::weight::MapWeights;
    use rlx_ir::{DType, Shape};

    #[test]
    fn minimal_embed_flow() {
        let mut w = MapWeights::default();
        w.insert("embed.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        let flow = ModelFlow::new("smoke")
            .input("ids", Shape::new(&[1, 2], DType::F32))
            .embed("embed.weight");

        let built = flow.build(&mut w).unwrap();
        let hir = built.into_hir().unwrap();
        assert!(hir.len() >= 3);
    }

    #[test]
    fn custom_stage_passthrough() {
        let mut w = MapWeights::default();
        w.insert("embed.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        let flow = ModelFlow::new("custom")
            .input("ids", Shape::new(&[1, 2], DType::F32))
            .embed("embed.weight")
            .custom(|_emit, input| Ok(input));

        let built = flow.build(&mut w).unwrap();
        assert_eq!(built.primary_shape().rank(), 3);
    }

    #[test]
    fn layer_stack_builds_sequence() {
        let mut w = MapWeights::default();
        w.insert("ln.weight", vec![1.0; 4], vec![4]);

        let stage = LayerStack::named("block")
            .rms_norm("ln.weight", 1e-5)
            .build();

        let flow = ModelFlow::new("stack")
            .input("x", Shape::new(&[1, 2, 4], DType::F32))
            .zero_beta(4)
            .raw_stage(stage);

        let built = flow.build(&mut w).unwrap();
        assert!(built.into_hir().unwrap().len() >= 4);
    }

    #[test]
    fn spdnet_stack_composes_and_lowers() {
        use rlx_ir::Op;
        use rlx_ir::hir::HirOp;

        // BiMap weight + SPD batch-norm params are F64 and uploaded out of band
        // (declare_param_f64), so MapWeights need not carry them.
        let mut w = MapWeights::default();

        let stage = LayerStack::named("spd_block")
            .bimap("bimap.w", 3)
            .reeig(1e-5)
            .logeig(1e-5)
            .build();

        // SPD input `[4, 4]` → BiMap → `[3, 3]` → ReEig/LogEig (shape-preserving).
        let flow = ModelFlow::new("spdnet")
            .input("x", Shape::new(&[4, 4], DType::F64))
            .raw_stage(stage);

        let built = flow.build(&mut w).unwrap();
        // Output is the LogEig `[3, 3]` block, F64.
        let ps = built.primary_shape();
        assert_eq!(ps.rank(), 2);
        assert_eq!(ps.dim(0).unwrap_static(), 3);
        assert_eq!(ps.dim(1).unwrap_static(), 3);
        assert_eq!(ps.dtype(), DType::F64);

        let hir = built.into_hir().unwrap();
        let has = |want: &Op| {
            hir.nodes().iter().any(|n| match &n.op {
                HirOp::Mir(op) => std::mem::discriminant(op) == std::mem::discriminant(want),
                _ => false,
            })
        };
        assert!(has(&Op::BiMap), "expected a BiMap node");
        assert!(has(&Op::ReEig { eps: 0.0 }), "expected a ReEig node");
        assert!(has(&Op::LogEig { eps: 0.0 }), "expected a LogEig node");
        // spectral_layer packs then narrows + reshapes the `Y` block.
        assert!(
            has(&Op::Narrow {
                axis: 0,
                start: 0,
                len: 0
            }),
            "expected a Narrow node from spectral_layer"
        );
        // Every F64 SPD op node must stay F64 (CPU kernels expect_f64).
        for n in hir.nodes() {
            if let HirOp::Mir(Op::BiMap | Op::ReEig { .. } | Op::LogEig { .. }) = &n.op {
                assert_eq!(n.shape.dtype(), DType::F64);
            }
        }
    }

    #[test]
    fn spd_batch_norm_stage_declares_f64_params_and_lowers() {
        use rlx_ir::Op;
        use rlx_ir::hir::HirOp;

        let mut w = MapWeights::default();
        let stage = LayerStack::named("spd_bn")
            .spd_batch_norm("bn.g", "bn.running_mean", 1e-5)
            .build();

        // `[batch, n, n]` = `[2, 3, 3]`; output matches the input.
        let flow = ModelFlow::new("spdbn")
            .input("x", Shape::new(&[2, 3, 3], DType::F64))
            .raw_stage(stage);

        let built = flow.build(&mut w).unwrap();
        let ps = built.primary_shape();
        assert_eq!(ps.rank(), 3);
        assert_eq!(ps.dim(0).unwrap_static(), 2);
        assert_eq!(ps.dim(2).unwrap_static(), 3);
        assert_eq!(ps.dtype(), DType::F64);

        let hir = built.into_hir().unwrap();
        // Two F64 Param nodes (G + running mean) shaped `[3, 3]`.
        let f64_params = hir
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, HirOp::Param { .. }) && n.shape.dtype() == DType::F64)
            .count();
        assert_eq!(f64_params, 2, "expected G + running-mean F64 params");
        assert!(
            hir.nodes()
                .iter()
                .any(|n| matches!(&n.op, HirOp::Mir(Op::SpdBatchNorm { .. }))),
            "expected a SpdBatchNorm node"
        );
    }

    #[test]
    fn when_conditional_embed() {
        let mut w = MapWeights::default();
        w.insert("embed.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        let with_embed = ModelFlow::new("cond")
            .input("ids", Shape::new(&[1, 2], DType::F32))
            .when(true, |f| f.embed("embed.weight"))
            .build(&mut w)
            .unwrap();
        assert!(with_embed.into_hir().unwrap().len() >= 3);

        let mut w2 = MapWeights::default();
        w2.insert("embed.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let skipped = ModelFlow::new("cond")
            .input("ids", Shape::new(&[1, 2], DType::F32))
            .when(false, |f| f.embed("embed.weight"))
            .build(&mut w2)
            .unwrap();
        // Skipped embed — graph is input-only passthrough.
        assert_eq!(skipped.into_hir().unwrap().len(), 1);
    }
}
