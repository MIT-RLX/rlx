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

//! [`GraphModule`] — unified higher-order DX over HIR / MIR / LIR.

use std::ops::{Deref, DerefMut};

use crate::hir::{FusionPolicy, HirModule, HirNodeId, LowerError};
use crate::inspect::{inspect_hir, inspect_lir, inspect_mir};
use crate::lir::LirModule;
use crate::mir::MirModule;
use crate::op::Activation;
use crate::op::MaskKind;
use crate::quant::QuantScheme;
use crate::{Graph, NodeId, Op, Shape};

/// Which stage of the HIR → MIR → LIR pipeline a [`GraphModule`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphStage {
    Hir,
    Mir,
    Lir,
}

#[derive(Debug, Clone)]
enum Stage {
    Hir(HirModule),
    Mir(MirModule),
    Lir(LirModule),
}

/// Unified model module — primary builder surface above HIR/MIR/LIR.
#[derive(Debug, Clone)]
pub struct GraphModule {
    stage: Stage,
}

impl GraphModule {
    pub fn define(
        name: impl Into<String>,
        build: impl FnOnce(&mut HirModule) -> HirNodeId,
    ) -> Self {
        let mut hir = HirModule::new(name);
        let out = build(&mut hir);
        hir.set_outputs(vec![out]);
        Self {
            stage: Stage::Hir(hir),
        }
    }

    /// Start an empty HIR-stage module (like [`Graph::new`] for MIR).
    pub fn hir(name: impl Into<String>) -> Self {
        Self {
            stage: Stage::Hir(HirModule::new(name)),
        }
    }

    /// Start an empty MIR-stage module.
    pub fn mir(name: impl Into<String>) -> Self {
        Self {
            stage: Stage::Mir(MirModule::new(name)),
        }
    }

    pub fn from_hir(hir: HirModule) -> Self {
        Self {
            stage: Stage::Hir(hir),
        }
    }

    pub fn from_graph(graph: Graph) -> Self {
        Self {
            stage: Stage::Mir(MirModule::from_graph(graph)),
        }
    }

    pub fn from_mir(mir: MirModule) -> Self {
        Self {
            stage: Stage::Mir(mir),
        }
    }

    pub fn from_lir(lir: LirModule) -> Self {
        Self {
            stage: Stage::Lir(lir),
        }
    }

    pub fn block(
        hir: &mut HirModule,
        name: impl Into<String>,
        build: impl FnOnce(&mut HirModule) -> HirNodeId,
    ) -> HirNodeId {
        hir.named(name, build)
    }

    pub fn fusion_policy(&self) -> Option<FusionPolicy> {
        self.as_hir().map(|h| h.fusion_policy)
    }

    pub fn with_fusion_policy(mut self, policy: FusionPolicy) -> Self {
        if let Stage::Hir(h) = &mut self.stage {
            h.fusion_policy = policy;
        } else {
            panic!("GraphModule::with_fusion_policy requires HIR stage");
        }
        self
    }

    /// Set graph outputs at the current stage.
    ///
    /// At HIR stage accepts [`HirNodeId`] values; at MIR/LIR the same
    /// indices map to [`NodeId`] (both are insertion-order node ids).
    pub fn set_outputs(&mut self, outputs: Vec<HirNodeId>) {
        match &mut self.stage {
            Stage::Hir(h) => h.set_outputs(outputs),
            Stage::Mir(m) => m.set_outputs(outputs.into_iter().map(|h| NodeId(h.0)).collect()),
            Stage::Lir(l) => l
                .mir
                .set_outputs(outputs.into_iter().map(|h| NodeId(h.0)).collect()),
        }
    }

    pub fn set_hir_outputs(&mut self, outputs: Vec<HirNodeId>) {
        self.set_outputs(outputs);
    }

    /// Finish HIR construction and set the module output.
    pub fn finish_hir(mut self, output: HirNodeId) -> Self {
        self.set_hir_outputs(vec![output]);
        self
    }

    fn hir_mut(&mut self) -> &mut HirModule {
        self.as_hir_mut()
            .expect("GraphModule: HIR builder methods require HIR stage — use GraphModule::hir() or Graph::define()")
    }

    // ── HIR block builders (forward to HirModule) ─────────────────

    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> HirNodeId {
        match &mut self.stage {
            Stage::Hir(h) => h.input(name, shape),
            Stage::Mir(m) => {
                let id = m.as_graph_mut().input(name, shape);
                HirNodeId(id.0)
            }
            Stage::Lir(l) => {
                let id = l.mir.as_graph_mut().input(name, shape);
                HirNodeId(id.0)
            }
        }
    }

    pub fn param(&mut self, name: impl Into<String>, shape: Shape) -> HirNodeId {
        match &mut self.stage {
            Stage::Hir(h) => h.param(name, shape),
            Stage::Mir(m) => {
                let id = m.as_graph_mut().param(name, shape);
                HirNodeId(id.0)
            }
            Stage::Lir(l) => {
                let id = l.mir.as_graph_mut().param(name, shape);
                HirNodeId(id.0)
            }
        }
    }

    pub fn linear(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        bias: Option<HirNodeId>,
        activation: Option<Activation>,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut()
            .linear(x, weight, bias, activation, out_shape)
    }

    pub fn linear_fused(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        bias: HirNodeId,
        activation: Option<Activation>,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut()
            .linear_fused(x, weight, bias, activation, out_shape)
    }

    pub fn shared_linear_pair(
        &mut self,
        x: HirNodeId,
        w_first: HirNodeId,
        w_second: HirNodeId,
        out_shape: Shape,
    ) -> (HirNodeId, HirNodeId) {
        self.hir_mut()
            .shared_linear_pair(x, w_first, w_second, out_shape)
    }

    pub fn swiglu_ffn(
        &mut self,
        x: HirNodeId,
        up_w: HirNodeId,
        gate_w: HirNodeId,
        down_w: HirNodeId,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut()
            .swiglu_ffn(x, up_w, gate_w, down_w, out_shape)
    }

    pub fn residual_rms_norm(
        &mut self,
        x: HirNodeId,
        residual: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        eps: f32,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut()
            .residual_rms_norm(x, residual, gamma, beta, eps, out_shape)
    }

    pub fn attention(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: Option<HirNodeId>,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut()
            .attention(q, k, v, mask, num_heads, head_dim, mask_kind, out_shape)
    }

    pub fn depthwise_conv1d_causal(
        &mut self,
        input: HirNodeId,
        weight: HirNodeId,
        left_pad: HirNodeId,
        kernel_size: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut()
            .depthwise_conv1d_causal(input, weight, left_pad, kernel_size, out_shape)
    }

    pub fn dequant_matmul(
        &mut self,
        x: HirNodeId,
        w: HirNodeId,
        scale: Option<HirNodeId>,
        zp: Option<HirNodeId>,
        scheme: QuantScheme,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut()
            .dequant_matmul(x, w, scale, zp, scheme, out_shape)
    }

    pub fn gated_delta_net(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state_size: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut()
            .gated_delta_net(q, k, v, g, beta, state_size, out_shape)
    }

    pub fn gated_delta_net_carry(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state: HirNodeId,
        state_size: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut()
            .gated_delta_net_carry(q, k, v, g, beta, state, state_size, out_shape)
    }

    pub fn rope(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        n_rot: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut().rope(x, cos, sin, head_dim, n_rot, out_shape)
    }

    pub fn rms_norm(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        eps: f32,
        out_shape: Shape,
    ) -> HirNodeId {
        self.hir_mut().rms_norm(x, gamma, beta, eps, out_shape)
    }

    pub fn hir_mir(&mut self, op: Op, inputs: Vec<HirNodeId>, shape: Shape) -> HirNodeId {
        self.hir_mut().mir(op, inputs, shape)
    }

    pub fn named(
        &mut self,
        name: impl Into<String>,
        build: impl FnOnce(&mut HirModule) -> HirNodeId,
    ) -> HirNodeId {
        self.hir_mut().named(name, build)
    }

    pub fn stage(&self) -> GraphStage {
        match &self.stage {
            Stage::Hir(_) => GraphStage::Hir,
            Stage::Mir(_) => GraphStage::Mir,
            Stage::Lir(_) => GraphStage::Lir,
        }
    }

    pub fn name(&self) -> &str {
        match &self.stage {
            Stage::Hir(h) => &h.name,
            Stage::Mir(m) => m.name(),
            Stage::Lir(l) => l.name(),
        }
    }

    pub fn lower(self) -> Result<Self, LowerError> {
        match self.stage {
            Stage::Hir(hir) => Ok(Self {
                stage: Stage::Mir(hir.lower_to_mir()?),
            }),
            other => Ok(Self { stage: other }),
        }
    }

    pub fn into_hir(self) -> Option<HirModule> {
        match self.stage {
            Stage::Hir(h) => Some(h),
            _ => None,
        }
    }

    pub fn into_mir(self) -> Result<MirModule, LowerError> {
        match self.stage {
            Stage::Hir(hir) => hir.lower_to_mir(),
            Stage::Mir(m) => Ok(m),
            Stage::Lir(l) => Ok(l.mir),
        }
    }

    pub fn into_lir(self) -> Option<LirModule> {
        match self.stage {
            Stage::Lir(l) => Some(l),
            _ => None,
        }
    }

    pub fn into_graph(self) -> Result<Graph, LowerError> {
        Ok(self.into_mir()?.into_graph())
    }

    pub fn as_hir(&self) -> Option<&HirModule> {
        match &self.stage {
            Stage::Hir(h) => Some(h),
            _ => None,
        }
    }

    pub fn as_hir_mut(&mut self) -> Option<&mut HirModule> {
        match &mut self.stage {
            Stage::Hir(h) => Some(h),
            _ => None,
        }
    }

    pub fn as_mir(&self) -> Option<&MirModule> {
        match &self.stage {
            Stage::Mir(m) => Some(m),
            Stage::Lir(l) => Some(&l.mir),
            _ => None,
        }
    }

    pub fn as_lir(&self) -> Option<&LirModule> {
        match &self.stage {
            Stage::Lir(l) => Some(l),
            _ => None,
        }
    }

    pub fn as_graph(&self) -> Option<&Graph> {
        match &self.stage {
            Stage::Mir(m) => Some(m.as_graph()),
            Stage::Lir(l) => Some(l.as_graph()),
            Stage::Hir(_) => None,
        }
    }

    pub fn inspect(&self) -> String {
        match &self.stage {
            Stage::Hir(h) => inspect_hir(h),
            Stage::Mir(m) => inspect_mir(m),
            Stage::Lir(l) => inspect_lir(l),
        }
    }
}

impl Deref for GraphModule {
    type Target = Graph;

    fn deref(&self) -> &Graph {
        self.as_graph()
            .expect("GraphModule: HIR stage — call lower() before accessing MIR Graph")
    }
}

impl DerefMut for GraphModule {
    fn deref_mut(&mut self) -> &mut Graph {
        match &mut self.stage {
            Stage::Mir(m) => m.as_graph_mut(),
            Stage::Lir(l) => l.mir.as_graph_mut(),
            Stage::Hir(_) => panic!("GraphModule: HIR stage — use as_hir_mut() or lower() first"),
        }
    }
}

impl From<Graph> for GraphModule {
    fn from(graph: Graph) -> Self {
        Self::from_graph(graph)
    }
}

impl TryFrom<GraphModule> for Graph {
    type Error = LowerError;

    fn try_from(module: GraphModule) -> Result<Self, LowerError> {
        module.into_graph()
    }
}

impl From<MirModule> for GraphModule {
    fn from(mir: MirModule) -> Self {
        Self::from_mir(mir)
    }
}

impl From<HirModule> for GraphModule {
    fn from(hir: HirModule) -> Self {
        Self::from_hir(hir)
    }
}

impl std::fmt::Display for GraphModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.stage {
            Stage::Hir(h) => write!(f, "{h}"),
            Stage::Mir(m) => write!(f, "{m}"),
            Stage::Lir(l) => write!(f, "lir @{}", l.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::Graph;
    use crate::Shape;

    fn f32_shape(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn define_lowers_to_mir_graph() {
        let module = GraphModule::define("m", |m| {
            let x = m.input("x", f32_shape(&[2, 8]));
            let w = m.param("w", f32_shape(&[8, 8]));
            m.linear(x, w, None, None, f32_shape(&[2, 8]))
        });
        assert_eq!(module.stage(), GraphStage::Hir);
        let module = module.lower().expect("lower");
        assert_eq!(module.stage(), GraphStage::Mir);
        assert!(module.len() >= 3);
    }

    #[test]
    fn mir_module_deref_builds_graph() {
        let mut module = GraphModule::mir("raw");
        let x = module.input("x", f32_shape(&[4]));
        module.set_outputs(vec![x]);
        assert_eq!(module.len(), 1);
    }

    #[test]
    fn hir_module_block_builders_via_graph_module() {
        use crate::quant::QuantScheme;

        let mut module = GraphModule::hir("layer");
        let x = module.input("x", f32_shape(&[2, 128]));
        let w = module.param("w", f32_shape(&[128, 128]));
        let y = module.dequant_matmul(x, w, None, None, QuantScheme::GgufQ4K, f32_shape(&[2, 128]));
        module.set_outputs(vec![y]);
        assert_eq!(module.stage(), GraphStage::Hir);

        let module = module.lower().expect("lower");
        assert_eq!(module.stage(), GraphStage::Mir);
        assert!(module.len() >= 3);
    }

    #[test]
    fn graph_hir_entry_matches_define() {
        let via_graph = Graph::hir("m");
        let via_define = Graph::define("m", |m| {
            let x = m.input("x", f32_shape(&[4]));
            m.rms_norm(x, x, x, 1e-5, f32_shape(&[4]))
        });
        assert_eq!(via_graph.stage(), GraphStage::Hir);
        assert_eq!(via_define.stage(), GraphStage::Hir);
    }
}
