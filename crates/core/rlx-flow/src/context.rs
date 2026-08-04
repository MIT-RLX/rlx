// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Flow context — internal HIR emission surface (not for model authors).

use std::collections::HashMap;

use anyhow::Result;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::Activation;
use rlx_ir::{DType, GraphModule, HirGraphExt, Shape};

use crate::profile::CompileProfile;
use crate::value::FlowValue;
use crate::weight::WeightSource;

/// Handles for a [`Op::GatedDeltaNet`] / carry scan.
#[derive(Debug, Clone, Copy)]
pub struct GdnInputSlots {
    pub q: HirNodeId,
    pub k: HirNodeId,
    pub v: HirNodeId,
    pub g: HirNodeId,
    pub beta: HirNodeId,
}

/// Cross-stage shared handles (RoPE tables, zero-beta, tied embed, …).
#[derive(Debug, Default)]
pub struct FlowState {
    pub rope_cos: Option<HirNodeId>,
    pub rope_sin: Option<HirNodeId>,
    pub zero_beta: Option<HirNodeId>,
    pub embed_weight: Option<HirNodeId>,
    pub hidden_shape: Option<Shape>,
    pub decode: Option<DecodeBindings>,
    pub residual_skip: Option<HirNodeId>,
    pub residual_shape: Option<Shape>,
    /// Named tensor streams (`img`, `txt`, …) for multi-stream models.
    pub streams: HashMap<String, FlowValue>,
    /// Graph inputs beyond the primary tensor flow (`encoder`, `temb`, …).
    pub inputs: HashMap<String, (HirNodeId, Shape)>,
    /// Named scalar/tensor node refs (RoPE tables, mod params, carry state, …).
    pub named: HashMap<String, HirNodeId>,
    /// Last-published GDN q/k/v/g/beta handles for [`crate::blocks::GdnScanStage`].
    pub gdn: Option<GdnInputSlots>,
    /// Reuse param nodes when multiple stages in one layer load the same key
    /// (e.g. [`crate::blocks::LlamaKvTapStage`] + fused decoder).
    pub loaded_params: HashMap<String, HirNodeId>,
    /// Auxiliary graph outputs a stage published via
    /// [`FlowCtx::publish_side_output`] (KV taps, aux heads). Appended after the
    /// primary output by [`crate::ModelFlow::build`].
    pub side_outputs: Vec<(String, HirNodeId)>,
}

/// KV-cache decode inputs bound by [`crate::blocks::BindDecodeInputsStage`].
#[derive(Debug, Clone)]
pub struct DecodeBindings {
    pub cos: HirNodeId,
    pub sin: HirNodeId,
    pub mask: Option<HirNodeId>,
    pub past_k: Vec<HirNodeId>,
    pub past_v: Vec<HirNodeId>,
}

/// Internal builder context. Blocks emit through this — tier-2 via [`crate::escape::Emit`].
pub struct FlowCtx<'a> {
    pub(crate) module: GraphModule,
    pub(crate) params: &'a mut HashMap<String, Vec<f32>>,
    pub(crate) weights: &'a mut dyn WeightSource,
    pub(crate) profile: &'a CompileProfile,
    pub(crate) state: &'a mut FlowState,
}

impl FlowCtx<'_> {
    pub fn hir(&mut self) -> &mut HirModule {
        self.module
            .as_hir_mut()
            .expect("flow context requires HIR stage")
    }

    pub fn node_shape(&self, id: HirNodeId) -> Result<Shape> {
        Ok(self
            .module
            .as_hir()
            .ok_or_else(|| anyhow::anyhow!("flow context requires HIR stage"))?
            .node(id)
            .shape
            .clone())
    }

    pub fn load_param(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        self.load_param_typed(key, transpose, DType::F32)
    }

    /// Load a param with an explicit graph dtype. The weight bytes come from
    /// the loader as `f32` and are stored in the flow `params` map as `f32`;
    /// when `dtype` is `F16`/`BF16` the backend converts f32→low-precision at
    /// bind time (Metal: `set_param` → `arena.write_from_f32` /
    /// `write_weight_from_f32`), so the matmul RHS ends up truly f16-resident
    /// (2 bytes/elem, half the weight-read bandwidth). Used to store decode
    /// matmul weights as F16 on bandwidth-bound backends.
    pub fn load_param_typed(
        &mut self,
        key: &str,
        transpose: bool,
        dtype: DType,
    ) -> Result<HirNodeId> {
        let cache_key = param_cache_key(key, transpose);
        if let Some(&id) = self.state.loaded_params.get(&cache_key) {
            return Ok(id);
        }
        let (data, shape) = self.weights.take(key, transpose)?;
        let ir_shape = Shape::new(&shape, dtype);
        let id = self.hir().param(key, ir_shape);
        self.params.insert(key.to_string(), data);
        self.state.loaded_params.insert(cache_key, id);
        Ok(id)
    }

    /// Declare an F64 param node without routing data through the F32
    /// auto-upload map. SPD/Riemannian layers (BiMap / SPD batch-norm bias +
    /// running mean) are F64-first — the CPU kernels `expect_f64` and error on
    /// F32. The flow `params` map is `Vec<f32>`-only, so F64 param *bytes* must
    /// be uploaded out of band at session time via `set_param_typed(name,
    /// &bytes, DType::F64)`, exactly like GGUF U8 quant blobs. This method only
    /// creates the correctly-typed graph node; the caller/trainer supplies the
    /// bytes.
    pub fn declare_param_f64(&mut self, key: &str, shape: &[usize]) -> HirNodeId {
        let cache_key = param_cache_key(key, false);
        if let Some(&id) = self.state.loaded_params.get(&cache_key) {
            return id;
        }
        let ir_shape = Shape::new(shape, DType::F64);
        let id = self.hir().param(key, ir_shape);
        self.state.loaded_params.insert(cache_key, id);
        id
    }

    pub fn synth_param(&mut self, name: &str, data: Vec<f32>, shape: Shape) -> HirNodeId {
        let id = self.hir().param(name, shape);
        self.params.insert(name.to_string(), data);
        id
    }

    pub fn synth_zeros(&mut self, name: &str, len: usize) -> HirNodeId {
        self.synth_param(name, vec![0f32; len], Shape::new(&[len], DType::F32))
    }

    pub fn input(&mut self, name: &str, shape: Shape) -> HirNodeId {
        self.hir().input(name, shape)
    }

    pub fn wrap(&self, id: HirNodeId, shape: Shape) -> FlowValue {
        FlowValue::new(id, shape)
    }
}

/// Primitive builders — the composition surface for [`crate::LayerStage`]
/// authors. Each wraps the HIR builder and returns a [`FlowValue`] with its
/// inferred shape, so a downstream block composes ops (`ctx.matmul`,
/// `ctx.rms_norm`, …) without importing `rlx_ir::hir` or touching `HirMut`
/// directly. Because these emit ordinary primitives, the block stays visible to
/// fusion and hits the fast path on every backend.
impl FlowCtx<'_> {
    /// Load a weight as a [`FlowValue`] (param node + its shape).
    pub fn param(&mut self, key: &str, transpose: bool) -> Result<FlowValue> {
        let id = self.load_param(key, transpose)?;
        let shape = self.node_shape(id)?;
        Ok(self.wrap(id, shape))
    }

    /// Matrix multiply `lhs @ rhs`.
    pub fn matmul(&mut self, lhs: &FlowValue, rhs: &FlowValue) -> FlowValue {
        self.binary_shaped(|gb| gb.mm(lhs.id, rhs.id))
    }

    /// Linear projection `input @ W[weight_key]` (loads the weight).
    pub fn linear(
        &mut self,
        input: &FlowValue,
        weight_key: &str,
        transpose: bool,
    ) -> Result<FlowValue> {
        let w = self.load_param(weight_key, transpose)?;
        Ok(self.binary_shaped(|gb| gb.mm(input.id, w)))
    }

    /// Elementwise add (`a + b`, broadcasting).
    pub fn add(&mut self, a: &FlowValue, b: &FlowValue) -> FlowValue {
        self.binary_shaped(|gb| gb.add(a.id, b.id))
    }

    /// Elementwise subtract (`a - b`, broadcasting).
    pub fn sub(&mut self, a: &FlowValue, b: &FlowValue) -> FlowValue {
        self.binary_shaped(|gb| gb.sub(a.id, b.id))
    }

    /// Elementwise multiply (`a * b`, broadcasting).
    pub fn mul(&mut self, a: &FlowValue, b: &FlowValue) -> FlowValue {
        self.binary_shaped(|gb| gb.mul(a.id, b.id))
    }

    /// Residual add — sugar for [`add`](Self::add) reading as `input + skip`.
    pub fn residual(&mut self, input: &FlowValue, skip: &FlowValue) -> FlowValue {
        self.add(input, skip)
    }

    /// Apply an [`Activation`] (shape-preserving).
    pub fn activation(&mut self, act: Activation, input: &FlowValue) -> FlowValue {
        let shape = input.shape.clone();
        let out = shape.clone();
        let id = {
            let mut gb = HirMut::new(self.hir());
            gb.activation(act, input.id, out)
        };
        self.wrap(id, shape)
    }

    /// ReLU (`max(0, x)`).
    pub fn relu(&mut self, input: &FlowValue) -> FlowValue {
        self.activation(Activation::Relu, input)
    }

    /// GELU.
    pub fn gelu(&mut self, input: &FlowValue) -> FlowValue {
        self.activation(Activation::Gelu, input)
    }

    /// SiLU / swish.
    pub fn silu(&mut self, input: &FlowValue) -> FlowValue {
        self.activation(Activation::Silu, input)
    }

    /// RMSNorm over the last axis with weight `gamma_key`. Reuses the flow's
    /// zero-beta slot, auto-provisioning one sized to `gamma` if absent (so a
    /// downstream block needs no explicit `ZeroBeta` stage).
    pub fn rms_norm(&mut self, input: &FlowValue, gamma_key: &str, eps: f32) -> Result<FlowValue> {
        let gamma = self.load_param(gamma_key, false)?;
        let zero_beta = match self.state.zero_beta {
            Some(z) => z,
            None => {
                let gs = self.node_shape(gamma)?;
                let len = gs.dim(gs.rank().saturating_sub(1)).unwrap_static();
                let z = self.synth_zeros("__flow_zero_beta", len);
                self.state.zero_beta = Some(z);
                z
            }
        };
        let shape = input.shape.clone();
        let id = {
            let mut gb = HirMut::new(self.hir());
            gb.rms_norm(input.id, gamma, zero_beta, eps)
        };
        Ok(self.wrap(id, shape))
    }

    /// Cast to a different dtype.
    pub fn cast(&mut self, input: &FlowValue, to: DType) -> FlowValue {
        self.binary_shaped(|gb| gb.cast(input.id, to))
    }

    /// Publish an auxiliary graph output (KV cache tap, aux head). Collected
    /// after the primary output by [`crate::ModelFlow::build`] — the block does
    /// not have to thread it through the return value.
    pub fn publish_side_output(&mut self, name: impl Into<String>, value: &FlowValue) {
        self.state.side_outputs.push((name.into(), value.id));
    }

    /// Emit a node via `build` and wrap it with its inferred shape.
    fn binary_shaped(&mut self, build: impl FnOnce(&mut HirMut<'_>) -> HirNodeId) -> FlowValue {
        let (id, shape) = {
            let mut gb = HirMut::new(self.hir());
            let id = build(&mut gb);
            (id, gb.shape(id).clone())
        };
        self.wrap(id, shape)
    }
}

fn param_cache_key(key: &str, transpose: bool) -> String {
    if transpose {
        format!("{key}\0t")
    } else {
        key.to_string()
    }
}
