// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Flow context — internal HIR emission surface (not for model authors).

use std::collections::HashMap;

use anyhow::Result;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::op::Activation;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, GraphModule, HirGraphExt, Shape};

use crate::GgufPackedLinear;
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
    /// Packed U8 quant blobs registered by [`FlowCtx::linear`] for packed
    /// weights (GGUF/MLX). Drained into [`crate::BuiltModel::typed_params`] at
    /// build end and uploaded at session time via `set_param_typed`.
    pub typed_params: Vec<(String, Vec<u8>, DType)>,
    /// Packed-linear dedup: `weight_key` → (registered U8 param node, out_dim,
    /// scheme). Lets a projection loaded by more than one stage (KV tap +
    /// decoder) reuse the same node without re-`take_packed`-ing a *consuming*
    /// loader — the second load would otherwise see `None` and wrongly fall
    /// back to F32.
    pub packed_linears: HashMap<String, (HirNodeId, usize, QuantScheme)>,
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

/// A linear projection weight resolved for emission *inside* a hand-rolled
/// [`HirMut`] block. Such blocks (e.g. the qwen3 decode/prefill layers) can't
/// call [`FlowCtx::linear`] once a `HirMut` borrows the ctx, so they resolve
/// each projection to a `LinearWeight` up front via [`FlowCtx::resolve_linear`]
/// and then [`LinearWeight::emit`] the matmul into the live `HirMut`.
pub enum LinearWeight {
    /// Dense F32/F16 weight param → plain `mm`.
    Dense(HirNodeId),
    /// Packed U8 quant blob → fused `DequantMatMul` (no F32 materialization).
    Packed {
        wq: HirNodeId,
        out_dim: usize,
        scheme: QuantScheme,
    },
}

impl LinearWeight {
    /// Emit `input @ W` into `gb`, choosing `mm` vs fused `DequantMatMul`. The
    /// packed output shape is `input`'s shape with the last dim set to `out_dim`
    /// (the GGUF blob is `[out_dim, in_dim]` and the op transposes internally).
    pub fn emit(&self, gb: &mut HirMut<'_>, input: HirNodeId) -> HirNodeId {
        match *self {
            LinearWeight::Dense(w) => gb.mm(input, w),
            LinearWeight::Packed {
                wq,
                out_dim,
                scheme,
            } => {
                let out = gb.shape(input).clone();
                let last = out.rank() - 1;
                let out = out.with_dim(last, Dim::Static(out_dim));
                gb.0.dequant_matmul(input, wq, None, None, scheme, out)
            }
        }
    }
}

/// Result of [`FlowCtx::resolve_linear_fused`]. `Fused` = one combined GEMV to
/// emit then `narrow_`-split by `dims`; `Separate` = the projections could not be
/// fused (mixed packed quant schemes) so emit each weight on its own input.
pub enum FusedProj {
    Fused {
        weight: LinearWeight,
        dims: Vec<usize>,
    },
    Separate(Vec<LinearWeight>),
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
    ///
    /// If the [`WeightSource`] hands out a packed (quantized) weight for this
    /// key via [`WeightSource::take_packed`], emits a fused `DequantMatMul` over
    /// the U8 quant blob — same topology, no F32 weight materialization, so
    /// decode runs at `m=1` instead of re-projecting a padded window. Otherwise
    /// falls back to a plain F32 matmul. Gated: F32-only sources (the default
    /// `take_packed` returns `None`) never enter the packed path, so existing
    /// models are byte-for-byte unchanged.
    pub fn linear(
        &mut self,
        input: &FlowValue,
        weight_key: &str,
        transpose: bool,
    ) -> Result<FlowValue> {
        // Reuse an already-registered packed projection (a prior stage loaded
        // it) without re-`take_packed`-ing a consuming loader.
        if let Some(&(wq_id, out_dim, scheme)) = self.state.packed_linears.get(weight_key) {
            return self.emit_dequant(input, weight_key, wq_id, out_dim, scheme);
        }
        if let Some(packed) = self.weights.take_packed(weight_key)? {
            return self.register_packed_linear(input, weight_key, packed);
        }
        let w = self.load_param(weight_key, transpose)?;
        Ok(self.binary_shaped(|gb| gb.mm(input.id, w)))
    }

    /// First load of a packed weight: register the U8 quant blob as a graph
    /// param (bytes attached at session time via `set_param_typed`), remember
    /// it for later stages, then emit the `DequantMatMul`.
    fn register_packed_linear(
        &mut self,
        input: &FlowValue,
        weight_key: &str,
        packed: GgufPackedLinear,
    ) -> Result<FlowValue> {
        let GgufPackedLinear {
            w_q,
            scheme,
            out_dim,
            ..
        } = packed;
        let wq_key = format!("{weight_key}\0q");
        let wq_shape = Shape::new(&[w_q.len()], DType::U8);
        let wq_id = self.hir().param(&wq_key, wq_shape);
        self.state.typed_params.push((wq_key, w_q, DType::U8));
        self.state
            .packed_linears
            .insert(weight_key.to_string(), (wq_id, out_dim, scheme));
        self.emit_dequant(input, weight_key, wq_id, out_dim, scheme)
    }

    /// Emit a fused `DequantMatMul` for a registered packed weight. The output
    /// shape is `input`'s shape with the last dim replaced by `out_dim`; leading
    /// dims — including a dynamic `m` — are preserved, so the same packed graph
    /// runs at any sequence length. The GGUF blob is already `[out_dim, in_dim]`
    /// and the op transposes internally, so `transpose` does not apply here.
    fn emit_dequant(
        &mut self,
        input: &FlowValue,
        weight_key: &str,
        wq_id: HirNodeId,
        out_dim: usize,
        scheme: QuantScheme,
    ) -> Result<FlowValue> {
        if input.shape.rank() == 0 {
            return Err(anyhow::anyhow!(
                "linear: cannot project scalar input for '{weight_key}'"
            ));
        }
        let mut dims: Vec<Dim> = input.shape.dims().to_vec();
        let last = dims.len() - 1;
        dims[last] = Dim::Static(out_dim);
        let out_shape = Shape::from_dims(&dims, input.shape.dtype());
        let out_for_op = out_shape.clone();
        let id = self
            .hir()
            .dequant_matmul(input.id, wq_id, None, None, scheme, out_for_op);
        Ok(self.wrap(id, out_shape))
    }

    /// Resolve a projection weight for a hand-rolled [`HirMut`] block (see
    /// [`LinearWeight`]). Mirrors [`Self::linear`]'s packed dispatch: if the
    /// [`WeightSource`] hands out a quant blob for `key`, register the U8 param
    /// and return [`LinearWeight::Packed`] (emit becomes a fused
    /// `DequantMatMul`); otherwise load a dense F32/F16 param (`dtype`,
    /// `transpose`). F32-only sources (default `take_packed` → `None`) always
    /// take the dense branch, so non-packed builds are byte-for-byte unchanged.
    pub fn resolve_linear(
        &mut self,
        key: &str,
        transpose: bool,
        dtype: DType,
    ) -> Result<LinearWeight> {
        if let Some(packed) = self.weights.take_packed(key)? {
            let GgufPackedLinear {
                w_q,
                scheme,
                out_dim,
                ..
            } = packed;
            let wq_key = format!("{key}\0q");
            let wq = self
                .hir()
                .param(&wq_key, Shape::new(&[w_q.len()], DType::U8));
            self.state.typed_params.push((wq_key, w_q, DType::U8));
            return Ok(LinearWeight::Packed {
                wq,
                out_dim,
                scheme,
            });
        }
        Ok(LinearWeight::Dense(
            self.load_param_typed(key, transpose, dtype)?,
        ))
    }

    /// Fused multi-projection resolve: concatenate several PACKED weights that
    /// share an input dim + quant scheme (e.g. q/k/v projected from the same
    /// hidden) into ONE `DequantMatMul` — one GEMV dispatch instead of N, cutting
    /// per-token kernel launches on the decode path. The GGUF blobs are
    /// `[out_i, in_dim]` row-major, so the fused weight is just their bytes
    /// concatenated (`out = Σ out_i`). Returns the combined [`LinearWeight`] plus
    /// each part's `out_dim` (to `narrow_`-split the result). Returns `None` when
    /// the weights are not packed — `take_packed` on an F32 source yields `None`
    /// WITHOUT consuming, so the caller safely falls back to per-key `resolve_linear`.
    pub fn resolve_linear_fused(
        &mut self,
        keys: &[&str],
        transpose: bool,
        dtype: DType,
    ) -> Result<FusedProj> {
        // `take_packed` on a GGUF source CONSUMES (returns None without consuming
        // for F32/F16 sources). So this method fully owns q/k/v resolution — the
        // caller must NOT also `resolve_linear` these keys, or the loader double-
        // takes. Probe the first key to pick the packed vs dense branch.
        match self.weights.take_packed(keys[0])? {
            Some(first) => {
                let mut parts = vec![first];
                for &key in &keys[1..] {
                    parts.push(self.weights.take_packed(key)?.ok_or_else(|| {
                        anyhow::anyhow!("resolve_linear_fused: mixed packed/dense for {keys:?}")
                    })?);
                }
                let scheme = parts[0].scheme;
                if parts.iter().all(|p| p.scheme == scheme) {
                    // Uniform scheme → byte-concat the [out_i, in] blobs → one GEMV.
                    let mut dims = Vec::with_capacity(keys.len());
                    let mut bytes = Vec::new();
                    for p in &parts {
                        dims.push(p.out_dim);
                        bytes.extend_from_slice(&p.w_q);
                    }
                    let out_dim: usize = dims.iter().sum();
                    let wq_key = format!("{}\0fused", keys[0]);
                    let wq = self
                        .hir()
                        .param(&wq_key, Shape::new(&[bytes.len()], DType::U8));
                    self.state.typed_params.push((wq_key, bytes, DType::U8));
                    Ok(FusedProj::Fused {
                        weight: LinearWeight::Packed {
                            wq,
                            out_dim,
                            scheme,
                        },
                        dims,
                    })
                } else {
                    // Mixed K-quant schemes (Q4_K_M) — can't concat differing block
                    // formats. Register each as its own packed weight (== unfused).
                    let mut ws = Vec::with_capacity(keys.len());
                    for (i, p) in parts.into_iter().enumerate() {
                        let wq_key = format!("{}\0q", keys[i]);
                        let wq = self
                            .hir()
                            .param(&wq_key, Shape::new(&[p.w_q.len()], DType::U8));
                        self.state.typed_params.push((wq_key, p.w_q, DType::U8));
                        ws.push(LinearWeight::Packed {
                            wq,
                            out_dim: p.out_dim,
                            scheme: p.scheme,
                        });
                    }
                    Ok(FusedProj::Separate(ws))
                }
            }
            None => {
                // Dense (F16/F32): concat the [in, out_i] weight nodes along the out
                // axis → one `mm`. Weight-only concat bakes once (RLX_QWEN3_BAKE_WEIGHTS).
                let mut nodes = Vec::with_capacity(keys.len());
                let mut dims = Vec::with_capacity(keys.len());
                for &key in keys {
                    let n = self.load_param_typed(key, transpose, dtype)?;
                    match self.node_shape(n)?.dims().last() {
                        Some(Dim::Static(d)) => dims.push(*d),
                        _ => {
                            // Unknown out dim → emit unfused rather than guess.
                            let mut ws: Vec<LinearWeight> =
                                nodes.into_iter().map(LinearWeight::Dense).collect();
                            ws.push(LinearWeight::Dense(n));
                            for &k2 in &keys[ws.len()..] {
                                ws.push(LinearWeight::Dense(
                                    self.load_param_typed(k2, transpose, dtype)?,
                                ));
                            }
                            return Ok(FusedProj::Separate(ws));
                        }
                    }
                    nodes.push(n);
                }
                let last_ax = self.node_shape(nodes[0])?.rank() - 1;
                let concat = HirMut::new(self.hir()).concat_(nodes, last_ax);
                Ok(FusedProj::Fused {
                    weight: LinearWeight::Dense(concat),
                    dims,
                })
            }
        }
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
