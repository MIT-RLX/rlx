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

//! SPDNet / Riemannian manifold flow stages — BiMap / ReEig / LogEig and SPD
//! batch-norm.
//!
//! These operate on symmetric-positive-definite (SPD) matrices — the geometric
//! deep-learning counterparts of Linear / ReLU / batch-norm. They are
//! **F64-first**: the underlying CPU kernels (`rlx_cpu::spd`) `expect_f64` and
//! error at runtime on F32. Because the flow `params` auto-upload map is
//! `Vec<f32>`-only, SPD params are declared as F64 graph nodes via
//! [`FlowCtx::declare_param_f64`] and their bytes must be uploaded out of band
//! at session time (`set_param_typed(key, &bytes, DType::F64)`), exactly like
//! GGUF U8 quant blobs.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// BiMap (bilinear mapping) SPDNet layer: `Y = W · X · Wᵀ`.
///
/// Loads the semi-orthogonal weight `W [out_dim, in_dim]` (F64) and applies it
/// to the current SPD input `X [in_dim, in_dim]`, producing
/// `Y [out_dim, out_dim]`. `in_dim` is read from the input's leading axis.
#[derive(Debug, Clone)]
pub struct BiMapStage {
    pub weight_key: String,
    pub out_dim: usize,
}

impl BiMapStage {
    pub fn new(weight_key: impl Into<String>, out_dim: usize) -> Self {
        Self {
            weight_key: weight_key.into(),
            out_dim,
        }
    }
}

impl BlockStage for BiMapStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let in_dim = input
            .shape
            .dim(input.shape.rank().saturating_sub(1))
            .unwrap_static();
        let w = ctx.declare_param_f64(&self.weight_key, &[self.out_dim, in_dim]);
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.bimap(w, input.id);
        let out_shape = gb.shape(id).clone();
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}

/// ReEig (eigenvalue rectification) SPDNet nonlinearity: `Y = U·max(ε,Σ)·Uᵀ`.
/// The SPD analogue of ReLU. No learnable weights — just `eps`. `X` is
/// `[n, n]`; output `Y` is the same shape.
#[derive(Debug, Clone)]
pub struct ReEigStage {
    pub eps: f32,
}

impl ReEigStage {
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }
}

impl BlockStage for ReEigStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.reeig(input.id, self.eps);
        let out_shape = gb.shape(id).clone();
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}

/// LogEig SPDNet layer: `Y = logm(X) = U·log(Σ)·Uᵀ`. Maps the SPD manifold to
/// the tangent space at the identity. No learnable weights — just `eps`.
#[derive(Debug, Clone)]
pub struct LogEigStage {
    pub eps: f32,
}

impl LogEigStage {
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }
}

impl BlockStage for LogEigStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.logeig(input.id, self.eps);
        let out_shape = gb.shape(id).clone();
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}

/// SPD batch-norm — **eval / inference mode** affine transport:
/// `Y_i = G^{1/2} (M^{-1/2} X_i M^{-1/2}) G^{1/2}`.
///
/// Loads the learnable SPD bias `G [n, n]` and the **frozen running Fréchet
/// mean** `M [n, n]` (both F64), then applies [`HirMut::spd_batch_norm_transport`]
/// to the batch `X [batch, n, n]`; `n` is read from the input's trailing axis.
///
/// # Running-mean lifecycle
/// This flow stage is **eval-mode only** — it transports against the *stored*
/// running mean and computes no batch statistic. Training-time batch-mean
/// computation + running-mean update is the **trainer's** job, not this DSL's:
/// the MIR builder [`rlx_ir::Graph::spd_batch_norm`] does the training-mode
/// Karcher-mean + transport, and the running buffer is updated out of band via
/// `rlx_cpu::spd::geodesic_interp(running, batch_mean, momentum)`. This mirrors
/// Euclidean batch-norm, where inference uses the frozen running mean/var.
#[derive(Debug, Clone)]
pub struct SpdBatchNormStage {
    pub g_key: String,
    pub running_mean_key: String,
    pub eps: f32,
}

impl SpdBatchNormStage {
    pub fn new(g_key: impl Into<String>, running_mean_key: impl Into<String>, eps: f32) -> Self {
        Self {
            g_key: g_key.into(),
            running_mean_key: running_mean_key.into(),
            eps,
        }
    }
}

impl BlockStage for SpdBatchNormStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        // Input is `[batch, n, n]`; `n` is the trailing axis.
        let n = input
            .shape
            .dim(input.shape.rank().saturating_sub(1))
            .unwrap_static();
        let g = ctx.declare_param_f64(&self.g_key, &[n, n]);
        let mean = ctx.declare_param_f64(&self.running_mean_key, &[n, n]);
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.spd_batch_norm_transport(input.id, mean, g, self.eps);
        let out_shape = gb.shape(id).clone();
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
