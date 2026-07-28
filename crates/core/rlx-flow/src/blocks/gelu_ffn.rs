// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Activation applied between the two FFN linears (`fc2(act(fc1(x)))`).
///
/// Generalizes the FFN stage so post-norm / ReLU trunks (PyTorch's default
/// `nn.TransformerEncoderLayer` activation) can be assembled in-graph without a
/// custom-stage escape hatch, alongside the GELU variants used by ViT encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    /// Exact erf GELU (`torch.nn.GELU()`), e.g. BERT / LaBraM / EEGPT.
    GeluErf,
    /// tanh-approx GELU (`torch.nn.GELU(approximate="tanh")`), DINOv2 default.
    GeluTanh,
    /// ReLU (`nn.TransformerEncoderLayer` default activation).
    Relu,
}

#[derive(Debug, Clone)]
pub struct GeluFfnStage {
    pub intermediate_w: String,
    pub intermediate_b: String,
    pub output_w: String,
    pub output_b: String,
    pub activation: FfnActivation,
}

impl GeluFfnStage {
    pub fn new(
        intermediate_w: impl Into<String>,
        intermediate_b: impl Into<String>,
        output_w: impl Into<String>,
        output_b: impl Into<String>,
    ) -> Self {
        Self::with_activation(
            intermediate_w,
            intermediate_b,
            output_w,
            output_b,
            FfnActivation::GeluErf,
        )
    }

    /// FFN over explicit fc1/fc2 keys with a chosen [`FfnActivation`].
    pub fn with_activation(
        intermediate_w: impl Into<String>,
        intermediate_b: impl Into<String>,
        output_w: impl Into<String>,
        output_b: impl Into<String>,
        activation: FfnActivation,
    ) -> Self {
        Self {
            intermediate_w: intermediate_w.into(),
            intermediate_b: intermediate_b.into(),
            output_w: output_w.into(),
            output_b: output_b.into(),
            activation,
        }
    }

    /// HuggingFace BERT-style FFN keys under a layer prefix.
    pub fn hf_bert(layer_prefix: impl Into<String>) -> Self {
        let p = layer_prefix.into();
        Self::new(
            format!("{p}.intermediate.dense.weight"),
            format!("{p}.intermediate.dense.bias"),
            format!("{p}.output.dense.weight"),
            format!("{p}.output.dense.bias"),
        )
    }

    /// DINOv2 ViT MLP (`mlp.fc1` / `mlp.fc2`) with tanh-approx GELU.
    pub fn dinov2(layer_prefix: impl Into<String>) -> Self {
        Self::dinov2_with(layer_prefix, true)
    }

    /// DINOv2 ViT MLP with exact (erf) GELU — for parity with checkpoints
    /// trained against `torch.nn.GELU()` (e.g. LaBraM, EEGPT).
    pub fn dinov2_exact(layer_prefix: impl Into<String>) -> Self {
        Self::dinov2_with(layer_prefix, false)
    }

    /// DINOv2 ViT MLP, choosing tanh-approx (`approx=true`) or exact-erf GELU.
    pub fn dinov2_with(layer_prefix: impl Into<String>, approx: bool) -> Self {
        let p = layer_prefix.into();
        let activation = if approx {
            FfnActivation::GeluTanh
        } else {
            FfnActivation::GeluErf
        };
        Self::with_activation(
            format!("{p}.mlp.fc1.weight"),
            format!("{p}.mlp.fc1.bias"),
            format!("{p}.mlp.fc2.weight"),
            format!("{p}.mlp.fc2.bias"),
            activation,
        )
    }
}

impl BlockStage for GeluFfnStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let int_w = ctx.load_param(&self.intermediate_w, true)?;
        let int_b = ctx.load_param(&self.intermediate_b, false)?;
        let out_w = ctx.load_param(&self.output_w, true)?;
        let out_b = ctx.load_param(&self.output_b, false)?;

        let mut gb = HirMut::new(ctx.hir());
        let int_mm = gb.mm(input.id, int_w);
        let int_add = gb.add(int_mm, int_b);
        let act = match self.activation {
            FfnActivation::GeluErf => gb.gelu(int_add),
            FfnActivation::GeluTanh => gb.gelu_approx(int_add),
            FfnActivation::Relu => gb.relu(int_add),
        };
        let out_mm = gb.mm(act, out_w);
        let out = gb.add(out_mm, out_b);
        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}
