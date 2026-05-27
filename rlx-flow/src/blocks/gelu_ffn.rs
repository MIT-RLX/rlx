// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct GeluFfnStage {
    pub intermediate_w: String,
    pub intermediate_b: String,
    pub output_w: String,
    pub output_b: String,
    pub approx_gelu: bool,
}

impl GeluFfnStage {
    pub fn new(
        intermediate_w: impl Into<String>,
        intermediate_b: impl Into<String>,
        output_w: impl Into<String>,
        output_b: impl Into<String>,
    ) -> Self {
        Self {
            intermediate_w: intermediate_w.into(),
            intermediate_b: intermediate_b.into(),
            output_w: output_w.into(),
            output_b: output_b.into(),
            approx_gelu: false,
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
        let p = layer_prefix.into();
        Self {
            intermediate_w: format!("{p}.mlp.fc1.weight"),
            intermediate_b: format!("{p}.mlp.fc1.bias"),
            output_w: format!("{p}.mlp.fc2.weight"),
            output_b: format!("{p}.mlp.fc2.bias"),
            approx_gelu: true,
        }
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
        let act = if self.approx_gelu {
            gb.gelu_approx(int_add)
        } else {
            gb.gelu(int_add)
        };
        let out_mm = gb.mm(act, out_w);
        let out = gb.add(out_mm, out_b);
        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}
