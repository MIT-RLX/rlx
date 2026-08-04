// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, Shape};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;
#[derive(Debug, Clone)]
pub struct LmHeadStage {
    pub weight_key: Option<String>,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub tied_param_name: String,
}

impl LmHeadStage {
    pub fn tied(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weight_key: None,
            tie_word_embeddings: true,
            vocab_size,
            hidden_size,
            tied_param_name: "lm_head.tied_t".into(),
        }
    }

    pub fn separate(weight_key: impl Into<String>, vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weight_key: Some(weight_key.into()),
            tie_word_embeddings: false,
            vocab_size,
            hidden_size,
            tied_param_name: "lm_head.tied_t".into(),
        }
    }
}

impl BlockStage for LmHeadStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        // The LM head is the single largest weight read per decoded token
        // ([hidden, vocab]); store it F16-resident to halve that bandwidth on
        // batch-1 decode. Same opt-in as the projection weights; the backend
        // converts the f32 bytes at bind. (Only the head weight — the embed
        // table used by the token gather stays F32.)
        let w_dt = if rlx_ir::env::flag("RLX_QWEN3_F16_WEIGHTS") {
            DType::F16
        } else {
            DType::F32
        };
        let lm_head_w = if self.tie_word_embeddings {
            let embed_key = "model.embed_tokens.weight";
            let embed = ctx
                .params
                .get(embed_key)
                .ok_or_else(|| anyhow::anyhow!("missing {embed_key} for tied lm_head"))?;
            let vocab = self.vocab_size;
            let hidden_size = self.hidden_size;
            let mut transposed = vec![0f32; embed.len()];
            for v in 0..vocab {
                for hi in 0..hidden_size {
                    transposed[hi * vocab + v] = embed[v * hidden_size + hi];
                }
            }
            ctx.synth_param(
                &self.tied_param_name,
                transposed,
                Shape::new(&[hidden_size, vocab], w_dt),
            )
        } else {
            let key = self
                .weight_key
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LmHead: weight_key required when not tied"))?;
            ctx.load_param_typed(key, true, w_dt)?
        };

        let mut gb = HirMut::new(ctx.hir());
        let id = gb.mm(input.id, lm_head_w);
        let dims = input.shape.dims();
        let out_shape = Shape::from_dims(
            &[dims[0], dims[1], rlx_ir::Dim::Static(self.vocab_size)],
            input.shape.dtype(),
        );
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
