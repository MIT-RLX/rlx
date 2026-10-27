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
pub struct GatherLastTokenStage {
    pub batch: usize,
    pub input_name: String,
}

impl GatherLastTokenStage {
    pub fn dynamic(batch: usize) -> Self {
        Self {
            batch,
            input_name: "last_token_idx".into(),
        }
    }

    pub fn static_last(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            input_name: format!("__static_last_{seq}"),
        }
    }
}

impl BlockStage for GatherLastTokenStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let id = if self.input_name.starts_with("__static_last_") {
            let seq: usize = self
                .input_name
                .strip_prefix("__static_last_")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| anyhow::anyhow!("invalid static last token stage"))?;
            let mut gb = HirMut::new(ctx.hir());
            gb.narrow_(input.id, 1, seq - 1, 1)
        } else {
            // Token-position indices — must be integer; MLX `take` and
            // the Metal gather kernel both treat the index tensor as I32.
            // Declaring this F32 produced garbage logits (stable wrong
            // token streams like "< as as as…").
            // A flow that declared the input itself (its tail reads it too)
            // keeps that one node: a second input under the same name is
            // never bound and reads zeros, i.e. position 0.
            let declared = ctx.state.inputs.get(&self.input_name).cloned();
            let mut gb = HirMut::new(ctx.hir());
            let idx = match declared {
                Some((id, shape)) if shape.dtype() == DType::I32 => id,
                Some((id, shape)) => gb.0.mir(
                    rlx_ir::Op::Cast { to: DType::I32 },
                    vec![id],
                    Shape::from_dims(shape.dims(), DType::I32),
                ),
                None => gb.input(&self.input_name, Shape::new(&[self.batch], DType::I32)),
            };
            let idx_2d = gb.reshape_(idx, vec![self.batch as i64, 1]);
            gather_rows_axis1(&mut gb, input.id, idx_2d, self.batch)
        };
        let out_shape = if input.shape.rank() >= 2 {
            let batch = input.shape.dim(0).unwrap_static();
            let hidden = input.shape.dim(2).unwrap_static();
            Shape::new(&[batch, 1, hidden], input.shape.dtype())
        } else {
            input.shape.clone()
        };
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}

/// Row `b` of `x` (`[batch, seq, c]`) at positions `idx[b]` (`[batch, k]`),
/// as `[batch, k, c]` — `take_along_axis` on axis 1.
///
/// ONNX `Gather` splices the whole index shape into its output, so
/// `gather(x, idx, 1)` is `[batch, batch, k, c]`: every row at every row's
/// positions. The answer is its `[b, b]` diagonal. At batch 1 that is a
/// reshape; above it a plain reshape does not even keep the element count.
/// The result is truly rank 3 because downstream nodes declare it so, and a
/// backend that sizes its write from the declared rank fills only part of a
/// rank-4 buffer (on MLX: silently zero logits whenever the gathered position
/// was not 0).
pub fn gather_rows_axis1(
    gb: &mut HirMut,
    x: rlx_ir::HirNodeId,
    idx: rlx_ir::HirNodeId,
    batch: usize,
) -> rlx_ir::HirNodeId {
    let c = {
        let sh = gb.shape(x);
        sh.dim(sh.rank() - 1).unwrap_static()
    };
    let k = gb.shape(idx).dim(1).unwrap_static();
    let w = gb.gather_(x, idx, 1);
    if batch == 1 {
        return gb.reshape_(w, vec![1, k as i64, c as i64]);
    }
    let rows: Vec<rlx_ir::HirNodeId> = (0..batch)
        .map(|b| {
            let row = gb.narrow_(w, 0, b, 1);
            let diag = gb.narrow_(row, 1, b, 1);
            gb.reshape_(diag, vec![1, k as i64, c as i64])
        })
        .collect();
    gb.concat_(rows, 0)
}
