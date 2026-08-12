// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::SynthReconstruct` to primitives — the all-backend correctness
//! oracle (bit-identical to the fused kernel). `w_bt[n,k]` from indices `[n, k/d]`
//! + codebook: `Cast → Reshape → Gather → Reshape(→w_bt[n,k])`. The caller emits
//! the `Transpose` to `W[k,n]` separately. Backends with the native fused kernel
//! (Metal) keep the node.

use crate::rewriter::{MatchRewrite, RewriteCtx};
use rlx_ir::*;

pub struct LowerSynthReconstruct;

impl MatchRewrite for LowerSynthReconstruct {
    fn name(&self) -> &str {
        "lower_synth_reconstruct"
    }

    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::SynthReconstruct]
    }

    fn rewrite(&self, node: &Node, ctx: &mut RewriteCtx) -> Option<NodeId> {
        let Op::SynthReconstruct {
            kind: SynthKind::Codebook { entry_dim, .. },
        } = &node.op
        else {
            return None;
        };

        let d = (*entry_dim as usize).max(1);
        let (indices, codebook) = (ctx.input(0), ctx.input(1));
        // Index shape comes from the operand as it stands in the output graph;
        // this rewrite only ever reads it, never the op behind it.
        let idx_shape = ctx.out.node(indices).shape.clone();
        let n = idx_shape.dim(0).unwrap_static();
        let kb = idx_shape.dim(1).unwrap_static();
        let p = n * kb;

        let idx_i64 = ctx.emit(
            Op::Cast { to: DType::I64 },
            vec![indices],
            Shape::new(&[n, kb], DType::I64),
        );
        let idx_flat = ctx.emit(
            Op::Reshape {
                new_shape: vec![p as i64],
            },
            vec![idx_i64],
            Shape::new(&[p], DType::I64),
        );
        let rows = ctx.emit(
            Op::Gather { axis: 0 },
            vec![codebook, idx_flat],
            Shape::new(&[p, d], DType::F32),
        );
        // → w_bt[n,k] (node.shape); the forward `Transpose` is emitted by the caller.
        Some(ctx.emit(
            Op::Reshape {
                new_shape: vec![n as i64, (kb * d) as i64],
            },
            vec![rows],
            node.shape.clone(),
        ))
    }
}
