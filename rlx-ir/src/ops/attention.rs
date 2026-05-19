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

//! Attention builders: SDPA with custom or kernel-synthesized
//! masks (plan #53).

use crate::op::MaskKind;
use crate::{Graph, NodeId, Op, Shape};

impl Graph {
    /// Scaled dot-product attention with a custom (caller-supplied) mask.
    /// Equivalent to `attention_kind(.., MaskKind::Custom, ..)`.
    pub fn attention(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        mask: NodeId,
        num_heads: usize,
        head_dim: usize,
        shape: Shape,
    ) -> NodeId {
        self.push(
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind: MaskKind::Custom,
            },
            vec![q, k, v, mask],
            shape,
            None,
        )
    }

    /// Scaled dot-product attention with a kernel-synthesized mask
    /// (`None` / `Causal` / `SlidingWindow`). Inputs are Q, K, V only —
    /// no mask tensor is allocated or read in the inner loop. Use
    /// `MaskKind::None` for a single un-padded sequence.
    pub fn attention_kind(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
    ) -> NodeId {
        debug_assert!(
            !matches!(mask_kind, MaskKind::Custom | MaskKind::Bias),
            "attention_kind() requires a non-tensor MaskKind; use attention() for Custom or attention_bias() for Bias"
        );
        self.push(
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
            },
            vec![q, k, v],
            shape,
            None,
        )
    }

    /// Scaled dot-product attention with an additive bias tensor of shape
    /// `[batch, num_heads, query_len, key_len]` added to the
    /// `QK^T · scale` scores before softmax. Lets boxRPB / per-query
    /// position biases reuse the fast `Op::Attention` kernel path.
    pub fn attention_bias(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        bias: NodeId,
        num_heads: usize,
        head_dim: usize,
        shape: Shape,
    ) -> NodeId {
        self.push(
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind: MaskKind::Bias,
            },
            vec![q, k, v, bias],
            shape,
            None,
        )
    }
}
