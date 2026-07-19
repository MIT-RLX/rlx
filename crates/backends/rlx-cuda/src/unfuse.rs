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

//! IR-level "unfusion" pass for the CUDA backend.
//!
//! The shared decompose driver lives in `rlx-unfuse`; this module only
//! supplies CUDA's [`DecomposePolicy`]. CUDA lowers `FusedMatMulBiasAct`
//! (matmul folds bias + activation into its epilogue) and
//! `fused_residual_{ln,rms_norm}.cu` (Add[+bias] + norm in one kernel)
//! natively, so those stay fused rather than being decomposed to
//! FusedMatMulBiasAct/FusedResidualLN — i.e. the folds are OFF here. CUDA's
//! Attention/AttentionBackward kernels are rank-4-only, and a native
//! `fused_attn_block` kernel can serve small-sequence FusedAttentionBlock
//! nodes intact.

use rlx_ir::{Graph, Shape};
use rlx_unfuse::DecomposePolicy;

/// CUDA's decompose policy: keep small `FusedAttentionBlock` nodes native
/// (`fused_attn_block` kernel), promote rank-3 `AttentionBackward` to rank-4
/// like the forward op, and lower everything to primitives (no
/// FusedMatMulBiasAct / FusedResidualLN folding, rank-4-only attention).
pub(crate) struct CudaPolicy;

impl DecomposePolicy for CudaPolicy {
    /// True when the native `fused_attn_block` kernel can serve this block: the
    /// `[seq, seq]` score matrix must fit the GPU's default 48 KB dynamic
    /// shared-memory budget (with margin). Larger sequences decompose to the
    /// primitive chain. The CUDA arena is f32-uniform, so dtype is always fine.
    fn fab_native(&self, out_shape: &Shape) -> bool {
        let dims = out_shape.dims();
        if dims.len() != 3 {
            return false;
        }
        let s = dims[1].unwrap_static();
        // seq*seq*4 bytes of shared memory; 96 → 36 KB, comfortably under 48 KB.
        s > 0 && s <= 96
    }

    fn promote_attention_backward(&self) -> bool {
        true
    }
}

pub fn unfuse(graph: Graph) -> Graph {
    rlx_unfuse::unfuse(graph, &CudaPolicy)
}
