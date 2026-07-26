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

//! IR-level "unfusion" pass for the wgpu backend.
//!
//! The shared decompose driver lives in `rlx-unfuse`; this module only
//! supplies wgpu's [`DecomposePolicy`] and re-exports `collapse_reshapes`.
//!
//! The wgpu backend lowers `FusedMatMulBiasAct` (matmul folds bias +
//! activation into its WGSL epilogue) and `FusedResidualLN`
//! (`fused_residual_ln.wgsl` does Add[+bias] + LayerNorm in one pass)
//! natively — so the decompose pass folds biased projections into
//! FusedMatMulBiasAct and residual+norm pairs into FusedResidualLN. Its
//! Attention kernel reads Q/K/V (and the mask) via per-axis strides, so it
//! accepts rank-3 `[B, S, H·D]` inputs directly and skips the
//! reshape/transpose to `[B, H, S, D]`.

use rlx_ir::Graph;
use rlx_unfuse::DecomposePolicy;

/// wgpu's decompose policy: fold biased matmuls into `FusedMatMulBiasAct`,
/// fold residual+norm pairs into `FusedResidualLN`, and pass rank-3 Q/K/V/
/// mask straight to the stride-driven Attention kernel.
pub(crate) struct WgpuPolicy;

impl DecomposePolicy for WgpuPolicy {
    fn fold_matmul_bias_act(&self) -> bool {
        !rlx_ir::env::flag("RLX_WGPU_NO_FOLD_MMBA")
    }

    fn fold_residual_ln(&self) -> bool {
        !rlx_ir::env::flag("RLX_WGPU_NO_FOLD_RESLN")
    }

    fn attention_accepts_rank3(&self) -> bool {
        true
    }
}

pub fn unfuse(graph: Graph) -> Graph {
    rlx_unfuse::unfuse(graph, &WgpuPolicy)
}

pub fn collapse_reshapes(graph: Graph) -> Graph {
    rlx_unfuse::collapse_reshapes(graph)
}
