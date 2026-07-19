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

//! IR-level "unfusion" pass for the ROCm/HIP backend.
//!
//! The shared decompose driver lives in `rlx-unfuse`; this module only
//! supplies ROCm's [`DecomposePolicy`]. ROCm shares CUDA's `.cu` kernels but
//! (unlike CUDA) does not keep any `FusedAttentionBlock` native and does not
//! promote `AttentionBackward` — so the policy is the plain default: lower
//! every composite to primitives, materialize rank-4 attention, no
//! FusedMatMulBiasAct / FusedResidualLN folding.

use rlx_ir::Graph;
use rlx_unfuse::DecomposePolicy;

/// ROCm's decompose policy — all defaults (see [`DecomposePolicy`]).
pub(crate) struct RocmPolicy;

impl DecomposePolicy for RocmPolicy {}

pub fn unfuse(graph: Graph) -> Graph {
    rlx_unfuse::unfuse(graph, &RocmPolicy)
}
