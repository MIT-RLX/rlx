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

//! Fusion policy for HIR → MIR lowering.

/// How HIR block ops lower to MIR.
///
/// Fusion is a first-class concern at the HIR layer: model builders
/// choose whether to express **intent** (`Fusable`) or emit **fused
/// MIR ops directly** (`Direct`).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FusionPolicy {
    /// Lower to primitive chains the optimizer recognizes (`MatMul →
    /// Add → Silu`, shared-input matmul pairs, …). Fusion passes in
    /// `rlx_opt` collapse them to fused ops.
    #[default]
    Fusable,
    /// Lower straight to fused MIR ops (`FusedMatMulBiasAct`,
    /// `FusedSwiGLU`, `FusedResidualRmsNorm`, …). Passes still run for
    /// escape-hatch `HirOp::Mir` nodes and opportunistic cleanup.
    Direct,
}

impl FusionPolicy {
    pub fn is_direct(self) -> bool {
        matches!(self, Self::Direct)
    }

    /// Lower HIR to **primitive MIR chains** so autodiff can skip
    /// tier-2 fused-op unfuse when you control lowering explicitly.
    ///
    /// Same as [`Self::Fusable`]. Fusion passes may still fuse later
    /// during [`CompilePipeline`](../../rlx-opt/src/compiler.rs) for
    /// inference; run [`rlx_opt::prepare_graph_for_ad`] before AD when
    /// the graph already contains fused ops from `Direct` lowering.
    pub const fn for_autodiff() -> Self {
        Self::Fusable
    }
}
