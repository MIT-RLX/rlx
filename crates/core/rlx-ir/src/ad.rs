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

//! Autodiff staging in the HIR → MIR → LIR pipeline.
//!
//! Differentiation is implemented in `rlx_autodiff` on MIR ([`crate::Graph`]).
//! This module documents how IR stages connect:
//!
//! ```text
//!  HIR (blocks)  ──lower──▶  MIR  ──prepare_graph_for_ad──▶  MIR (primitive)
//!                                    └── grad_with_loss ──▶  backward MIR Graph
//!  MIR (opt)     ──plan──▶   LIR  (inference compile only — not an AD input)
//! ```
//!
//! - **HIR**: use [`crate::hir::FusionPolicy::for_autodiff`] or `Direct` +
//!   `rlx_autodiff::prepare_graph_for_ad`.
//! - **MIR**: [`rlx_autodiff::grad_with_loss`] / [`rlx_autodiff::grad_with_loss_module`].
//! - **LIR**: do not differentiate; lower from MIR first.

/// Named stages in a training-oriented compile flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdPipelineStage {
    /// Block builders (`Graph::define`, `HirModule`).
    HirBuild,
    /// Tensor DAG after [`crate::hir::HirModule::lower_to_mir`].
    MirLowered,
    /// After `rlx_autodiff::prepare_graph_for_ad` (fused ops unfused, scans rewritten).
    MirPrepared,
    /// Gradient graph returned by `rlx_autodiff::grad_with_loss`.
    MirBackward,
}
