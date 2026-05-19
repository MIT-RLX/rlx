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

//! RLX graph optimizer — transforms IR graphs for maximum performance.
//!
//! Passes are composable and run in sequence on a [`rlx_ir::Graph`]. Each pass
//! produces a new graph (functional style, no in-place mutation) for
//! easy debugging and pass composition.
//!
//! # Default pass pipeline (run by every backend)
//!
//! 1. **Const-fold** (`ConstantFolding`) — fold constant subgraphs.
//! 2. **Fusion** (`fusion::*`) — matmul+bias+act, residual+LN, QKV
//!    concat, SwiGLU concat, attention block.
//! 3. **Mark elementwise regions** (`MarkElementwiseRegions`) — collapse
//!    elementwise chains into a single region op.
//! 4. **Legalize for backend** (`legalize_for_backend`) — reject ops
//!    the target backend can't lower. Catches missing op coverage at
//!    compile time instead of runtime.
//! 5. **Memory planning** (`memory::*`) — liveness analysis → arena
//!    buffer assignment.
//!
//! # Opt-in passes
//!
//! Run by specific backends or user code; not in the default flow.
//!
//! * [`LegalizeBroadcast`] — materialize non-trailing broadcasts via
//!   `Op::Expand`. Required for TPU (HLO needs explicit broadcasts)
//!   and cortexm; CPU/Metal handle modulo broadcasts inline.
//! * [`insert_q_dq`] — post-training quantization Q/DQ insertion.
//!   Caller supplies a `CalibrationRecord` from a calibration run.
//! * [`LowerControlFlow`] / [`LowerDotGeneral`] — lower XLA-shaped
//!   primitives to the standard op set. Run by backends that prefer
//!   primitive ops.
//!
//! # Transforms (JAX-shaped)
//!
//! * [`autodiff`] — reverse-mode AD (`grad_with_loss`).
//! * [`jvp`] / [`hvp`] — forward-mode AD.
//! * [`vmap()`] — batched function transform (the function; the same
//!   name is reused for the module that defines it).

pub mod autodiff;
pub mod autodiff_fwd;
pub mod const_fold;
pub mod control_flow;
pub mod dce;
pub mod fusion;
pub mod inline;
pub mod lower_dot_general;
pub mod memory;
pub mod pass;
pub mod precision;
pub mod promote_params;
pub mod svg;
pub mod vmap;

pub use autodiff_fwd::{hvp, jvp};
pub use control_flow::LowerControlFlow;
pub use inline::inline_into;
pub use promote_params::promote_params_to_inputs;
pub use vmap::vmap;

pub use const_fold::ConstantFolding;
pub use dce::DeadCodeElimination;
pub use fusion::{
    FuseAttentionBlock, FuseMatMulBiasAct, FuseResidualLN, FuseSharedInputMatMul, FuseSwiGLU,
    MarkElementwiseRegions, UnfuseElementwiseRegions,
};
pub use lower_dot_general::LowerDotGeneral;
pub use pass::{Pass, run_passes};
pub use precision::{AutoMixedPrecision, CastConfig, OpKind, Precision, PrecisionPolicy};
pub mod legalize;
pub use legalize::{LegalizeResult, format_legalize_error, legalize_for_backend};
pub mod legalize_broadcast;
pub use legalize_broadcast::LegalizeBroadcast;
pub mod quant_insert;
pub use quant_insert::{CalibrationEntry, CalibrationRecord, insert_q_dq};
pub mod quant_propagate;
pub use memory::is_pure_view;
