// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase-aware streaming inference (plan #16).
//!
//! Re-exports the LIR phase types from [`rlx_ir`] — phase assignment
//! is computed during LIR planning and stored on [`rlx_ir::LirBufferPlan`].

pub use rlx_ir::{Phase, PhaseSchedule, derive_phases};
