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

//! Phase-aware streaming inference (plan #16).
//!
//! Re-exports the LIR phase types from [`rlx_ir`] — phase assignment
//! is computed during LIR planning and stored on [`rlx_ir::LirBufferPlan`].

pub use rlx_ir::{Phase, PhaseSchedule, derive_phases};
