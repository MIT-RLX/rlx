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

//! CPU host-fallback for the core Riemannian / SPD-manifold ops on ROCm.
//!
//! Evaluation lives in [`rlx_gpu_host`]; this module re-exports the predicate
//! and eval entry points used by compile/runtime. `Op::Eigh` / `Op::EighBatch`
//! with `n ≤ 32` prefer the on-device hipSOLVER path (`Step::EighNative`) when
//! available; larger `n` and backwards still land here.

pub use rlx_gpu_host::{eval_spd as eval, is_spd_host};
