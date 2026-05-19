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

//! ISA-split intrinsics layer (plan #85).
//!
//! Borrowed from MAX's
//! `linalg/arch/cpu/{apple_amx,neon,vnni}_intrinsics.mojo` pattern:
//! one file per ISA, each file is *only* thin typed wrappers around
//! the raw intrinsics — no algorithm logic.
//!
//! Why a layer instead of inline `std::arch::aarch64::*` in kernel
//! code?
//!   - Single place to add target-feature gates (`#[target_feature]`)
//!     when we eventually want runtime AVX2 / SSE4.2 selection.
//!   - Algorithm files (kernels.rs, thunk.rs) read as math, not
//!     as `vfmaq_f32(vmulq_f32(_, _), _, _)`.
//!   - When porting the same kernel to a new ISA you swap one
//!     `use` line, not 50 inline call sites.
//!
//! Migration is incremental. New code added since plan #85 lives
//! here; the existing 19 inline `std::arch::aarch64::*` sites in
//! kernels.rs / thunk.rs migrate as their surrounding kernels
//! are touched.

#[cfg(target_arch = "aarch64")]
pub mod neon;
