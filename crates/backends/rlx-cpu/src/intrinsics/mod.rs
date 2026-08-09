// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// x86-64 VNNI (`VPDPBUSD`) int8 dot kernels for low-bit quant matmul.
/// Runtime-detected (AVX-512-VNNI or AVX-VNNI); scalar fallback inside.
#[cfg(target_arch = "x86_64")]
pub mod vnni;

/// Apple AMX / SME matrix-coprocessor fast paths. Only compiled on Apple
/// platforms; the `amx-*` cargo features light the per-path submodules within.
#[cfg(target_vendor = "apple")]
pub mod apple_amx;
