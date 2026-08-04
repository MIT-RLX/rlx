// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Apple matrix-coprocessor fast paths (AMX / SME).
//!
//! On Apple Silicon the fastest CPU matmul is not NEON — it's the matrix
//! coprocessor. Three independent, opt-in routes onto it live here, each behind
//! its own cargo feature so the base build stays lean ([[feedback_empty_prelude]],
//! [[feedback_optimizations_behind_features]]):
//!
//! * [`bnns`]  (`amx-bnns`)  — low-precision (int8 W8A8) matmul via BNNS. Apple
//!   dispatches BNNS to AMX on M1–M3 and SME on M4+ for us, so we get the
//!   coprocessor for quantized inference without owning per-generation asm.
//! * [`dense`] (`amx-dense`) — verification + benchmark that Accelerate's sgemm
//!   already *is* the AMX path for dense f32/f64. Nothing to beat; we measure.
//! * [`sme`]   (`amx-sme`)   — a direct ARM SME2 `FMOPA` GEMM microkernel via
//!   `global_asm!`. The documented M4+ path; hand-written, runtime-gated.
//!
//! Whichever path is compiled in, [`detect`] provides the runtime probe that
//! decides whether the *current* chip actually has the unit before dispatch.

pub mod detect;

#[cfg(rlx_cpu_amx_bnns)]
pub mod bnns;

#[cfg(rlx_cpu_amx_dense)]
pub mod dense;

#[cfg(rlx_cpu_amx_sme)]
pub mod sme;
