// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The convolution torch 2.10's CPU build runs through NNPACK
//! (`_ConvBackend.NnpackSpatial`: float32, no dilation, batch ≥ 16) — NNPACK's
//! 16×16 Fourier algorithm (`ft16x16`), which rounds nothing like a direct sum —
//! reproduced bit for bit in **pure Rust** ([`ft16x16`]): no C, no SIMD intrinsics,
//! so it runs (and agrees) on every target, wasm included.
//!
//! A translation of `Maratyszcza/NNPACK@c07e3a0` (BSD-2, `LICENSE-NNPACK`) and
//! `Maratyszcza/psimd@072586a` (MIT, `LICENSE-PSIMD`) as torch's macOS arm64 wheel
//! compiles them; `tests/torch_parity.rs` checks it against torch's own
//! `_nnpack_spatial_convolution` (`parity/export_torch_nnpack.py`).

pub mod ft16x16;

pub use ft16x16::{conv_output, status};
