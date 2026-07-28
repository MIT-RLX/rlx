// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Re-exports of the cortexm weight blob, plus a tiny helper to read
//! per-channel `(M0, shift)` tables from the `*_MULT: &[f32]` slices.
//!
//! The cortexm `model_weights.rs` is the source of truth: re-running the
//! trainer regenerates it, and this crate picks up the new blob on the
//! next build. We don't re-quantize anything here — the i8 weights / i32
//! biases / f32 multipliers are all reused as-is — but the f32 mults
//! get split into `(M0, shift)` *once*, in `model::tinyconv_mnist_from_cortexm`,
//! before either Verilog emission or the Rust reference path runs.

pub use rlx_cortexm::model_weights::{
    C1_SCALE, C2_SCALE, CONV1_B, CONV1_MULT, CONV1_W, CONV2_B, CONV2_MULT, CONV2_W, FC_B, FC_MULT,
    FC_OUT_SCALE, FC_W, P1_SCALE, P2_SCALE, TEST_IMAGE, TEST_LABEL, WEIGHT_BITS, X_SCALE,
};
