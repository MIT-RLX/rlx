// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side sequence search shared by every RLX decoder.
//!
//! A model graph ends at a score row per step (`[vocab]` logits for a language
//! model, `[frames, classes]` log-probabilities for a CTC head). What turns
//! those rows into tokens is the same few algorithms whichever model produced
//! them, so they live here once instead of in each model crate:
//!
//! - [`select`] — [`argmax`], [`top_k`], [`log_softmax`], [`log_sum_exp`];
//!   one tie rule (lowest index wins) everywhere.
//! - [`beam`] — autoregressive beam search as a stepping state machine
//!   ([`BeamSearch`]) or a loop ([`beam_search`]), with the Hugging Face and
//!   `openai/whisper` rules as presets and [`Constraint`] for grammar /
//!   suppression masks.
//! - [`ctc`] — greedy, streaming prefix beam, and the CTC trellis
//!   [`ctc::forward`] over a [`Semiring`].
//! - [`align`] — monotonic Viterbi forced alignment.
//! - [`semiring`] — [`Log`] (sum over paths) and [`Tropical`] (best path).
//!
//! No dependencies, so a model crate that only decodes need not pull in the
//! runtime. `rlx-runtime` re-exports this crate as `rlx_runtime::search`.

pub mod align;
pub mod beam;
pub mod ctc;
pub mod select;
pub mod semiring;

pub use align::{AlignError, Span, monotonic_align};
pub use beam::{
    BeamConfig, BeamSearch, Constraint, FinishRule, Hypothesis, LiveAtEnd, Scores, Step,
    beam_search,
};
pub use ctc::{CtcPath, CtcToken, PrefixBeam};
pub use select::{argmax, argmax_u32, log_add, log_softmax, log_sum_exp, top_k};
pub use semiring::{Log, Semiring, Tropical};
