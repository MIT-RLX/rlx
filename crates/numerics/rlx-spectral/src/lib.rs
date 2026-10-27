// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-spectral — spectral front-ends, pure Rust, in f32 or f64
//!
//! The audio feature pipelines models consume, built from shared pieces and
//! generic over the working precision ([`Float`]: `f32` for production,
//! `f64` for reference-grade runs):
//!
//! * [`fft`] — [`FftPlan`] (Stockham mixed radix 4/2/3/5, Bluestein for other
//!   primes) and [`RealFftPlan`] (half-length complex + split);
//! * [`window`] — Hann (periodic / symmetric), Hamming, Povey;
//! * [`stft`] — framing (centered reflect / snip-edges), Kaldi frame
//!   conditioning, power spectra frame by frame;
//! * [`mel`] — Slaney / HTK / Kaldi mel scales, filterbanks in Hz or mel
//!   space, sparse application;
//! * [`pipeline`] — [`WhisperLogMel`] and [`KaldiFbank`] (SeamlessM4T /
//!   w2v-BERT `input_features`).
//!
//! These aim at speed and accuracy, not at another library's rounding; the
//! bit-exact NumPy/SciPy paths live in rlx-pocketfft / rlx-signal, for parity.

pub mod fft;
pub mod float;
pub mod mel;
pub mod pipeline;
pub mod stft;
pub mod window;

pub use fft::{FftPlan, RealFftPlan};
pub use float::{Complex, Float};
pub use mel::{MelBank, MelNorm, MelScale, SparseBank};
pub use pipeline::{KaldiFbank, WhisperLogMel};
pub use stft::{FramePrep, Pad, Stft};
pub use window::Window;
