// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-signal — `scipy.signal` bit-for-bit
//!
//! The parts of SciPy 1.17.1's `scipy.signal` that EEG/fNIRS pipelines (MNE's
//! `raw.filter`, TDDR, SCI) run, reproduced to the last bit on the macOS arm64
//! reference build:
//!
//! * [`iir`] — Butterworth design (`buttap`, `lp2lp/lp2hp/lp2bp_zpk`,
//!   `bilinear_zpk`), `zpk2tf`, `zpk2sos`, `lfilter`, `sosfilt`, `lfilter_zi`,
//!   `sosfilt_zi`, `filtfilt`, `sosfiltfilt`.
//! * [`fir`] — `firwin` and the cosine-sum windows.
//! * [`complex`] — NumPy's `complex128` arithmetic as NumPy rounds it.
//!
//! **Why the spelling matters.** SciPy's inner loops are C/Cython compiled by
//! clang with `-ffp-contract=on`, and NumPy's complex ufuncs likewise: `a*b + c`
//! in one expression is one FMA. The two filter loops ([`iir`]'s `lfilter` /
//! `sosfilt` kernels) are SciPy's expressions translated to Rust with an FMA
//! exactly where clang fuses, and the complex helpers name which product is fused.

pub mod complex;
pub mod fir;
pub mod iir;
