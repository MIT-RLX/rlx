// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-linux-ref — the Linux x86-64 reference, bit for bit, in pure Rust
//!
//! What NumPy 2.3.5 / SciPy 1.17.1 compute on the Linux x86-64 reference machine
//! (an AVX2 Intel, Haswell kernels; msi), reproduced operation for operation so a
//! Rust port agrees with them exactly on Linux and in wasm:
//!
//! * [`openblas`] — OpenBLAS 0.3.30's Haswell kernels and level-2/3 drivers
//!   (`ddot`, `dnrm2`, `drot`, `dger`, `dgemv`, `dgemm` incl. its thread
//!   partitioning, `dsyrk`, `dtrmm`, `dgesv`);
//! * [`lapack`] — the reference LAPACK 3.12 OpenBLAS bundles (gfortran, no FMA):
//!   `dgelsd` with blocked `dgebrd`/`dgeqrf`/`dormqr`/`dormlq` and the divide-and-
//!   conquer `dlalsd`; `dsyevr` (`scipy.linalg.eigh`) with blocked `dsytrd`,
//!   the MRRR `dstemr` and `dormtr`;
//! * [`xss`] — x86-simd-sort's AVX2 `argsort` (what `np.argsort` dispatches to);
//! * [`x87`] — 64-bit-significand `long double` arithmetic (`nrm2.S`, pocketfft's
//!   `ldbl_t`).
//!
//! Every module is checked against the reference in `tests/` (fixtures generated
//! on the reference by `tools/`), zero tolerance.

pub mod lapack;
mod lapack_dc;
mod lapack_dqds;
mod lapack_eig;
mod lapack_mrrr;
pub mod openblas;
pub mod x87;
pub mod xss;

/// CBLAS `CblasRowMajor` / `CblasColMajor`.
pub const ROW_MAJOR: i32 = 101;
pub const COL_MAJOR: i32 = 102;
/// CBLAS `CblasNoTrans` / `CblasTrans`.
pub const NO_TRANS: i32 = 111;
pub const TRANS: i32 = 112;

pub use lapack_eig::{DSYEVR_MAX_N, dsyevr_lower, dsytrd_lower_pub as dsytrd_lower};
