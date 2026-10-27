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

// Deliberate in a line-for-line port of reference LAPACK / OpenBLAS (flagged
// by clippy 1.99). The control flow mirrors the Fortran/C source so the port
// can be reviewed against it, and every result is checked bit for bit:
// - `mut_range_bound`: Fortran evaluates DO-loop bounds once, so updating `I1`
//   inside DLARRB's loop leaves the iteration unchanged but feeds later tests —
//   exactly what a Rust range does.
// - `if_same_then_else`: DLARRV's `IF … ELSE IF … ELSE GO TO` keeps the second
//   test unevaluated when the first holds, as the reference does.
// - `manual_memcpy`, `explicit_counter_loop`, `needless_late_init`,
//   `int_plus_one`, `assign_op_pattern`, `manual_div_ceil`: shape only, kept as
//   the source writes them.
#![allow(
    clippy::mut_range_bound,
    clippy::if_same_then_else,
    clippy::manual_memcpy,
    clippy::explicit_counter_loop,
    clippy::needless_late_init,
    clippy::int_plus_one,
    clippy::assign_op_pattern,
    clippy::manual_div_ceil
)]

pub mod lapack;
mod lapack_dc;
mod lapack_dqds;
mod lapack_eig;
mod lapack_mrrr;
mod lapack_syevd;
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
pub use lapack_syevd::{dstevd_v, dsyevd_lower};
