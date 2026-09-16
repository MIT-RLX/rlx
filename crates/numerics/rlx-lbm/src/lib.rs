// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **rlx-lbm** — moment-encoded lattice Boltzmann (HOME-LBM) for RLX.
//!
//! Stores the first three velocity moments per node and rebuilds the
//! populations inside the kernel, rather than storing 9 (D2Q9) or 27 (D3Q27)
//! distribution values. From Li, Wang, Pan, Gao, Wu and Desbrun, *High-Order
//! Moment-Encoded Kinetic Simulation of Turbulent Flows*, ACM TOG 42(6), 2023.
//!
//! ## Why this lives in RLX
//!
//! Two reasons, neither of them fluid dynamics.
//!
//! 1. **It is the same trick as the quantized-weight path.** `DequantMatMul`,
//!    `ScaledMatMul` and `SynthMatMul` all keep a compact representation and
//!    expand it in the inner loop to trade DRAM for ALU. HOME-LBM does exactly
//!    that for a stencil: 27 values become 10, and streaming becomes a gather of
//!    moments. Same shape, a different access pattern.
//!
//! 2. **It exercises access patterns nothing else in the workspace does.** A
//!    27-point periodic stencil and a large-grid-to-small-array reduction look
//!    nothing like a transformer, so they stress the fusion and region machinery
//!    where transformer graphs never reach.
//!
//! ## Modules
//!
//! - [`lattice`] — D2Q9 / D3Q27 velocities, weights, Hermite tensors
//! - [`moment`] — the stored state, third-order reconstruction, closed-form collision
//! - [`invariant`] — the moment round-trip oracle ([`invariant::round_trip_d2q9`])
//! - [`sim`] — a periodic host reference simulator and Taylor–Green setup
//! - [`graph`] *(feature `ir`)* — one LBM step as an rlx [`rlx_ir::Graph`],
//!   streaming via [`rlx_ir::Op::Roll`]
//!
//! ## Correctness posture
//!
//! The reconstruction has no reference implementation to be checked against —
//! it *is* the scheme. So the tests assert properties instead: the moment
//! round-trip ([`invariant`]), exact mass conservation, and Taylor–Green decay
//! against the analytic `exp(−2νk²t)`. The last one measures effective
//! viscosity, which is precisely what a mis-scaled reconstruction corrupts while
//! still producing plausible-looking flow.
//!
//! ```
//! use rlx_lbm::invariant::{round_trip_d2q9, sample_states_2d};
//! for m in sample_states_2d() {
//!     assert!(round_trip_d2q9(&m).within(1e-12));
//! }
//! ```

pub mod invariant;
pub mod lattice;
pub mod moment;
pub mod sim;

#[cfg(feature = "ir")]
pub mod graph;

pub use invariant::{Residual, round_trip_d2q9, round_trip_d3q27};
pub use lattice::CS2;
pub use moment::{
    Moments2d, Moments3d, collide_d2q9, moments_d2q9, moments_d3q27, omega, reconstruct_d2q9,
    reconstruct_d3q27,
};
pub use sim::{Field2d, taylor_green, taylor_green_decay};
