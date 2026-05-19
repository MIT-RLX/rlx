// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! RLX Cortex-M backend — INT8 kernels for ARMv7E-M microcontrollers.
//!
//! Targets the nRF52840 dongle (Cortex-M4F, 64 MHz, 256 KB RAM, 1 MB
//! flash) but the kernel layer is plain `no_std` Rust and runs on any
//! 32-bit target, including the host for tests.
//!
//! ## Quantization
//!
//! Per-tensor symmetric for activations, per-tensor for weights too in
//! this first cut (per-channel weights is a follow-up). Output of each
//! op is requantized via a single f32 multiplier — the FPU on the M4F
//! makes this cheaper than the fixed-point dance CMSIS-NN does for
//! M0-class chips.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod argmax;
pub mod conv2d;
pub mod dense;
pub mod maxpool;
pub mod quant;
pub mod relu;

pub mod model;
pub mod model_weights;

#[cfg(feature = "std")]
pub mod reference;
