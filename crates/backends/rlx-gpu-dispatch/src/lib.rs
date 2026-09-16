// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shape-keyed GPU kernel-variant dispatch, shared by every RLX GPU backend.
//!
//! See the crate README for the layering rationale. In short: this crate holds
//! *decisions* (which physical schedule runs for which shape on which device)
//! and the parameter space they range over. It deliberately holds no kernel
//! sources, so Metal and wgpu can depend on it without linking the CUDA/HIP
//! text in `rlx-gpu-kernels`.

pub mod cost;
pub mod dispatch;
pub mod indexing;
pub mod tiles;
