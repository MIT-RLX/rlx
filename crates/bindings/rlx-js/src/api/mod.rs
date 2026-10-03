// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `rlx` global, assembled one concern per module.

pub mod autodiff;
pub mod buffers;
pub mod check;
pub mod compile;
pub mod console;
pub mod devices;
pub mod dsl;
pub mod fs;
#[cfg(feature = "gguf")]
pub mod gguf;
pub mod graph;
pub mod graph_ext;
pub mod routing;
pub mod session;
#[cfg(feature = "text")]
pub mod text;
pub mod timers;
#[cfg(feature = "training")]
pub mod train;
#[cfg(feature = "weights")]
pub mod weights;
