// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The whole rlx C ABI, with the reference plugin's models built in.

pub use rlx_capi::*;

rlx_runtime::link_model_plugins!(rlx_model_plugin);
