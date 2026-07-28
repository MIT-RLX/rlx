// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

mod ops;
mod options;

pub use ops::{build_hir_from_bundle, build_hir_from_parts, resolve_shape};
pub use options::{DurationLoopLowering, ImportOptions, ImportReport};

use anyhow::Result;

/// Resolve a bundle shape dim (re-exported for shape propagation).
pub fn resolve_dim(v: &serde_json::Value, opts: &ImportOptions) -> Result<usize> {
    ops::resolve_dim(v, opts)
}
