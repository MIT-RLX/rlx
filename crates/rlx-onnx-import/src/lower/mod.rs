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

mod ops;
mod options;

pub use ops::{build_hir_from_bundle, build_hir_from_parts, resolve_shape};
pub use options::{DurationLoopLowering, ImportOptions, ImportReport};

use anyhow::Result;

/// Resolve a bundle shape dim (re-exported for shape propagation).
pub fn resolve_dim(v: &serde_json::Value, opts: &ImportOptions) -> Result<usize> {
    ops::resolve_dim(v, opts)
}
