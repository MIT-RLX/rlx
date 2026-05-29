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

//! Per-op graph builders (plan #53).
//!
//! Borrowed from MAX's `max/python/max/graph/ops/` layout: each
//! op family lives in its own file, all of them
//! `impl crate::Graph { ... }`, so the IR core (`Graph` struct +
//! `push` + analysis helpers) stays small and ops can evolve
//! independently.
//!
//! Adding a new op family = drop a new file here, register it in
//! `mod.rs`, write the `impl Graph { ... }` block. No edits to
//! `graph.rs`.

pub mod attention;
pub mod axial_rope2d;
pub mod backward;
pub mod blocks;
pub mod conv2d;
pub mod elementwise;
pub mod fft_ops;
pub mod io;
pub mod linalg;
pub mod normalization;
pub mod reduction;
pub mod shape_ops;
pub mod special;
pub mod splat;
