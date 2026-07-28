// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
pub mod audio_ops;
pub mod axial_rope2d;
pub mod backward;
pub mod blocks;
pub mod conv2d;
pub mod conv3d;
pub mod dsp;
pub mod elementwise;
pub mod fft_ops;
pub mod io;
pub mod linalg;
pub mod manifold;
pub mod normalization;
pub mod reduction;
pub mod shape_ops;
pub mod spd_eig;
pub mod spd_graph;
pub mod special;
pub mod spectral;
pub mod splat;
pub mod upsample;
pub mod vq;
