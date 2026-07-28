// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! I/O for jax_fdm-compatible network files.

pub mod json;
pub mod mesh;

pub use json::{
    from_json_path, from_json_str, merge_mesh, mesh_from_json_path, mesh_from_json_str,
    to_json_path, to_json_str,
};
pub use mesh::MeshDocument;
