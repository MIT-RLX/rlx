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

//! I/O for jax_fdm-compatible network files.

pub mod json;
pub mod mesh;

pub use json::{
    from_json_path, from_json_str, merge_mesh, mesh_from_json_path, mesh_from_json_str,
    to_json_path, to_json_str,
};
pub use mesh::MeshDocument;
