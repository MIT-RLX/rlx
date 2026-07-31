// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **rlx-geo** — exact integer 2D Delaunay triangulation and discrete Voronoi
//! diagrams, exposed as RLX custom ops with per-backend kernels.
//!
//! # Layers
//! * Pure geometry (`predicates`, `triangulate`, `voronoi`) — no rlx deps,
//!   builds under `--no-default-features`.
//! * RLX ops (`ops`, feature `cpu`) — `geo.delaunay` and `geo.voronoi_grid`
//!   registered as `Op::Custom` with an `OpExtension` (shape inference) and a
//!   `CpuKernel`. Every backend can run these via the CPU kernel.
//! * Native GPU kernels (feature `gpu`) — a wgpu/WGSL `geo.voronoi_grid` kernel
//!   that dispatches straight against the tensor arena.
//!
//! Call [`register`] once per process before compiling graphs that use the ops.

pub mod fastest;
pub mod flip;
pub mod predicates;
pub mod triangulate;
pub mod voronoi;

pub use fastest::{Backend, triangulate_fastest};
pub use flip::{flip_all_convex_once, flip_to_delaunay, hull_seed, interior_quads};
pub use predicates::{FAST_COORDINATE_SPAN, MAX_COORDINATE_SPAN, PredicateWidth};
pub use triangulate::{triangulate, triangulate_dwyer, triangulate_par};
pub use voronoi::{voronoi_dual, voronoi_grid_exact};

#[cfg(feature = "cpu")]
pub mod ops;
#[cfg(feature = "cpu")]
pub use ops::{GEO_DELAUNAY, GEO_VORONOI_GRID, register_geo_ops};

#[cfg(feature = "gpu")]
pub mod flip_gpu;
#[cfg(feature = "gpu")]
pub mod predicates_wgsl;
#[cfg(feature = "gpu")]
pub mod wgpu_kernels;

/// Register all rlx-geo IR extensions and kernels. Idempotent-ish (re-register
/// warns). Call once before compiling graphs that reference the geo ops.
#[cfg(feature = "cpu")]
pub fn register() {
    ops::register_geo_ops();
}
