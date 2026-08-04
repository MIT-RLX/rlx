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
//!
//! # Variations & which is fastest (and why)
//!
//! rlx-geo carries several implementations of the same exact result. Run them all with
//! `cargo run --example bench_variants -- [N] [K]` (GPU paths: `flip_gpu_bench`,
//! `leaf_bench`). Representative numbers (M-series, all **bit-exact**):
//!
//! **Full triangulation** (200k random points):
//!
//! | approach | time | why |
//! |---|---|---|
//! | [`triangulate_par`] — Dwyer, all cores | **~4 ms** ★ | best cache behavior + near-linear thread scaling |
//! | [`delaunay32::Triangulator`] `with_threads(0)` — Guibas-Stolfi D&C, parallel | ~18 ms | GS merge is Θ(√n)-serial at the top → scales worse than Dwyer |
//! | [`triangulate`] — Dwyer, serial | ~30 ms | |
//! | `Triangulator::new()` — GS D&C, serial | ~76 ms | textbook recursive D&C, one arena |
//!
//! **On the GPU** the flip pipeline (`flip_gpu`, feature `gpu`) loses end-to-end (~5–8×
//! at 1M) — the seed build is serial and the flip is O(rounds·T) bandwidth/latency-bound —
//! *except* the on-chip cooperative `leaf_gpu` phase, which does 90% of the work at
//! 20–130× the CPU rate (it's dense, parallel, and never touches DRAM). GPUs win the dense
//! leaf; the CPU wins the gather-bound assembly.
//!
//! **In-circle predicate** — the hot inner test. [`incircle_gemm`] recasts it as a matmul
//! (`DET = L·C` via the paraboloid lift) and is the fast, portable, still-exact choice:
//!
//! | predicate | throughput | why |
//! |---|---|---|
//! | [`incircle_gemm`] on Apple **AMX** (`cblas`) | ~24 G tests/s | matrix coprocessor at function-call latency; f32 GEMM + i128 fallback |
//! | [`incircle_gemm`] in **WASM** (simd128) | ~6–7 G/s | portable CPU-SIMD, no dispatch; inner loop vectorizes |
//! | [`incircle_gemm`] portable Rust (this crate) | ~0.6 G/s | auto-vectorized matmul |
//! | scalar i128 tight loop | ~0.17 G/s | correct but scalar, one test at a time |
//!
//! The lesson from benchmarking this across 13 backends (AMX / WASM / CPU-SIMD / Metal /
//! MLX / wgpu / ANE / CUDA / ROCm / Vulkan / XDNA): for a **cheap, K=4 predicate** the
//! winner is whichever engine has the *least invocation overhead* — a low-dispatch matrix/
//! SIMD unit (AMX, WASM, CPU), **not** a driver-gated accelerator (discrete CUDA/ROCm lose
//! to their own host CPU; the ANE is dispatch-bound). Accelerators win on *arithmetic
//! intensity and reuse* (the dense leaf), which this predicate doesn't have. See the
//! [`incircle_gemm`] module docs for the full analysis.

pub mod delaunay32;
pub mod fastest;
pub mod flip;
pub mod incircle_gemm;
pub mod predicates;
pub mod topology;
pub mod triangulate;
pub mod voronoi;

pub use fastest::{Backend, triangulate_fastest};
pub use flip::{flip_all_convex_once, flip_to_delaunay, hull_seed, interior_quads};
pub use predicates::{FAST_COORDINATE_SPAN, MAX_COORDINATE_SPAN, PredicateWidth};
pub use topology::{NO_NEIGHBOR, convex_hull, triangle_adjacency};
pub use triangulate::{GeoError, release_scratch, triangulate, triangulate_dwyer, triangulate_par};
pub use voronoi::{voronoi_dual, voronoi_grid_exact};

#[cfg(feature = "cpu")]
pub mod ops;
#[cfg(feature = "cpu")]
pub use ops::{GEO_DELAUNAY, GEO_VORONOI_GRID, register_geo_ops};

#[cfg(feature = "gpu")]
pub mod construct_gpu;
#[cfg(feature = "gpu")]
pub mod flip_gpu;
#[cfg(feature = "gpu")]
pub mod gdel_gpu;
#[cfg(feature = "gpu")]
pub mod leaf_gpu;
#[cfg(feature = "gpu")]
pub mod predicates_wgsl;
// The Voronoi GPU kernel registers against the CPU op registry and uses the
// rlx-wgpu device helper — bundled behind the `gpu-ops` feature.
#[cfg(feature = "gpu-ops")]
pub mod wgpu_kernels;

/// Register all rlx-geo IR extensions and kernels. Idempotent-ish (re-register
/// warns). Call once before compiling graphs that reference the geo ops.
#[cfg(feature = "cpu")]
pub fn register() {
    ops::register_geo_ops();
}
