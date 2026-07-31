# rlx-geo

Exact integer 2D **Delaunay triangulation** and discrete **Voronoi** diagrams for
RLX, exposed as `Op::Custom` with per-backend kernels. Sibling to `rlx-umap`
(same numerics-crate → backend-registry pattern).

## Layers

| Layer | Module | rlx deps | Status |
|---|---|---|---|
| Pure geometry | `predicates`, `triangulate`, `voronoi` | none | ✅ tested (5/5) |
| RLX ops (CPU) | `ops` (feature `cpu`, default) | `rlx-ir`, `rlx-cpu` | ✅ compiles clean |
| Native GPU kernel | `wgpu_kernels` (feature `gpu`) | `rlx-wgpu` | ✅ **on-device validated (Metal)** |
| Exact GPU predicates | `predicates_wgsl` (feature `gpu`) | — | ✅ **on-device validated (Metal)** |

Build/validate, all green:

```sh
cargo test  -p rlx-geo --no-default-features         # geometry core: 5/5
cargo check -p rlx-geo --features cpu                 # CPU ops
cargo run   -p rlx-geo --example gpu_validate \
            --features gpu                            # dispatches WGSL on the GPU:
#   voronoi_grid: GPU == CPU  (48x32 grid, 8 sites, 1536 cells) OK
#   orient2d:     GPU == CPU  (4096 random triples, exact i32) OK
#   in_circle:    GPU == CPU  (4096 random quads, emulated i64) OK
```

## Ops

| Op | Inputs | Attrs | Output | Kernels |
|---|---|---|---|---|
| `geo.delaunay` | `points [n,2] I32` | — | `[2n,3] I32` triangles, CCW; unused rows `[-1,-1,-1]` | CPU |
| `geo.voronoi_grid` | `sites [n,2] I32` | `{width,height}` | `[h,w] I32` nearest-site labels (`-1` empty) | CPU + wgpu |

Delaunay output uses a fixed `[2n,3]` upper-bound buffer (a triangulation has
`< 2n` triangles) so shape inference stays static; the caller reads rows until the
`-1` sentinel.

## Fastest path — backend auto-dispatch

`triangulate_fastest(points) -> (tris, Backend)` picks the fastest *available*
backend for the input:

- `n < 50 000` → **CPU serial** (tuned D&C wins; thread/dispatch overhead isn't
  worth it),
- `n ≥ 50 000` → **CPU parallel** (`triangulate_par`: `std::thread` chunk-build +
  merge),
- **GPU is not auto-selected for host-resident points** — the flip loop is
  latency-bound (per-round sync) and loses to CPU D&C once transfer is counted;
  use `triangulate_on_gpu` explicitly when the points already live in VRAM.

**NPU / ANE / TPU are excluded by design, not as a TODO.** Delaunay is irregular
pointer-chasing with exact-integer branch decisions; those accelerators are
fixed-function dense-tensor (matmul/conv) units with no way to run branchy integer
graph code — there is nothing to offload. See `src/fastest.rs`.

Measured on Apple Silicon (`geo_fastest`, the auto-picked path, vs C++
`delaunay32` parallel; ms, domain 29 000):

| N | C++ par | **geo_fastest** | ratio |
|---|---|---|---|
| 50k  | 2.89 | **1.95** | **0.68× (1.5× faster)** |
| 100k | 3.32 | **3.24** | **~tied** |
| 200k | 4.80 | **6.30** | 1.31× |
| 500k | 11.3 | **13.5** | 1.19× |
| 1M   | 19.1 | **25.9** | 1.36× |

`geo_fastest` **beats the tuned C++ up to ~100k** and is 1.2–1.4× at 200k–1M —
down from 3.3–7.6× before optimization. The parallel path's `prepare` is a
**parallel counting/bucket sort** (per-thread histograms + disjoint scatter,
prepare 7 → 4.85 ms at 1M); the build (~54%, ~92% parallel-efficient) is the
memory-latency-bound floor.

### How the parallel path was sped up (profiled at 200k)

`prepare` (sort) → `par build` (chunks) → concat → trunk merge → `export`. The build
parallelizes ~7× (47 → 6 ms); the serial tails were the target:

- **`export` parallelized** — each thread scans a disjoint dart range and emits a
  face only from its minimum dart (no dupes, no coordination): 5.2 → 1.0 ms.
- **`prepare` parallelized** — an x-**bucket sort** (buckets partition the x-range
  so they're globally ordered → no merge), with contiguous bucket-groups sorted
  in parallel via `split_at_mut`: 5.0 → 1.2 ms. (A chunk-sort + heap merge was
  tried first and was *slower*, so it was reverted.)
- **Free list for deleted darts** — the arena was append-only and grew to ~18.5
  darts/point even though a planar triangulation only has ≤ 6 alive at once.
  Reusing deleted slots dropped it to **6.0 darts/point** (a 3× smaller arena,
  now *below* the C++'s ~8.1), which is a pure DRAM/cache win: at 1M the build
  went 29 → 24 ms, concat 5 → 2.4 ms, export 6 → 5.5 ms. Reserve right-sized to
  8/point.

- **Dwyer alternating-cut build** (`build_dwyer`) — replaced the vertical-strip
  recursion (tall pieces → long, scattered merge fronts) with **Morton
  alternating x/y cuts** (compact pieces → far fewer cache misses). The same
  `merge` serves both axes, fed x-extreme hull edges for vertical cuts and
  y-extreme for horizontal (directional-hull tracking via an `lnext` hull walk) —
  exactly as the C++ reference does. Serial: 1M build 256 → 206 ms. Used per-chunk
  in the parallel path too. Fast path only (16-bit Morton); the wide path keeps
  the x-cut build. Validated against the reference (`tests/dwyer.rs`).

- **Shared-arena concat + parallel trunk merge** — profiling the old
  `concat+trunk merge` line as one number hid where the cost was: at 1M the copy
  was only ~0.75 ms while the *serial* left-fold trunk merge was ~2.2 ms. The
  copy now scatters each piece into a disjoint window of one shared buffer in
  parallel (the `bucket_sort` raw-pointer pattern). The merge became a **balanced
  tree reduction** (`merge_tree_par`): each level stitches adjacent pairs
  concurrently (odd tail carries up), halving the serial chain. All sibling
  merges share one dart buffer — new edges bump-allocate from a single
  `AtomicU32` cursor, deletes go on a per-merge free list, and a barrier between
  levels orders the dependents; since siblings touch disjoint sub-triangulations
  and claim disjoint cursor slots, the shared raw pointer is only ever
  dereferenced at non-overlapping addresses (sound, lock-free). Trunk merge 1M:
  **2.2 → 0.85 ms** (~2.6×). Validated race-free (25× full-Delaunay validation +
  serial/parallel count parity at 200k/1M); bump high-water stays at ~6.0
  darts/point, far under the reserved cap.

With these, `geo_fastest` **beats tuned C++ at ≤ 50k**, **crushes the fastest
public Rust library `delaunator` (S-hull) ~9× at 1M** (it's single-threaded), and
is **~1.5× behind C++ at 1M** — down from 3.3–7.6× at the start. The residual gap
is now almost entirely the memory-latency-bound `par build` (~15 ms at 1M); an
AoS dart layout was tried and measured neutral, confirming the build is at the
cache-miss frontier rather than a layout or serial-tail problem.

So the auto-dispatched CPU-parallel path went from ~3.3× behind to **≤ 1× at 50k
and ~2× at 1M**, ~4× faster than the serial port, and orders of magnitude faster
than the flip loop for host-resident data.

```rust
rlx_geo::register();          // once per process, before compiling graphs
// then reference GEO_DELAUNAY / GEO_VORONOI_GRID as Op::Custom in a graph,
// or call the library directly:
let tris = rlx_geo::triangulate(&[[0,0],[100,0],[50,80]]);
let vor  = rlx_geo::voronoi_grid_exact(&[[1,4],[8,4]], 10, 8);
```

## Per-backend story

- **CPU** — both ops run natively (exact Guibas-Stolfi Delaunay; brute-exact
  Voronoi). Verified.
- **wgpu** — native WGSL kernel for `geo.voronoi_grid` (one dispatch, one thread
  per cell, exact nearest via `bitcast<i32>` over the `array<f32>` arena). Source
  follows the fixed `WgpuGpuKernel` binding convention; run it on a real device
  before production use.
- **Metal / CUDA / ROCm / MLX** — add a native kernel by implementing that
  backend's public `Kernel` trait (`rlx_metal::op_registry::MetalKernel`, etc.)
  inside this crate and calling its `register_*` from `register_geo_ops`, exactly
  as `wgpu_kernels.rs` does. No backend depends on `rlx-geo`; the coupling is the
  `Op::Custom("geo.*")` string + the per-backend registry.

## Delaunay via edge-flip (`flip`)

A parallel independent-set Lawson flip loop that drives **any** valid
triangulation to Delaunay (`flip_to_delaunay`). Each round: build edge→(triangle,
apex) adjacency, mark interior edges that are convex **and** illegal (opposite
apex inside the circumcircle — the exact `in_circle`), take an independent set
(each triangle in ≤ 1 flip), flip all at once. Adjacency is rebuilt per round, so
a flip only rewrites its two triangles — no pointer surgery, which is exactly what
makes the round a race-free data-parallel kernel.

Validated (`tests/flip.rs`): seed = GS Delaunay scrambled by one convex-flip round
into a valid **non-Delaunay** mesh (confirmed to contain illegal edges); the loop
restores a valid Delaunay mesh (empty-circumcircle, same triangle count, zero
illegal edges) across small, wide-span, and 400-point cases.

### On the GPU — the whole loop runs on-device

`flip_gpu::flip_to_delaunay_gpu` runs the **entire** flip loop on the GPU. The
triangle buffer stays resident across rounds; each round is three compute passes —
`reset`, `mark`, `apply` — and the host only reads back a 4-byte "flips this round"
counter to decide when to stop. All validated on Metal by `gpu_validate`:

```
voronoi_grid: GPU == CPU  (seed step)
orient2d:     GPU == CPU  (4096 triples, exact i32)
in_circle:    GPU == CPU  (4096 quads, emulated i64)
flip marking: GPU == CPU  (863 interior edges of a real mesh)
flip loop:    GPU end-to-end -> Delaunay  (250 pts scrambled -> valid Delaunay)
```

The loop (`flip_gpu.rs`), per round:
- **`build_hash` + `resolve_twins`** find each edge's twin triangle in **O(T)**
  via a GPU hash table (race-safe `atomicCompareExchangeWeak` open addressing) —
  not the earlier O(T²) scan.
- **`mark`** does an O(1) twin lookup, tests convex + illegal with the exact
  `orient` / emulated-i64 `in_circle`, and stakes an independent-set claim via
  `atomicMin` on both triangles.
- **`apply`** fires a flip iff it owns *both* its triangles (writes never
  conflict); adjacency is rebuilt next round, so there's no pointer surgery.
- Output is a valid manifold Delaunay mesh (empty-circumcircle, correct count),
  checked against the CPU reference — validated at **~3000 triangles**.

Valid for spans ≤ **29 609** (the `in_circle` i64 bound) and < 65 536 points
(16-bit edge keys); wider needs 128-bit limbs / 64-bit keys (the `rlxsl`
integer-prelude direction).

### Seed: JFA-Voronoi dual extraction

`voronoi_dual` extracts Delaunay triangles from a Voronoi label grid: at each
interior grid vertex, a 2×2 block with exactly three distinct labels ⇒ that
triangle. Fed by the on-device `geo.voronoi_grid` kernel, this is the GPU seed
path. Measured (`tests/dual.rs`, and `gpu_validate`'s `dual seed` line):

- **precision 1.000** — every extracted triangle is a genuine Delaunay triangle,
- **recall ~0.90** — it recovers the interior exactly but **misses hull
  triangles**, whose circumcenters (Voronoi vertices) fall outside any finite grid
  (for near-collinear hull points the circumcenter → ∞). This is a fundamental
  grid-dual limitation, not a resolution knob.

So the dual is an exact **interior** seed but misses the hull.

### Fixing the hull: a complete seed

`hull_seed` produces a **complete** valid triangulation — hull triangles included
— by an incremental convex-hull sweep: sort by (x,y); each new point is outside
the current hull, so connect it to every hull edge it sees. It's generally not
Delaunay, but it covers every point up to the convex hull. Feed it to the flip
loop and you get the complete Delaunay:

```
hull_seed -> GPU flip -> COMPLETE Delaunay  (2983 tris, all 1500 pts) OK
```

Validated (`tests/flip.rs::hull_seed_completes_to_delaunay` and `gpu_validate`):
the flipped result is a valid Delaunay mesh (empty-circumcircle), references
**every** point (hull included), and has the same triangle count as the reference
Guibas-Stolfi triangulation. That's the piece the grid dual couldn't supply.

The two seeds are complementary: `voronoi_dual` recovers the interior on the GPU
(exact, parallel) but not the hull; `hull_seed` is a complete robust seed. A
production path can warm-start from the dual and complete the hull, or just use
`hull_seed`; either way the GPU flip loop perfects it.

`geo.delaunay` (the registered op) uses the exact serial CPU triangulator;
`flip_to_delaunay` (CPU) / `flip_to_delaunay_gpu` (GPU) are the flip paths;
`hull_seed` + `voronoi_dual` are the seed builders.

## Exactness

Predicates are exact for coordinate spans up to `MAX_COORDINATE_SPAN` (≈1.94e9):
i64 orientation always; in-circle in i64 (span ≤ 29 609) or i128. Integer
arithmetic is deterministic, so results are bit-identical across runs.
