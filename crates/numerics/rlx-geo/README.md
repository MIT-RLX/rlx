# rlx-geo

Exact integer 2D **Delaunay triangulation** and discrete **Voronoi** diagrams for
RLX, exposed as `Op::Custom` with per-backend kernels. Sibling to `rlx-umap`
(same numerics-crate → backend-registry pattern).

## Layers

| Layer | Module | rlx deps | Status |
|---|---|---|---|
| Pure geometry | `predicates`, `triangulate`, `voronoi` | none | ✅ tested (5/5) |
| RLX ops (CPU) | `ops` (feature `cpu`, default) | `rlx-ir`, `rlx-cpu` | ✅ compiles clean |
| On-device flip pipeline | `flip_gpu` (feature `gpu`) | — (only `wgpu`) | ✅ **exact on Metal (Apple) + Vulkan (NVIDIA/Intel/llvmpipe)** |
| Exact GPU predicates | `predicates_wgsl` (feature `gpu`) | — (only `wgpu`) | ✅ **on-device validated (Metal)** |
| Native GPU voronoi op | `wgpu_kernels` (feature `gpu-ops`) | `rlx-wgpu` + `cpu` | ✅ **on-device validated (Metal)** |

Build/validate, all green:

```sh
cargo test  -p rlx-geo --no-default-features         # geometry core: 5/5
cargo check -p rlx-geo --features cpu                 # CPU ops
# Standalone on-device flip + f32-filter A/B — needs only wgpu, no rlx-wgpu:
cargo run   -p rlx-geo --example flip_gpu_bench \
            --no-default-features --features gpu --release -- pts.bin
# Full GPU validation (incl. the voronoi op) needs the rlx-wgpu-backed gpu-ops:
cargo run   -p rlx-geo --example gpu_validate \
            --features gpu-ops                        # dispatches WGSL on the GPU:
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

- `n < 8 000` → **CPU serial** (tuned D&C wins; below the measured crossover the
  partition/merge overhead isn't worth it),
- `n ≥ 8 000` → **CPU parallel** (`triangulate_par`: `std::thread` chunk-build +
  merge). The persistent worker pool drops the per-call spawn cost, so the
  serial→parallel crossover measures at **~6.8k** sites (idle x86: ~5k; loaded ARM:
  ~7k) — far below the old 50k default, which had left [8k, 50k] on the serial path
  at up to **2–4× the parallel cost**. `GEO_PARALLEL_MIN` overrides it for tuning.
- **GPU is not auto-selected for host-resident points** — the on-device Lawson
  flip converges in ~30 rounds and runs ~15–30 ms up to 50k points, but the CPU
  D&C finishes the same input in single-digit ms, so it wins once transfer is
  counted; use the GPU flip explicitly when the points already live in VRAM. The
  WGSL predicates are **exact over the full certified span** (i64 orient, i64-inner
  / **i128** in-circle emulated in `vec4<u32>`, mirroring the CPU `PredWide`) — an
  earlier i32-inner version silently oscillated for spans > ~32k. `gpu_validate`
  checks the flip output is complete, illegal-free Delaunay at spans up to
  `MAX_COORDINATE_SPAN`.

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
  in the parallel path too, at **every** coordinate span: the Morton code is now a
  **full 64-bit interleave of the 32-bit coordinates** (`morton_spread`), so it is
  injective over the entire i32 range — not just ≤16-bit — which is what
  `equal key ⟺ equal point` (the dedup + alternating-cut invariant) requires.
  Both predicate widths (`PredFast` i64, `PredWide` i128) run the same Morton/Dwyer
  build; only the arithmetic differs. Previously any span > 29 609 fell back to the
  plain x-cut Guibas-Stolfi `run`, which was **~4× slower** — the benchmark domain
  (100 000) hit exactly that fallback. Validated against the reference
  (`tests/dwyer.rs`) and by exact `i128` `validate_scale` at 1M/2M.

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

- **Work-stealing over heterogeneous cores** — profiling showed `build_dwyer`
  only reached ~54% parallel efficiency (7.6× on 14 threads). The cause is core
  heterogeneity: an Apple M4 Pro is **10 P-cores + 4 E-cores**, and with one
  equal chunk per thread the 4 E-core chunks straggle while the P-cores idle.
  Fix: cut **more chunks than threads** (`GEO_CHUNK_MUL`×, default 4 — re-tuned
  from 3 once the per-op cost dropped; smaller chunks are more L2-resident and
  balance the P/E cores better, worth ~4–9% on the build across 200k–1M) and hand
  them out from an `AtomicUsize` work-stealing cursor — fast cores pull more
  chunks, slow cores fewer. Build efficiency rose to ~79%: **1M build 14.8 → 11.8
  ms**. The same pattern applied to `export` (finer ranges, work-stealing cursor):
  **export 5.9 → 4.85 ms**. The parallel trunk merge already balances via the
  shared cursor. (On homogeneous x86 the oversubscription is neutral-to-helpful —
  it also absorbs OS jitter; tune with `GEO_CHUNK_MUL`.)

- **In-place leaf sort (build allocation cut)** — `build_dwyer` recurses on
  Morton-bit cuts down to leaves it can no longer bisect (~m/3 leaves at 1M for
  random input). Each leaf was allocating a fresh tiny `Vec` (`pos[lo..hi]
  .to_vec()`) just to sort ~2–3 ids — hundreds of thousands of allocations.
  Leaves own disjoint `pos` ranges, so `build_dwyer` now takes `&mut pos` and
  sorts each leaf **in place**: **1M par build 11.8 → 10.9 ms** (~8%).

- **Orientation-free export** — `export` emitted a face only if its min-dart's
  three vertices were CCW (`orient3`, three scattered coord reads per face, ~2 M
  faces at 1M). But *every bounded face is already CCW* by the Guibas-Stolfi
  invariant; the only CW face is the unbounded outer one, and it's a triangle
  **iff the convex hull is a triangle**. So the build hands export one hull dart,
  export finds the outer triangle's min-dart once (O(1)) and skips it, and the
  per-face `orient3` is gone: **export ~5–6 % faster** (1M 4.8 → 4.6 ms). Output
  is byte-for-byte identical (exact 1M validation + the 3-point hull tests pass).

- **3-pass Morton radix** — the per-chunk sort now does three 11-bit passes
  instead of four 8-bit (25 % less data moved; the 2048-entry histogram still fits
  L1): **build+dedup 2.7 % faster at 1M** (thermal-matched A/B).

- **16-point Morton leaf** — profiling the residual gap against the reference C++
  turned up its `kMortonLeafSize = 16`: it stops the Morton recursion at 16-point
  leaves and finishes them with the x-cut D&C. Ours recursed Morton-cuts to 2–3
  points, and *every* Morton merge must `scan_dhulls` (walk the full outer hull)
  to recover the four directional extremes — whereas the x-cut merge tracks
  `(le, re)` for free. Pushing the many small bottom-level merges into
  `delaunay_slice` cuts total hull-scan work ~6× for another **2.7 % on
  build+dedup** (contention-matched A/B; 16 beat 8/32 — bigger leaves re-sort more
  than they save in scans).

- **Lifted `in_circle`** — C++ precomputes a paraboloid lift `(x-mnx)²+(y-mny)²`
  per point and forms the in-circle determinant as `ap = lift[a]−lift[d]` instead
  of recomputing `ax²+ay²` each call: **9 multiplies vs 15** (translation-
  invariant; same `12·S⁴ ≤ 2⁶³` i64 span bound). Even with a *separate* lift array
  (one extra read per vertex), it's **4.4 % faster on build+dedup** —
  which proves the merge's `in_circle` had compute-latency slack in the multiply
  chain, not pure memory-latency. Correct: exact 1M validation + all tests pass.
  The lift is packed into `coord[2]` (`[i32; 3] = [x, y, lift]`) so it shares a
  cache line with `x`/`y` and needs no separate array — though A/B showed that
  packing is perf-*neutral* vs a separate lift array (4.3% vs 4.4% benefit): the
  extra reads were cheap L2 hits, so the win really was the multiply count.

- **`deleted`-flag in the merge** — also from reading the C++. After each rising-bubble prune loop, the candidate's validity
  (`orient3`) must be re-tested *only if the loop actually advanced the
  candidate*; if nothing was pruned it's unchanged and still valid. We were
  re-testing unconditionally — an extra `orient3` (3 coord reads + a determinant)
  on every non-pruning outer merge iteration, the common case. Guarding the
  re-test with a `deleted` flag: **build+dedup 4.6 % faster at 1M**
  (contention-matched A/B). C++'s `connect`/`splice`/`delete_edge`/`in_circle`
  were otherwise identical to ours (its only structural difference is an SoA dart
  layout vs our AoS — measured neutral here).

- **Eliminating prepare's redundant double-sort** — the biggest recent win.
  Profiling `prepare` split it into: bucket scatter 0.3 ms, **within-bucket sort
  2.9 ms**, serial dedup 1.1 ms. But the build immediately **re-sorts each chunk
  by Morton code**, throwing prepare's `(x,y)` order away — the only reason for
  the full sort was to make duplicates adjacent for dedup. Key facts: the trunk
  merge needs chunks only *x-separable*, not fully x-sorted; and a Morton code is
  a bijection on the rebased coordinates (64-bit interleave → injective at any
  span), so **equal key ⟺ equal point**.
  So the fast parallel path now (`run_par_fast`): partitions points into
  x-separable *whole-bucket* chunks with the scatter only (no within-bucket sort),
  then **dedups inside each chunk during the Morton sort it already does** (equal
  keys are adjacent; keep the lowest original index). Because equal x → same
  bucket → same chunk, every copy of a point lands together, so per-chunk dedup is
  complete and chunk boundaries stay strictly x-separated. Result: **prepare 4.8 →
  0.54 ms**, and 1M total **22.8 → 19.5 ms**. Pathological input (a chunk of <2
  distinct points) falls back to the robust dedup-first path; the wide/i128 span
  now runs this same fast path (`run_par_fast::<PredWide>`) rather than the plain
  fallback. Validated: exact serial-vs-parallel agreement (modulo cocircular
  ties) at 1M/2M via `examples/validate_scale`.

- **Serial-only Morton coord compaction** — on the *serial* path (`run_dwyer`),
  permuting `coord`/`orig` into Morton order before the build makes the
  latency-bound `in_circle`/`orient` lookups sequential instead of gathering from
  the 8 MB x-sorted array: **serial build_dwyer 112 → 105 ms** (~6%). It is
  *deliberately not* applied per-chunk in the parallel path: each 42-way chunk
  (~24k pts / 192 KB) is already L2-resident, so the gather+remap measured
  neutral-to-negative there — a clean example of an optimization whose value
  depends entirely on the working-set size.

With these, `geo_fastest` now **beats tuned C++ at every size** — measured
contention-matched (interleaved best-of, the box shared with other load) at
`domain = 100 000`: **50k 0.56×, 200k 0.80×, 500k 0.91×, 1M 0.95×, 2M 0.97×** —
and **crushes the fastest public Rust library `delaunator` (S-hull) ~9× at 1M**
(it's single-threaded) — down from 3.3–7.6× at the start.

Two fixes closed the last (and largest) gap, both found by discovering the
benchmark's `domain = 100 000` never touched the fast path:

- **64-bit Morton → fast path at any span.** The Morton code was masked to 16 bits,
  so any span > 29 609 (the i64 predicate bound) fell back to the plain x-cut
  Guibas-Stolfi `run` — **~4.6× slower serial**. Widening `morton_spread` to a full
  32-bit-per-axis / 64-bit interleave (still injective, so dedup stays correct) lets
  **both** predicate widths run the Morton/Dwyer build. The radix sort adds passes
  only when the key magnitude needs them, so the ≤16-bit case is unchanged (3 passes).

- **i64-inner wide `in_circle`.** `PredWide` evaluated the whole 3×3 lifted
  determinant in i128 (15 wide multiplies). But `MAX_COORDINATE_SPAN` (1.94e9) is
  chosen so `2·span² < i64::MAX` — every inner term (the paraboloid lifts and the
  three 2×2 minors) fits i64; only the final 3-term accumulation can overflow. So
  the inner 12 multiplies are now i64 and just 3 widen to i128: **serial 2M
  build_dwyer 392 → 261 ms**, dropping the serial gap from 2.64× to a flat ~1.6×
  and the parallel 2M ratio from 1.27× to 0.97×.

The residual serial gap (~1.6×) is the memory-latency-bound build inner loop; AoS
dart layout and per-chunk coord compaction were both tried and measured neutral
(chunks are already L2-resident). Parallelism (persistent pool, below) recovers it.

### Ablation

Leave-one-out on an **idle Linux** box (`geo_fastest`, best-of-6; each row disables
one optimization via its `GEO_ABL_*` / `GEO_NO_REUSE` gate — the ratio is that
optimization's speedup):

| N | baseline | −64-bit-Morton/Dwyer | −i64-inner predicate | −persistent pool | −buffer reuse |
|----|----------|----------------------|----------------------|------------------|---------------|
| 200k | 5.0 ms | **2.20×** | 1.14× | 1.00× | 1.00× |
| 1M | 32.0 ms | **1.90×** | 1.10× | 1.02× | 1.09× |
| 2M | 60.0 ms | **2.02×** | 1.11× | 1.04× | 1.09× |

The 64-bit-Morton/wide-Dwyer dispatch is the dominant lever (≈2×) — it is what
moved the `domain = 100 000` benchmark off the slow plain-GS fallback. The i64-inner
predicate is a flat ~10%. Buffer reuse is ~9% at scale (and platform-split, below).
The persistent pool looks small on an *idle* box — its real value is on a *loaded*
one, where spawning ~14 threads per call contends with everything else (that's the
regime the shared benchmark machine is actually in). Gates are wired for
reproducibility; `abl_slowpred` is a build feature, the rest are env vars.

- **Visited-bit export** — the last C++ trick, and the one that pushed 1M *past*
  C++. Our export tested `lnext(e2)==e` on every dart to confirm the face is a
  triangle — one extra scattered `lnext` read across all ~6M darts. Instead, mark
  the outer (unbounded) face's darts once (serial, O(hull)) via the high bit of
  `org` (deleted darts, `org==u32::MAX`, already have it set), then the scan skips
  them with a single test and emits every remaining face with **no triangle
  check** — Delaunay guarantees every bounded face is a triangle. **Export 9.9%
  faster (0.46 ms) at 1M**. Combined with re-tuning the chunk multiplier to 4,
  this pushed 1M to **0.94× vs tuned C++** — a clear ~6% win (we now beat it at
  50k, 200k, and 1M). Correct: exact 1M validation + tiny-hull (h=3) tests pass.

- **Persistent thread pool** — the C++ reference reuses one `WorkerTeam` across
  triangulations; ours spawned ~14 fresh OS threads *per call* (across ~11 phases).
  On an idle box that's ~1.8 ms fixed overhead; on a **loaded** box it is far worse
  (spawning threads contends with everything else). A thread-local `Pool` — workers
  spawned once, parked on a condvar when idle, reused across every call and phase
  (`for_each` work-steals via an atomic cursor) — removes it. Cross-thread closures
  ride a lifetime-erased `FnRef` that is sound because `for_each` blocks until every
  worker stops dereferencing it; validated under Miri via `parallel_raw_pointer_path`.

- **Thread-local buffer reuse** — the other half of matching C++'s persistent
  `Triangulator`. Every large allocation is pooled per thread (`Scratch`) and handed
  back after each call (capacity retained): the ~144 MB concat dart arena at 2M, the
  ~32 MB `coord`/`orig` scatter buffers (filled by uninit `set_len` — the scatter is
  a bijection onto `[0,n)` — instead of a zero-filled `vec!`), and the ~50 per-chunk
  build buffers (~144 MB total, distributed to pool workers by index via `mem::take`
  and reclaimed after the concat). The win is **complementary across platforms** —
  measured with a `GEO_NO_REUSE=1` in-process A/B, immune to load drift:
    - **Linux/glibc:** the arena/scatter reuse gives **~6–11%** (1M 35.5 → 31.6 ms,
      2M 65.5 → 61.3 ms). The per-chunk pool is redundant here — glibc's *adaptive*
      mmap threshold already recycles those ~2.9 MB buffers after warmup — but the
      144 MB arena always mmaps, so pooling it is the lever.
    - **macOS:** inverted — libmalloc caches the big freed regions (arena reuse
      neutral), but the ~50 per-call malloc/free of the chunk buffers cost real
      bookkeeping, so the per-chunk pool is what pays: **~7–8%** (2M 39.1 → 36.2 ms).

  Neither platform regresses. Validated exact at 1M/2M and under Miri (the uninit
  `set_len` and the cross-thread `mem::take` buffer hand-off).

## Topology helpers & robustness

Beyond the triangle list, [`triangle_adjacency`] returns each triangle's three
neighbor triangles (`NO_NEIGHBOR` on the hull — the halfedge structure most
algorithms want), and [`convex_hull`] returns the CCW hull vertices (exact
monotone chain). Both use exact `i128` arithmetic and take indices into the
caller's arrays. `examples/validate_scale` independently re-checks the full 1M
output (CCW + manifold + empty-circumcircle, exact `i128`) — it is a valid
Delaunay triangulation; serial and parallel differ only at cocircular points,
where the triangulation is genuinely non-unique and both choices are correct.

Not pursued (measured or analyzed dead-ends): the i64 fast-path span (29 609) is
already at the exact i64 `in_circle` overflow limit (`12·S⁴ ≲ 2⁶³`), so it can't
widen without i128; software prefetch has no portable stable-Rust intrinsic on
aarch64; f64 input (adaptive predicates) and constrained Delaunay are larger
features left for later.

### Unsafe & API hygiene

The concurrent build/merge lean on raw pointers; three things keep that honest:

- **One merge, not two.** The Guibas-Stolfi merge and all navigation / predicates
  live once as default methods on a `DartStore` trait; the serial arena and the
  concurrent trunk-merge context (`MergeCtx`) each supply only ~9 storage
  primitives. Darts are read/written **by value** (`[u32;3]` is `Copy`), so no
  `&mut`-from-`&` reference is ever formed into the shared buffer — the previous
  `mut_from_ref` footgun is gone. Dispatch is static (monomorphized, inlined), so
  it is perf-neutral.
- **Provenance-preserving `Send`.** Cross-thread pointers use a `SendPtr<T>`
  newtype instead of a `ptr as usize → usize as ptr` round-trip, so real pointer
  provenance is carried across the `thread::scope` boundary.
- **Miri-checked.** `internal_tests::parallel_raw_pointer_path` exercises the
  whole raw-pointer path (SendPtr scatter → shared-arena concat → tree merge) on a
  small input; `cargo +nightly miri test` runs it clean (no UB / aliasing /
  provenance / data-race), and the exact-`i128` `validate_scale` confirms the 1M
  output.

Triangulation entry points return `Result<_, GeoError>` (oversized coordinate
span is a recoverable error, not a panic).

### Memory

Peak footprint is ~360 B/site (≈716 MB at 2M): the dart arena (~6 darts/site ×
12 B) dominates, with the parallel path briefly doubling darts across the
build→concat boundary. To amortise re-allocation/page-faulting, the parallel path
keeps its large buffers (dart arena, scatter buffers, per-chunk pool) in a
per-thread pool — worth ~8–11% on repeated calls (Linux), but it retains hundreds
of MB after a large `n`. Call [`release_scratch`] to hand those buffers back to
the allocator after a big one-off triangulation (the next call re-allocates); it
affects only the calling thread. (Measured: RSS dropped by ~half immediately after
`release_scratch` in a repeated-call harness.)

So the auto-dispatched CPU-parallel path went from ~3.3× behind to **≤ 1× at 50k
and ~2× at 1M**, ~4× faster than the serial port, and orders of magnitude faster
than the flip loop for host-resident data.

```rust
rlx_geo::register();          // once per process, before compiling graphs
// then reference GEO_DELAUNAY / GEO_VORONOI_GRID as Op::Custom in a graph,
// or call the library directly:
let tris = rlx_geo::triangulate(&[[0,0],[100,0],[50,80]]).unwrap();
let adj  = rlx_geo::triangle_adjacency(&tris);      // neighbor triangles
let hull = rlx_geo::convex_hull(&[[0,0],[100,0],[50,80]]);
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
- **`mark`** does an O(1) twin lookup, tests convex + illegal, and stakes an
  independent-set claim via `atomicMin` on both triangles. The illegal test uses
  an **f32 static in-circle filter with exact fallback**: a cheap floating
  determinant plus a conservative error bound (≈6× Shewchuk's, covering the
  i32→f32 input rounding and GPU relaxed-float); when `|det|` clears the bound the
  sign is *certified* — bit-identical to the exact path — otherwise it falls back
  to the emulated-`i128` determinant. Correctness is one-sided (a looser bound
  only adds fall-throughs, never a wrong sign; overflow is safe because `perm ≥
  |det|`, so if `det` reaches ∞ the bound already has). On uniform data the
  fall-through rate is O(unit-roundoff) — `gpu_validate` counts **0–3 fall-throughs
  across ~90k edge-tests up to span 1.94e9**, so essentially every edge is decided
  by the f32 path while the output stays exact. Net effect on the flip loop:
  **~1.05–1.22× (best-of), largest at wide span** where the `i128` path costs most.
  `GEO_FLIP_NOFILTER` forces the all-`i128` path (for A/B).
- **`apply`** fires a flip iff it owns *both* its triangles (writes never
  conflict); adjacency is rebuilt next round, so there's no pointer surgery.
- Output is a valid manifold Delaunay mesh (empty-circumcircle, correct count),
  checked against the CPU reference — validated at **~3000 triangles**.

**Batching the round-loop.** Every convergence check reads back the flip counter,
which is a full GPU→CPU stall — the loop's real bottleneck. Since the number of
rounds to converge is unknown up front, the batch **grows geometrically** (2 → 4 →
8, `GEO_FLIP_BATCH`-capped): a near-Delaunay seed stops after a couple of rounds
without over-running a large batch of no-op rounds (a no-op round still rebuilds
the whole edge adjacency, so it's nearly full cost), while a far-from-Delaunay
seed needs only O(log rounds) stalls. Shrinking near the finish was measured
*worse* — a stall dwarfs a round, so minimizing stalls wins. Net vs the old
fixed-4 loop: **~2–4× faster** at 20k–50k points (e.g. 20k ≈ 39 → ~9 ms). It is
still latency-bound and loses to the CPU D&C for host-resident points; the win is
for data already resident on the device.

Exact for spans up to `MAX_COORDINATE_SPAN` (emulated i128) and any u32 point
count — the edge hash stores half-edge ids and recomputes keys on probe (no
16-bit-key cap), and all kernels use 2D dispatch (fixed x, `gid.y*65536+gid.x`)
so they scale past the 65 535-workgroups-per-dimension limit to millions.

### Full on-device construction (`insert_gpu::delaunay_gpu`)

Delaunay **from scratch on the GPU**, not just flipping: seed a CCW fan of the
convex hull (all real points — no super triangle, so coordinates stay in-span and
there is no bounding-triangle hull deficit), then **parallel incremental
insertion** of the interior points (per round: rebuild adjacency by hash → walk
each pending point to its containing triangle → `atomicMin`-claim the triangle(s)
it rewrites → 3-way / on-edge 2- or 4-way split), then the exact flip. Validated
**exact at 200/1k/5k/60k/200k/1M** (Metal + Vulkan). Because splits only touch
disjoint claimed/allocated triangles the round is race-free, and adjacency is
rebuilt each round so no incremental-adjacency bookkeeping is needed.

**Measured (idle NVIDIA RTX 3080 Ti, exact):** 60k ≈ 31 ms, 200k ≈ 117 ms,
1M ≈ 871 ms. That is **~16–24× slower than the 20-core CPU D&C** (2 / 6 / 37 ms) —
and the gap *widens* with n. This is the expected outcome: the construction is a
sequence of ~25 insertion + ~54 flip **synchronized full-mesh rounds**, whereas
the CPU D&C is one cache-friendly near-optimal pass; the workload is latency- and
branch-bound, exactly where a many-core CPU beats a GPU. Published GPU-DT work
beats *single-threaded* Triangle, not a tuned 20-core build. **Use the GPU path
for data already resident in VRAM (no host transfer); for host-resident points the
CPU wins at every scale.**

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
i64 orientation always; in-circle fully in i64 for span ≤ 29 609 (fast path,
precomputed paraboloid lift), otherwise the i64-inner / i128-accumulate form (only
the final 3-term sum widens to i128).

**Determinism.** Every result is an exact, valid Delaunay triangulation, but the
triangle *set* is bit-identical across runs only on the **serial** path
(`triangulate`). The **parallel** path (`triangulate_par` / the auto-dispatched
`triangulate_fastest` above `PARALLEL_MIN`) can resolve **cocircular ties**
differently run-to-run: `merge_tree_par` hands out seam-dart slots via an atomic
cursor whose order depends on thread scheduling, and at four cocircular points
either diagonal is a valid Delaunay edge. The triangle *count* is always invariant;
the choice of diagonal at a degenerate quad is not. Use the serial path if you need
byte-reproducible output.
