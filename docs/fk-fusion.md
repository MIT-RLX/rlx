# FKL-style region fusion

RLX implements patterns from [Fused Kernel Libraries](https://arxiv.org/abs/2508.07071) (FKL): fuse memory-bound preprocessing (resize, future crops) with element-wise chains, and horizontal batch fusion over identical regions.

## What the compiler does

| Pass | Effect |
|------|--------|
| `MarkElementwiseRegions` | Collapse single-consumer element-wise chains into `Op::ElementwiseRegion` |
| `MarkBatchSliceRegions` | Fold `Narrow` -> unary batch slices into per-slice `ElementwiseRegion` before batch fusion |
| `MarkTransformRegions` | Mark transform chains (`TransformRegion`) |
| `FuseRegionPrologue` | Fold `ResizeNearest2x` into a region as `prologue: ResizeNearest2x` |
| `FuseBatchPreprocess` | Merge parallel `ElementwiseRegion` slices + `Concat` into `BatchElementwiseRegion` |
| `DecomposeFusionRegions` | Lower `TransformRegion` / `BatchElementwiseRegion` to primitives (skipped when native FKL regions are kept) |
| `UnfuseElementwiseRegions` | Split plain regions to per-op thunks; **GPU keeps prologue regions** |

End-to-end flow for `resize -> relu -> add -> mul` on Metal/CUDA/wgpu/MLX:

1. Mark chain -> `FuseRegionPrologue` removes standalone resize.
2. GPU unfuse keeps `ElementwiseRegion { prologue: ResizeNearest2x, ... }`.
3. Backend runs fused region kernel (MSL / CUDA / WGSL / MLX) or MPSGraph (Metal).

## Developer controls

### Python (`pyrlx`)

```python
import pyrlx as rlx

opts = rlx.FusionOptions.native_fk()  # or FusionOptions(native_fk_regions=True)
compiled = rlx.Session(device="metal").compile_with(graph, fusion_options=opts)

# Batch slice + relu + concat (fuses to BatchElementwiseRegion when native_fk is on):
g = rlx.batch_narrow_relu_graph("batch", batch_n=2, channels=3, height=8, width=8)

# Graph builder: resize -> relu -> ...
up = graph.resize_nearest_2x(x)

# Deferred device selection:
fs = rlx.FlexibleSession()
compiled = fs.compile_with_resolved(
    graph, device="metal", fusion_options=opts, kernel_dispatch="native"
)
```

Env vars (`RLX_NATIVE_FK_REGIONS`, etc.) are still merged at compile time.

### Session / `CompileOptions`

```rust
use rlx_opt::{FusionOptions, FusionTarget};
use rlx_runtime::{CompileOptions, Session};

let mut opts = CompileOptions::new().fusion_target(FusionTarget::Metal);
opts.fusion_opts = FusionOptions {
    // Disable all FKL passes (mark transform, prologue, batch).
    fk_fusion: false,
    // Do not fold resize into regions.
    fuse_region_prologue: false,
    // Do not merge batch slices.
    fuse_batch_preprocess: false,
    // Keep BatchElementwiseRegion / TransformRegion in MIR (needs backend support).
    native_fk_regions: true,
    // Keep every ElementwiseRegion (not only prologue) through lowering.
    keep_elementwise_regions: true,
    // Force decomposition of FKL regions before backends.
    decompose_fusion_regions: true,
    // Split regions to primitives (CPU always; GPU default keeps prologue).
    unfuse_elementwise_regions: true,
    ..FusionOptions::default()
};
let _ = Session::new(device).compile_with(graph, &opts);
```

### Environment variables

| Variable | When set | Effect |
|----------|----------|--------|
| `RLX_NO_FK_FUSION=1` | compile time | Skips `MarkTransformRegions`, `FuseRegionPrologue`, `FuseBatchPreprocess` |
| `RLX_FUSE_REGION_PROLOGUE=0` | compile time | Disables resize->region folding |
| `RLX_FUSE_BATCH_PREPROCESS=0` | compile time | Disables batch slice fusion |
| `RLX_NATIVE_FK_REGIONS=1` | compile time | Skips `DecomposeFusionRegions`; keeps `TransformRegion` / `BatchElementwiseRegion` |
| `RLX_NO_NATIVE_FK_REGIONS=1` | compile time | Disables auto native FKL regions on GPU-class targets |
| `RLX_FK_BATCH_SINGLE_KERNEL=1` | compile time | CUDA/ROCm/Metal/wgpu: one `batch_elementwise_region` launch per `BatchElementwiseRegion` (no prologue; max 64 slices) |
| `RLX_DECOMPOSE_FUSION_REGIONS=1` | compile time | Always decompose FKL regions (overrides native) |
| `RLX_KEEP_ELEMENTWISE_REGIONS=1` | compile time | Skips unfuse; keeps all `ElementwiseRegion` nodes |
| `RLX_METAL_NO_FUSION=1` | compile time | Skips pattern fusion on Metal (existing) |
| `RLX_METAL_UNFUSE_REGIONS=1` | compile time | Unfuse all regions on Metal including prologue |
| `RLX_MPSGRAPH_TRACE=1` | compile time | Log MPSGraph lowering bail reasons |

Env values are merged in `FusionOptions::merge_env()` during `Session::compile()`.

### Default behavior by target

Session and `fusion_passes_for_supported` call `apply_native_fk_defaults`: Metal, CUDA, ROCm, wgpu, MLX, and TPU keep `BatchElementwiseRegion` / `TransformRegion` unless `RLX_NO_NATIVE_FK_REGIONS=1`. CPU always decomposes and unfuses (no native region executor).

| Target | Prologue regions | Plain `ElementwiseRegion` | `BatchElementwiseRegion` |
|--------|------------------|---------------------------|---------------------------|
| CPU | Decomposed (no native region executor) | Unfused to primitives | Decomposed to slice regions + concat |
| Metal / CUDA / ROCm / wgpu | **Kept** (native kernel / MPSGraph) | Unfused unless `RLX_KEEP_ELEMENTWISE_REGIONS=1` | **Kept** by default (native slice kernels; optional single launch) |
| MLX | Prologue via `resize_nearest_2x_nchw` | Unfused unless `RLX_KEEP_ELEMENTWISE_REGIONS=1` | **Kept** by default: per-slice chain + `concat` axis 0 |
| TPU | Native HLO chain (resize via broadcast+reshape) | Unfused unless `RLX_KEEP_ELEMENTWISE_REGIONS=1` | **Kept** by default: per-slice chain + `concatenate` |

GPU unfuse mode: `UnfuseElementwiseRegions::FOR_GPU` (`unfuse_prologue: false`).

## Optimizing kernels

### Region metadata (`rlx-ir/src/region_encode.rs`)

GPU kernels share a 150-word metadata buffer:

- Words `0..16`: input byte offsets (in f32 units)
- Words `16..144`: encoded `ChainStep` program (max 32 steps)
- Words `144..149`: prologue tag + NCHW output `(n,c,h,w)`
- Word `149`: `prologue_input` (external input index for the prologue transform)

Prologue tag `1` = `ResizeNearest2x` on **external input `prologue_input`** (half-res NCHW in -> 2x HxW out). Kernels use a 3D grid `(W, H, NxC)` when prologue is set.

Metal / CUDA / ROCm / wgpu session compile enables native FKL regions on GPU-class targets by default; TPU direct compile and Session use the same policy. Set `RLX_NO_NATIVE_FK_REGIONS=1` to force decomposition.

To tune performance:

1. **Metal**  `rlx-metal/src/kernels.rs` (`elementwise_region`), `mps_graph_lower.rs` (MPSGraph path), env `RLX_MPSGRAPH_TRACE` to see fallback to MSL thunks.
2. **CUDA / ROCm**  `rlx-gpu-kernels/kernels/elementwise_region.cu` and optional `batch_elementwise_region.cu` (set `RLX_FK_BATCH_SINGLE_KERNEL=1`), dispatch in `rlx-cuda` / `rlx-rocm` `backend/`.
3. **wgpu**  `rlx-wgpu/src/kernels/elementwise_region.wgsl` (`elementwise_region` vs `elementwise_region_spatial`).
4. **MLX**  `rlx-mlx/src/lower/` (`ElementwiseRegion` in `env.rs` + `ops::resize_nearest_2x_nchw`).
5. **TPU**  `rlx-tpu/src/lower.rs` (inline HLO chain; `ir_passes::prepare_graph_for_hlo` runs FKL before lowering). Direct `TpuExecutable::compile` and orchestrated HLO segments share the same path.

### Fusion pass tuning (`rlx-fusion`)

- `rlx-fusion/src/fk_fusion.rs`  `FuseRegionPrologue`, `FuseBatchPreprocess`, `DecomposeFusionRegions`
- `rlx-fusion/src/fusion_fragment.rs`  fragment registry for new transform -> prologue mappings
- `rlx-fusion/src/limits.rs`  `FusionLimits::GPU_NATIVE` caps (16 inputs, 32 steps)

Register a new prologue: extend `RegionPrologue` in `rlx-ir/src/op.rs`, `prologue_for_transform_op` in `fusion_fragment.rs`, `encode_prologue_tail`, and each backend kernel branch (GPU MSL/CUDA/WGSL, TPU HLO broadcast+reshape, MLX).

### Benchmarks

```sh
just throttle   # or RLX_ALLOW_THROTTLE=1
RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bench --release --example bench_fk_fusion --features metal
RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bench --release --example bench_fk_fusion --features tpu
FK_BENCH_OPS=1 RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bench --release --example bench_fk_fusion --features metal
```

Variants include `session_default_pipeline` (production `CompileOptions`),
`session_default_batch` (primitive narrow+relu+concat; default session keeps `BatchElementwiseRegion` on GPU-class + TPU),
`session_native_batch` (`native_fk_regions` on the same primitive graph), and pre-fused IR
(`skip_fusion` in bench opts). Set `FK_BENCH_OPS=1` to print post-pipeline op counts without timing.

Python timing demo: `crates/bindings/pyrlx/examples/fk_fusion_bench.py` (`--batch` for narrow+relu+concat).

### Session API (native batch)

```rust
let mut opts = CompileOptions::new().fusion_target(FusionTarget::Metal);
opts.fusion_opts.native_fk_regions = true;
opts.fusion_opts.decompose_fusion_regions = false;
// Session::compile_with runs FuseBatchPreprocess and keeps BatchElementwiseRegion
```

### Tests

```sh
cargo test -p rlx-fusion gpu_unfuse_preserves_prologue_region
cargo test -p rlx-compile --lib metal_default_unfuse
cargo test -p rlx-runtime --features cpu,metal,gpu,mlx --test fk_prologue_parity
cargo test -p rlx-runtime --features cpu,metal --test fk_prologue_parity fk_batch_session_default_matches_cpu_metal
cargo test -p rlx-runtime --features cpu,tpu --test fk_prologue_parity fk_batch_session_pipeline_keeps_native_region_tpu
cargo test -p rlx-tpu --test fk_pipeline
cargo test -p rlx-tpu --test hlo_match batch_elementwise_region elementwise_region_resize_prologue
cargo test -p rlx-metal --test mps_graph_batch_region_lower
cargo test -p rlx-metal --test mps_graph_prologue_region_lower
cargo test -p rlx-mlx --test basic batch_elementwise_region_matches_atomic
```

HIP-CPU batch kernel validate (Docker/linux-gnu only; clones into `rlx-cuda/docker/vendor/HIP-CPU`):

```sh
just test-hip-cpu-validate
# or inside the image: cargo test -p rlx-cuda --features hip-cpu-validate batch_elementwise_region
```

## MPSGraph vs native thunks (Metal)

When the full graph lowers to MPSGraph:

- `ElementwiseRegion` with `ResizeNearest2x` prologue uses `resize_nearest_nchw` + chain replay.
- `BatchElementwiseRegion` runs the same chain per batch input, then `concat` on axis 0.

### Native batch execution (CUDA / ROCm / wgpu / Metal / TPU)

When native FKL regions are kept in MIR, backends write each
slice into subranges of the packed output tensor (helpers in `rlx-ir/src/region_encode.rs`:
`batch_region_slice_shape`, `batch_region_slice_elems`, `batch_region_slice_dst_off_f32`).
No separate concat kernel.

| Backend | Default batch dispatch | Single launch |
|---------|------------------------|---------------|
| CUDA / ROCm | N `elementwise_region` kernels | `RLX_FK_BATCH_SINGLE_KERNEL=1` -> `batch_elementwise_region` (Z = num_batch; no prologue; max 64 slices) |
| Metal / wgpu | N `ElementwiseRegion` thunks / WGSL dispatches | `RLX_FK_BATCH_SINGLE_KERNEL=1` -> one `batch_elementwise_region` dispatch (workgroup Z = num_batch) |
| TPU | Per-slice HLO chain + `concatenate` | N/A (no single-launch batch kernel; one HLO module per slice chain) |
| MLX | Per-slice chain + `concat` | N/A |

If MPSGraph lowering returns `None`, Metal falls back to MSL thunks (`Thunk::ElementwiseRegion` with spatial dispatch for prologue).

## Limitations (current)

- Prologue resize uses `prologue_input` in region metadata (word 149); fusion still swaps the resize source to input 0 and remaps chain slots when needed.
- `BatchElementwiseRegion` requires equal slice shapes and concat on axis 0 (as produced by `FuseBatchPreprocess`).
- Primitive `narrow + relu + concat` is folded by `MarkBatchSliceRegions` before batch fusion when `fk_fusion` is enabled (see `batch_narrow_relu_primitive_graph` / `pyrlx.batch_narrow_relu_graph`).
- `RLX_FK_BATCH_SINGLE_KERNEL` applies on CUDA/ROCm/Metal/wgpu, only without resize prologue, and only for at most 64 slices.
- Plain `ElementwiseRegion` on wgpu may still unfuse by default (broadcast/modulus); use `RLX_KEEP_ELEMENTWISE_REGIONS=1` to retain.
- CPU has no fused region executor; chains are always unfused after marking.
- `prepare_graph_for_ad` runs `DecomposeFusionRegions` so training/AD never sees `BatchElementwiseRegion` / `TransformRegion` in the backward graph.
- HIP-CPU validation for `batch_elementwise_region` clones HIP-CPU under `rlx-cuda/docker/vendor/` and runs only via `just test-hip-cpu-validate` (Linux Docker; not a git submodule).

## Related files

| Area | Path |
|------|------|
| Fusion pipeline | `rlx-compile/src/fusion_pipeline.rs` |
| FKL passes | `rlx-fusion/src/fk_fusion.rs` |
| Shared test graphs | `rlx-fusion/src/fk_graphs.rs` |
| Session merge | `rlx-runtime/src/stages.rs` |
| Encode | `rlx-ir/src/region_encode.rs` |
| Parity | `rlx-runtime/tests/fk_prologue_parity.rs` |
| TPU HLO + FKL | `rlx-tpu/src/lower.rs`, `rlx-tpu/src/ir_passes.rs`, `rlx-tpu/src/fk_pipeline.rs` |
## License

GPL-3.0-only.
