# rlx-cuda attention kernel variants

`Op::Attention` has several CUDA kernels. They are all bit-compatible (same
math, same signature/arena layout) — they differ only in *how* they compute, so
the backend can pick the fastest for a given shape without changing results.

| variant | entry | geometry | when |
|---|---|---|---|
| **scalar flash** | `attention` | 128 threads, 16-query tile, scalar FFMA | default; any head_dim ≤ 128, any mask |
| **warp** | `attention_warp` | one 32-lane group per query row | head_dim > 128 (or `RLX_CUDA_FORCE_ATTENTION_ROW`) when rows fill the device |
| **split-KV** | `attention_warp_split` | one block per query row, seq_k split across its 32-lane groups, merged in shared memory | few rows (decode): `attention_kv_split_groups` returns ≥ 4 groups; any head_dim ≤ 512, any mask |
| **row** | `attention_row` | 1 thread/query row, `head_dim` in local mem | `RLX_CUDA_LEGACY_ATTENTION_ROW=1` (A/B only) |
| **WMMA d64** | `attention_wmma` | 4 warps, 64-query tile, fp16 Tensor Cores | head_dim ≤ 64, mask None/Causal, sm ≥ 70 |
| **WMMA d128** | `attention_wmma_d128` | 2 warps, 32-query tile, fp16 Tensor Cores | 64 < head_dim ≤ 128, mask None/Causal, sm ≥ 70 |

The two WMMA kernels share one templated body (`kernels/attention_wmma.cu`);
they run QK^T and P@V on the Tensor Cores (fp16 in, f32 accumulate) with an
online-softmax O accumulator staged in shared memory. They are CUDA-only
(`nvcuda::wmma`); ROCm always uses the scalar path.

## Which is faster (RTX 3080 Ti / sm_86, head_dim=64)

Measured with `cargo run --release -p rlx-cuda --example bench_attention` and
`RLX_CUDA_ATTENTION=scalar|wmma`:

| shape (B·S·H·D) | scalar | WMMA | winner |
|---|--:|--:|---|
| 4 · 1024 · 8 · 64 | 9.65 ms | **8.51 ms** | WMMA (**+13%**) |
| 1 · 1024 · 16 · 64 | 4.90 ms | **4.55 ms** | WMMA (**+8%**) |
| 1 · 1024 · 8 · 64 | 2.51 ms | 2.55 ms | ~tie |
| 1 · 512 · 8 · 64 | **0.95 ms** | 1.03 ms | scalar |

The **d64** WMMA kernel wins when the workload (`batch · heads · seq_q`) is large
enough to amortize its per-block overhead; on small workloads the scalar kernel
is as fast or faster. At `head_dim=64` the matmul isn't the bottleneck — the win
comes from bigger warp tiles cutting the per-iteration softmax/addressing
overhead (see `tools/kernel-inspect` datapath analysis: the kernel is ~50%
integer addressing, ~1% tensor).

**d128 status:** correct (parity-validated, None + Causal) but currently ~7%
*slower* than the scalar kernel at `head_dim=128` (S2048·D128: 15.1 vs 14.1 ms)
— the 2-warp/32-query tile is under-parallelized. So `auto` skips it; it's
reachable only via `RLX_CUDA_ATTENTION=wmma`. Making it win needs a wider tile
(more warps) with the O accumulator in registers rather than shared memory —
which for head_dim=128 requires the >48 KB dynamic-shared-memory carveout.

All WMMA parity vs the scalar reference (single process, `wmma_parity` example):
`d64/none 2.7e-5 · d64/causal 1.2e-4 · d128/none 2.4e-5 · d128/causal 1.2e-4`.

## Decode: split-KV (RTX 3080 Ti / sm_86)

At decode `rows = batch · heads · seq_q` is a few dozen, so the flash kernel
(one 16-query tile per head) and `attention_warp` (one group per row) both walk
the whole cache on a handful of SMs. `attention_warp_split` gives each row a
block of up to 16 groups, each owning a contiguous seq_k slice, then merges the
partial (max, sum, accumulator) states with the usual log-sum-exp rescale. One
launch, no arena scratch, so it is safe inside a captured graph. The group count
comes from `rlx_gpu_kernels::attention_kv_split_groups` (shared with rlx-rocm):
~1024 resident groups, power of two, ≤ 16, at least 4 keys per group, at least
4 groups (a 2-way split lost to flash at 512 rows).

`cargo run --release -p rlx-cuda --example bench_decode_attention` launches the
kernels directly on device data and scores each against an f64 reference
(measured while the GPU was shared with another job, so absolute times are
inflated; ratios held across runs):

| shape (B·H·Sq·Sk·D) | flash | warp | split-KV (auto) |
|---|--:|--:|--:|
| 1·16·1·1024·128 | 834 µs | 1873 µs | **173 µs** |
| 1·16·1·4096·128 | 3391 µs | 7649 µs | **640 µs** |
| 1·10·1·1024·256 | — | 2073 µs | **197 µs** |
| 1·10·1·4096·256 | — | 8660 µs | **708 µs** |
| 8·16·1·1024·128 | 1777 µs | 2110 µs | **605 µs** |

Max error vs f64 is 1e-8–7e-8 for split-KV (flash/warp 6e-8–2e-7).
`RLX_CUDA_ATTENTION_KV_SPLITS=<n>` pins the group count (`0` = off); split-KV
runs only under `auto`, and a tuned `AttentionRow`/`AttentionScalar` entry in the
dispatch table still wins.

## Configuring the variant

`RLX_CUDA_ATTENTION` selects the policy (also `rlx_cuda::AttentionVariant` in
`CudaRuntimeConfig`):

- `auto` **(default)** — shape-aware: picks the **d64 WMMA** kernel only when it
  is the proven-faster choice — `head_dim ≤ 64`, eligible, **and**
  `batch·heads·seq_q ≥ RLX_CUDA_ATTENTION_WMMA_MIN_WORK` (default `12288`) — else
  scalar/row. Auto is therefore always ≥ scalar. It deliberately does **not**
  pick the d128 WMMA kernel (see note below).
- `scalar` — always the scalar flash kernel (never WMMA).
- `wmma` — WMMA whenever eligible, regardless of workload size.
- `row` — always the row kernel.

`RLX_CUDA_ATTENTION_WMMA_MIN_WORK=<n>` tunes the auto threshold.
`RLX_CUDA_ATTENTION_WMMA=1` is a back-compat alias for `RLX_CUDA_ATTENTION=wmma`.

WMMA eligibility (all must hold, else scalar/row): non-row `head_dim ≤ 128`,
`mask_kind ∈ {None, Causal}`, no logit softcap, compute capability ≥ 7.0.

## Re-measuring

```bash
# wall-clock A/B
RLX_CUDA_ATTENTION=wmma   cargo run --release -p rlx-cuda --example bench_attention
RLX_CUDA_ATTENTION=scalar cargo run --release -p rlx-cuda --example bench_attention

# registers / occupancy / SASS datapath (confirms HMMA emission)
RLX_DUMP_KERNELS=/tmp/kd RLX_CUDA_ATTENTION=wmma \
  cargo run --release -p rlx-cuda --example bench_attention
python3 tools/kernel-inspect/kinspect.py analyze /tmp/kd
```
## License

MIT OR Apache-2.0.
