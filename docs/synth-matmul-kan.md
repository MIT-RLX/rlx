# SynthMatMul & KAN — weights and activations as *functions*, not data

Two ops that represent model parameters by a **compact function reconstructed in
fast memory** instead of a dense tensor read from DRAM, plus the performance and
training machinery around them. The unifying idea: an ML kernel that is
*bandwidth-bound* (reads more bytes than it does math) can be made much faster by
storing a small functional form and **reconstructing the weight on-chip**,
trading DRAM traffic for ALU.

- `Op::SynthMatMul` — a matmul whose weight is a **learned codebook** (vector
  quantization); the weight is reconstructed inside the matmul, never
  materialized in DRAM.
- `Op::SplineActivation` — the **KAN** edge: each channel's "weight" is a
  learnable univariate function `φ(x) = Σ_g cᵍ·RBF_g(x)`; you pass coefficients,
  not a dense weight.

## TL;DR — what shipped and what moved the needle

- **Both ops, full stack:** core IR + CPU-native kernels + decompose oracle
  (all-backend correctness) + autodiff VJP + parity/finite-difference tests.
  Native Metal kernels for both.
- **Decode (inference, M=1) is where the bandwidth win is real:** the Metal
  `SynthMatMul` decode kernel is **3–4× faster than an f32 matmul** — it moves
  ~16× fewer weight bytes (1-byte codes) *and* saturates the GPU via **split-K
  GEMV**. This is the regime LLM inference lives in.
- **Prefill/training is compute-bound → reconstruct→GEMM (don't reconstruct-in-loop):**
  a reconstruct-in-loop kernel can't beat a real GEMM. Reconstruct the weight to
  f32 once, then GEMM: **6.5× faster than the fused kernel**, ≈ f32 parity. The
  GEMM itself is now a **64×64-tile `Simd64`** kernel (default-on) that *beats
  MPS ~1.3×* on the transformer's tall/short-K shapes — MPS's private async-copy
  pipeline can't amortize at short K (RE-derived) — with a **split-K** variant
  (`RLX_METAL_SGEMM_SPLITK`) ~1.5× MPS on the fat-K `dW=xᵀ·dq` backward shape.
- **KAN spline BACKWARD — fuse it (the biggest training lever):** the spline VJP
  decomposed into exp/mul/reduce primitives that materialized the `[.., C, G]`
  RBF basis (~25M elements) to DRAM. Native fused `Op::SplineActivationBackwardX`
  / `SplineActivationBackwardCoeff` (basis built in registers, loop over
  `num_basis`) cut the rlx-tiny training **backward ~56% (185→82ms)** — bit-exact,
  finite-difference-verified. Escape hatch `RLX_DECOMPOSE_SPLINE_BWD=1`. Lesson:
  fuse ops whose *backward* materializes a huge intermediate; small-op dispatch
  count is a red herring (per-dispatch ≈ 2.75µs).
- **Low-precision codebooks (fp8/fp4/nvf4/custom `fpXmYeZ`)** via composition
  with the existing `ScaledFormat`/`lowp_codec` system — zero new backend code.
- **Zero-copy resident training:** a full GPU-resident **Muon** loop where
  gradients never leave the unified-memory arena and the optimizer steps in
  place — no GPU→host→GPU roundtrip.

All numbers below are Apple M4 Pro, `examples/synth_roofline.rs` /
`examples/synth_prefill_bench.rs`.

---

## `Op::SynthMatMul` — codebook weight-synthesis matmul

```rust
// x [m,k] f32  ·  indices [n, k/entry_dim] u8  ·  codebook [num_entries, entry_dim] f32
//   → y [m,n] f32,   y = x · Wᵀ
//   W[n, kb·entry_dim + t] = codebook[indices[n, kb]][t]
let y = g.synth_matmul(x, indices, codebook,
    SynthKind::Codebook { entry_dim, num_entries }, out_shape);
```

The weight is stored **transposed** (`[n, k]`, GGUF "bt" layout) as codebook
indices: each contiguous `entry_dim`-length sub-vector along the contraction axis
`k` is replaced by a learned centroid. `num_entries ≤ 256` (u8 codes). The
op is the sibling of `Op::DequantMatMul` — same "coded weight reconstructed in
the matmul inner loop" shape, but with a *learned* codebook instead of a fixed
quant grid.

**Gradients (`vjp_synth_mat_mul`):**
- `dx = upstream · W` (reconstruct W via the same gather, then matmul)
- `d_codebook = ScatterAdd(upstreamᵀ·x blocks, indices)` — exactly the
  data-gradient of the reconstructing gather; accumulates over reused indices.
- `indices` is integer → no gradient.

## `Op::SplineActivation` — KAN Gaussian-RBF spline

```rust
// x [.., C]  ·  coeff [C, num_basis]  →  y [.., C]   (shape-preserving)
//   y[.., c] = Σ_g coeff[c,g] · exp(-((x[..,c] − center_g)·inv_h)²)
let y = g.spline_activation(x, coeff, num_basis, grid_min, grid_max);
```

Each channel `c` has its own univariate function in a fixed Gaussian-RBF basis
(`num_basis` centers uniform on `[grid_min, grid_max]`), with learned per-channel
coefficients. Full VJP wrt **both** `x` and `coeff` (finite-difference checked).
The forward builds the `[.., C, G]` basis once and contracts it two ways for the
two gradients.

## Backend support matrix

| | Core IR + shape | CPU native | VJP | Decompose oracle | Native Metal |
|---|:-:|:-:|:-:|:-:|:-:|
| `SynthMatMul` | ✅ | ✅ | ✅ | `LowerSynthMatMul` | ✅ split-K + recon→MPS |
| `SplineActivation` | ✅ | ✅ | ✅ | `LowerSplineActivation` | ✅ per-element RBF |

Other GPU backends (CUDA/ROCm/wgpu/Vulkan) run both ops via the decompose
oracles today (one-validated-increment-per-hw cadence).

> **Metal u8-Cast fix (2026-08-02):** a `Cast(u8-Param → i64)` on Metal used to
> read the packed-`u8` indices as `f32` (4 B/elem) via the `CastTruncF32` fast
> path → garbage indices (err ≈ 2.69), because the widen pass keeps `u8` *params*
> 1-byte-packed. Fixed in `rlx-metal/thunk/compile.rs`: packed sub-4-byte integer
> **Param** sources now route through the true-width host cast (`CastHost{U8→F32}`,
> the same path `Cast(u8→f32)` already used). So the `Cast(u8→i64/f32)→Gather/
> ScatterAdd` chain is now correct on Metal — both the `SynthMatMul` **decompose**
> and its **VJP (backward)** work, so the codebook is fully trainable on Metal (not
> just CPU). The forward still prefers the native split-K/recon→MPS kernel for
> speed. All-f32 ops (KAN) always decomposed fine. **wgpu/Vulkan still have the
> analogous packed-u8 bug** (err ≈ 2.69) — same fix applies to their cast path,
> not yet done.

---

## Performance — the honest roofline

The whole point is bandwidth, so the win only exists where bandwidth is the
limit. Measured, per regime:

### Decode / inference (M=1) — bandwidth-bound → **SynthMatMul wins**

| shape | f32-sgemm | synth (split-K) | speedup |
|---|--:|--:|--:|
| 1×4096×4096 | 0.49 ms | **0.13 ms** | **3.7×** |
| 1×4096×11008 | 0.88 ms | **0.31 ms** | **3.1×** |

An f32 GEMV must stream all `k·n` weights (67–180 MB/token) → bandwidth-bound.
SynthMatMul reads ~16× fewer weight bytes (u8 codes) and closes the remaining gap
with **split-K GEMV**: a full 32-lane SIMD group cooperates per output element,
each lane summing a strided slice of the K-blocks, then `simd_sum`. That gives
32× more threads than one-per-output — the fix for M=1 GEMV being *thread-starved*
(only `n` threads exist otherwise).

### Prefill / training (M large) — compute-bound → **reconstruct → MPS**

| prefill 256×2048×2048 | time |
|---|--:|
| recon→MPS (default) | **0.735 ms** |
| fused synth kernel | 4.786 ms |
| **speedup** | **6.5×** |

At large M the op is compute-bound: both kernels do the same `2·m·k·n` FLOPs, so
the winner is raw GEMM efficiency — and **a fused GPU kernel can't beat MPS**
(reconstruction is cache-cheap since the codebook is L1-resident; MPS is a
tiled SIMD GEMM at ~3900 GFLOP/s). So the m>8 path **reconstructs the weight into
an arena-tail scratch (`synth_reconstruct` kernel) and calls `encode_mps_sgemm_bt`**,
cloning the `DequantMatMulGguf` reconstruct→MPS mechanism. On CPU the m>1 path
does the same thing (reconstruct `[n,k]` contiguous + `sgemm_bt` → Accelerate/AMX).

### f16 reconstruct → MPS-f16 (opt-in, consumer-selectable)

`RLX_METAL_SYNTH_RECON_F16=1` switches the m>8 prefill path to reconstruct the
weight in **f16** and run **MPS `hgemm`**: cast `x`→f16, `synth_reconstruct_h`
writes `W[k,n]` as half (half the scratch), `encode_mps_hgemm`, then cast the
result back to f32. The consumer picks either path by setting the flag — the f32
`recon→MPS` remains the default (bit-accurate).

| axis | f32 recon → MPS (default) | **f16 recon → MPS-f16** (`RLX_METAL_SYNTH_RECON_F16`) |
|---|---|---|
| **precision** | f32 weight + f32 GEMM | f16 weight + f16 activation, **f32 accumulate** |
| **speed** (256×2048×2048) | 0.71–0.79 ms | **0.572–0.577 ms (~1.3×)** |
| **error** vs f32 CPU | 1.6e-5 (rounding) | **1.1e-2 max-rel** (cosine ≈ 0.9999) |
| **loss** (task) | reference | negligible for inference — f16 weights are standard; the K-long f32 accumulate keeps it stable |
| **memory** (weight scratch) | `k·n·4` = **16 MB** | `k·n·2` = **8 MB** (2× smaller; +small f16 `x`/`dst` scratch) |
| **latency stability** | swings with memory contention | **steadier** — moves half the DRAM (16 MB roundtrip vs 32 MB), so the win *grows* under load |

Measured, 3-run: the mandatory `x`→f16 / `dst`→f32 casts do **not** eat the win
(MPS-f16 is enough faster). The custom mixed-precision `metal_sgemm_f16w` (f32
act × f16 weight, no casts) was tried and is **3× slower** at prefill — it's
decode-tuned, not competitive with MPS's tiled GEMM. Parity:
`synth_matmul_parity::metal_recon_f16_matches_cpu`. Use it when a ~1% relative
weight error is acceptable and you want the speed / half the scratch / steadier
latency; keep the default when you need bit-accuracy.

### IO-lever ablation (what reducing bytes buys — measured)

`examples/synth_io_ablation.rs` isolates four IO levers. The headline: **reducing
IO *bytes* only speeds things up where the kernel is actually bandwidth/scratch
bound.**

| lever | regime | effect |
|---|---|---|
| sub-byte (4-bit) indices | decode | 2× smaller indices, **~1.0× speed** (decode is latency-bound, not bandwidth-bound: 50 GB/s ≪ 150) → a *footprint* lever |
| f16 codebook | decode | negligible (codebook is a few KB, L1-resident) |
| **f16 reconstruct** | prefill | **~1.3× + 2× smaller scratch** — the one real speed win (above) |
| double-buffer the fused tiled kernel | prefill | **~1.0×** — manual prefetch can't hide the load stall without the `simdgroup_async_copy` HW intrinsic MPS uses |

### Latency (decode is launch-bound, not compute/bandwidth-bound)

Decode's real cost is **kernel-launch overhead**, not the math. Measured on the
split-K gemv (`examples/synth_io_ablation.rs`, LATENCY section):

- **Launch tax:** per-dispatch (own command buffer + commit/wait) vs batched (one
  commit/wait) — the launch overhead is **~50% of the naive per-dispatch time**
  (~0.36 ms/dispatch). A real decode step is *dozens* of kernels/token, so the #1
  lever is amortizing that tax: **Indirect Command Buffers / graph capture** (the
  Metal analogue of the CUDA graph-capture already in the tree), plus fusion and
  GPU-resident decode.
- **Vectorized loads:** `float4` x + codebook loads (natural for `entry_dim==4`,
  one `dot` per block) → **1.2–1.33×**, bit-exact (relerr 4e-6) — a free in-kernel
  latency-hiding win, complementary to the launch-tax fix.

### What did NOT work (all measured)

- **Register-tiling the fused prefill kernel** (`q4k_mm_f32`-style, reuse
  centroids across 8 rows): *slower* (346 vs 488 GFLOP/s) — the 4 KB codebook is
  L1-resident, so there's nothing to amortize; tiling only cost x-coalescing and
  occupancy.
- **f16 for decode:** neutral (1.00×). SynthMatMul already moved the weight bytes
  to u8 codes, so the dominant decode traffic is the indices; halving the tiny
  x/dst is noise. f16's value here is **pipeline integration** (f16 residual
  stream), not speed.
- **AMX host-delegate for the forward GEMM:** MPS beat AMX (2.2 vs 3.57 ms for a
  256×4096×4096). AMX's place is the small **Muon** matmuls, not the big GEMM.

### The CPU stride-n bug (fixed)

The CPU `synth_matmul_codebook` m>1 path used to reconstruct into `[k,n]` with
stride-n scatter writes — cache-hostile, ~30× slower at real sizes (masked
because parity tests use tiny shapes). Fixed: reconstruct `[n,k]` contiguous
(`copy_from_slice` per block) + `sgemm_bt`.

---

## Low-precision codebooks (fp8 / fp4 / nvf4 / custom `fpXmYeZ`)

The codebook can be stored in any low-precision float format and decoded to f32
before the matmul — **reusing the entire `ScaledFormat`/`lowp_codec` system with
no new kernel:**

```rust
// codebook stored as `fmt` codes (+ scale); decoded via Op::ScaledDequantize,
// then fed to synth_matmul.
let y = g.synth_matmul_qcodebook(x, indices, codebook_codes, codebook_scale,
    kind, fmt, layout, out_shape);
```

`fmt` is any `ScaledFormat` — the named `F8E4M3 / F8E5M2 / F4E2M1 / F6*`, the
FNUZ variants, an NVFP4 layout, or a **parameterized `ScaledFormat::Custom {
exp_bits, mant_bits, bias }`** (the `fNeXmY` family, e.g. `custom(3,0)` =
`f4e3m0`). The codebook is tiny (≤ 256×entry_dim), so the decode is negligible.

**Honest scope:** on Apple this is *format completeness + footprint*, not speed —
FP4/NVFP4/custom are software-decoded on every backend, and Metal decodes on the
CPU host. FP8 tensor cores only accelerate **per-tensor E4M3/E5M2 on CUDA
sm_89+/ROCm CDNA3**, and never for a codebook. See `docs/scaled-matmul-fp8.md`.

---

## Training — VJP, Muon, and the zero-copy resident loop

### Muon for our params

`Muon` orthogonalizes **2-D** parameters via Newton-Schulz; anything else
silently falls back to SGD-with-momentum. The codebook `[num_entries,
entry_dim]` and KAN coeff `[C, num_basis]` **are** 2-D → they get true Muon,
**provided their 2-D shape is registered** with the optimizer (a training loop
that defaults unregistered names to a flat 1-D shape silently skips
orthogonalization). Muon's Newton-Schulz matmuls are already AMX-accelerated on
macOS (`rlx-optim` → Accelerate `cblas_sgemm`).

### Zero-copy resident training (kills the gradient roundtrip)

A typical step is `forward+backward (GPU) → read grads to host Vec → optimizer on
host → write params back`. On **unified memory** the two copies are pure waste —
the arena is host-visible zero-copy. `MetalExecutable::optimizer_step_resident`
removes them:

```rust
// weight is a resident arena Input (bind_gpu_handle); grad is at output slot 1+i.
exe.optimizer_step_resident(&trainable, |name, shape, param, grad| {
    muon.step(name, shape, param, grad)   // Optimizer::step on arena-aliased slices
});
```

It forms the param `&mut [f32]` (from `arena.byte_offset(input_id)`) and grad
`&[f32]` (from `output_slots()[1+i]`) as **disjoint aliases into
`buffer.contents()`** and steps the optimizer **in place** — no host `Vec`, no
D2H/H2D. Closure-based, so rlx-metal stays decoupled from rlx-optim.

**First-class runtime API.** `optimizer_step_resident` is on `CompiledGraph`
(via the `ExecutableGraph` trait: default `false` on non-resident backends,
overridden on Metal), so training loops use `Session`/`CompiledGraph` — they
never drop to `MetalExecutable`:

```rust
let mut c = Session::new(Device::Metal).compile(backward);
c.bind_gpu_handle("W", &w_init);
let mut muon = Muon::new(lr);
for _ in 0..steps {
    c.run_read_outputs(&[("d_output", &[1.0])], Some(&[0]));   // read only the loss
    c.optimizer_step_resident(&trainable, &mut |n, s, p, g| muon.step(n, s, p, g));
}
```

**Validated end-to-end on real ops, both resident on Metal:** a KAN spline layer
(loss **15.86 → 0.034**, `resident_kan_train.rs`) and a SynthMatMul codebook layer
(loss **62.26 → 0.87**, `synth_codebook_train::synth_codebook_layer_trains_resident_on_metal`)
both train GPU-resident on Metal with Muon through the same first-class API. After
the u8-`Cast` fix above, the codebook's u8-indexed backward (`Cast(u8→i64)→Gather`
for `dx`, `Cast(u8→f32)→ScatterAdd` for `d_codebook`) is correct on Metal — a Metal
vs CPU backward parity test agrees to ~1e-6 relative (`metal_backward_matches_cpu`).
The codebook also still trains on CPU (loss **62.26 → 0.827**, exact backward).

> **Backends for reconstruct→GEMM (measured):** wgpu fails the SynthMatMul
> decompose with the *same* u8 error Metal *had* (err ≈ 2.69) — its f32-uniform
> arena keeps u8 params packed and its cast fast-path needs the same fix Metal got.
> So **wgpu + Vulkan** still need the packed-u8 cast fix (or a native kernel);
> **CUDA + ROCm** (native-dtype arenas) get reconstruct→cuBLAS/rocBLAS *free* via
> the decompose.

The full low-level pattern (see `rlx-metal/tests/resident_muon_train.rs`):

```rust
let mut backward = rlx_opt::grad_with_loss(&forward, &[w]);
// Trainable must be a graph INPUT to be resident-bindable: rebind Param→Input.
for node in backward.nodes_mut() {
    if let Op::Param { name } = &node.op && name == "W" {
        node.op = Op::Input { name: name.clone() };
    }
}
let mut exe = MetalExecutable::compile(backward);
exe.bind_gpu_handle("W", &w_init);           // W resident in the arena
let mut muon = Muon::new(lr);
for _ in 0..steps {
    // Layer 2: read back ONLY the loss — grad_W stays resident (no D2H of grads).
    let loss = exe.run_read_outputs(&[("d_output", &[1.0])], Some(&[0]))[0][0];
    // Layer 3: Muon steps on the arena in place.
    exe.optimizer_step_resident(&trainable, |n, s, p, g| muon.step(n, s, p, g));
    // W stays resident → next forward reads it with no re-upload.
}
```

**Measured:** a 2-D-weight linear regression, 300 steps, **loss 26.55 → 0.00057
(46,000×)**, zero GPU→host→GPU roundtrip anywhere in the loop.

The four "gradient in one kernel, I/O-optimized" layers: (1) zero-copy grad/param
handoff, (2) grads stay resident (`run_read_outputs(Some(&[0]))`), (3) on-arena
optimizer step, (4) all state in unified DRAM — never spills to SSD/NVMe.

---

## Flag / API reference

| flag / API | effect |
|---|---|
| `RLX_METAL_SYNTH_RECON_F16=1` | m>8 prefill reconstructs the weight in f16 → MPS `hgemm` (~1.3×, 2× smaller scratch, ~1% rel error) |
| `RLX_METAL_SYNTH_MPS_DISABLE=1` | m>8 prefill uses the fused kernel instead of reconstruct→MPS (A/B) |
| `RLX_METAL_SYNTH_TILED=1` | m>8 uses the threadgroup-tiled fused kernel (zero-scratch / capture-friendly; measured slower than recon→MPS) |
| `RLX_METAL_SYNTH_TILED_F16=1` | with `…_TILED`, use the `simdgroup_half8x8` f16 tiled variant |
| `Graph::synth_matmul(x, idx, cb, kind, shape)` | codebook weight-synth matmul |
| `Graph::synth_matmul_qcodebook(..., fmt, layout, ...)` | fp8/fp4/nvf4/custom codebook |
| `Graph::spline_activation(x, coeff, num_basis, grid_min, grid_max)` | KAN spline |
| `MetalExecutable::optimizer_step_resident(trainable, step_fn)` | zero-copy in-place optimizer step |
| `bind_gpu_handle` / `run_read_outputs(_, Some(&[0]))` | make a weight resident / read only the loss |

Benchmarks: `crates/backends/rlx-metal/examples/synth_roofline.rs` (roofline,
f32/f16, AMX-vs-MPS) and `crates/core/rlx-runtime/examples/synth_prefill_bench.rs`
(end-to-end recon→MPS vs fused).

## Bottom line

- **Inference:** "functions not data" delivers a real **3–4× decode speedup** —
  it lives exactly where IO is the bottleneck (bandwidth-bound GEMV).
- **Training:** at compute-bound prefill, reconstruct→a real GEMM (MPS on Metal,
  AMX on CPU) — don't fuse; a fused kernel can't beat a world-class GEMM at equal
  FLOPs. And keep the whole loop **resident** on the unified-memory arena so
  gradients never round-trip through the host.
- **Low precision** is format-completeness on Apple; its speed lever is CUDA
  FP8 tensor cores, a separate backend.

## License

MIT OR Apache-2.0 — © 2026 Eugene Hauptmann, Nataliya Kosmyna.
