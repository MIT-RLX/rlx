# rlx-webgl

WebGL2 GPGPU backend for RLX — runs IR graphs (forward **and** backward) in the
browser via render-to-texture, for environments without WebGPU.

## Why render-to-texture

WebGL2 has **no compute shaders** (confirmed in wgpu's own source:
*"WebGL2, and GLES3.0 devices do not support compute"*). So this backend uses
the classic GPGPU approach: every tensor is a single-channel float (`R32F`)
texture, and every op is a **fragment shader** that renders one
output element per fragment. `readPixels` is synchronous, so the whole path is
synchronous (no async plumbing, unlike WebGPU).

## How it stays correct without a browser in CI

The design isolates the untestable part (GL plumbing) from the math:

```
Graph ──► plan::build_plan ──► Plan ──┬─► exec_cpu::run_cpu   (native, TESTED)
   (all index math precomputed here)  └─► exec_gl::GlBackend  (wasm, GLSL mirrors exec_cpu)
```

- **`plan`** lowers a graph to a flat op list. All index arithmetic is
  precomputed here in plain Rust: transpose / reshape / expand become
  *gather-by-index*, reduce becomes *sum-over-index-groups*.
- **`exec_cpu`** runs the `Plan` on the CPU. `tests/cpu_parity.rs` checks it
  against RLX's own CPU autodiff for a full MLP forward+backward graph — so the
  planner and every op formula are **verified natively**.
- **`exec_gl`** runs the same `Plan` on WebGL2; each fragment shader mirrors the
  corresponding `exec_cpu` formula and fetches inputs by the planner's
  precomputed indices. It inherits the verified math; only the GL plumbing
  (texture setup, framebuffers, `readPixels`) needs in-browser validation.
  Param/constant textures and the planner's index textures are uploaded once
  and reused across runs (`GlBackend::forget_param` drops one when
  `set_param` changes it); each run frees its own input and intermediate
  textures.
- Every tensor is one `rows × cols` texture, and WebGL caps each side at
  `MAX_TEXTURE_SIZE` (16384 on most desktop browsers).
  `Plan::max_texture_side()` / `Plan::largest_texture()` say what a plan needs
  before you upload it.

## Op coverage

`build_plan` first **legalizes** the graph (so a far larger op set works than
the kernels below): dedicated `*Backward` ops are decomposed via
`rlx_autodiff::decompose_backward_ops_except`, and other composites are
rewritten to primitives via `rlx_compile::rewrite_for_backend(graph,
supported_ops())`. So coverage is two layers:

**Native kernels** (`supported_ops()` — what the executor runs directly):

- **Leaves**: `Input`, `Param`, `Constant` (any dtype, decoded by value — e.g. the I64 index tables legalization builds), `Cast` (f32), `StopGradient`
- **Linear**: `MatMul`
- **Element-wise** (broadcasting via an inserted gather): `Binary{Add,Sub,Mul,Div,Max,Min,Pow,Mod,BitAnd,BitOr,BitXor,Shl,Shr,Atan2}`, `Compare{Eq,Ne,Lt,Le,Gt,Ge}`, `Where`
- **Activations** — all 29 (`Activation::ALL`), forward + `ActivationBackward`/`ReluBackward`; `Round` is half-to-even with a straight-through gradient, `Sign(0) = 0`
- **Reductions**: `Reduce{Sum,Mean,Max,Min,Prod}`, `Softmax` (last axis), `Cumsum` (last axis, via triangular matmul), `ArgMax`/`ArgMin`
- **Sorting**: `Sort`, `ArgSort` (any axis, stable), `TopK` (largest, ties → smaller index) — one fragment per output finds the element of its rank
- **Normalization**: `LayerNorm`, `RmsNorm` (last axis), `LayerNorm2d` (over channels: transpose + LayerNorm)
- **Positional**: `Rope` (NeoX half-split, full + partial)
- **Structural**: `Reshape`, `Transpose`, `Expand`, `Narrow`, `Reverse`, `Concat`, `Gather` (runtime indices, embedding-style), `Interpolate3d` (nearest)
- **Runtime-index gather/scatter**: `GatherElements`, `GatherNd` (incl. `batch_dims`), `ScatterNd`, `ScatterElements` (reductions none/add/mul/max/min), `ScatterAdd` (any axis). Negative indices count from the end. WebGL has no atomics, so a scatter runs one fragment per *output* that folds in the updates landing on it, in index order (`none` = last write wins, as on rlx-cpu)
- **Conv/pool (NCHW)**: `Conv` (groups = 1 via im2col-gather + matmul; grouped/depthwise via gather-multiply-reduce), `Im2Col`, `Pool` (Max/Avg, via reduce-with-window-groups)
- **3-D conv (NCDHW)**: `Conv3d` (same two forms; an ungrouped conv with ≥ 4 output channels runs up to 16 channels per fragment into up to four RGBA scratch targets (multiple render targets) with its weights regrouped into RGBA texels, so each input tap is fetched once per 16 channels: 3.5–3.8× on 3×3×3 volume models, identical outputs; an `Activation` that is the conv's (or a `GroupNorm`'s) only reader is applied as it writes instead of as its own pass, `RLX_WEBGL_NO_FOLD_ACT=1` to keep it apart; `tests/gl_conv_parity.rs` checks the shaders in a browser), `ConvTranspose3d` (gather form, strided/dilated/output-padded/grouped)
- **FFT**: `Fft` along the last axis, F32/F64 2N-block or interleaved C64, every `FftNorm`, any length, O(N log N): power-of-two lengths run `log₂ N` radix-2 Stockham passes (one butterfly per output lane), other lengths Bluestein (a power-of-two FFT convolution). Twiddles, chirp and filter spectrum are computed in f64 on the host, so no shader evaluates `sin`/`cos`; error matches rlx-cpu's (≈1e-7 of peak), and intermediates stay ≤ 4096 lanes wide
- **RNG**: `RngUniform`, `RngNormal` — counter-based and keyed by the node, so the planner computes them with the default RNG policy (what an rlx-cpu session uses) and uploads a constant; a session-level RNG policy change does not reach WebGL
- **Linalg** (host-staged, f64): `Cholesky`, `TriangularSolve`, `DenseSolve`, `BatchedDenseSolve`, `Det`, `LogDet`. Factorizations are sequential, so the GL executor reads the operands back, solves on the host, and uploads the result

**Lowered to the above by the legalization passes** (no native kernel needed):
`GroupNorm`, `BatchNormInference`, `SoftmaxCrossEntropy`, `DotGeneral`,
`Pad`, `Slice` (incl. negative / strided steps), `Roll`, `Clamp`, `Tile`,
`Trilu`, `CumProd`, `CumMax`, `Histogram`, `ResizeNearest2x`,
`ConvTranspose1d`/`2d`, `LoraMatMul`, non-last-axis / multi-axis `Reduce`,
control flow (`If`/`While`), fused & RNN ops
(`GatedDeltaNet`/`Lstm`/`Gru`/`Rnn`/`Mamba2`/`SelectiveScan`/`Fused*`/…), and
the dedicated `*Backward` ops for the supported forward ops
(`LayerNormBackward*`, `RmsNormBackward*`, `GroupNormBackward*`, `RopeBackward`,
`MaxPool2dBackward`, `CumsumBackward`, `GatherBackward`, `Conv*Backward*`,
`SoftmaxCrossEntropyBackward`, …) → so those training graphs (conv nets
included) work without per-op backward kernels.

**Host fallback** (`host.rs`, `Step::Host`): any op left after legalization
that rlx-cpu runs is executed *by* rlx-cpu — a one-node graph compiled once,
its operands encoded to their real dtypes (int8, I64, f64, C64, …). That covers
`Sample`, `Quantize`/`Dequantize`, `QMatMul`/`QConv2d`, `DequantMatMul` and the
other quantized matmuls, `Svd`/`Qr`/`Eigh`, registered (non-collective)
`Custom` ops, complex helpers, `LogMel`/`WelchPeaks`, the SPD ops, and splatting
(after `rlx_splat::register()`, as on the CPU). It matches rlx-cpu by
construction, but on the GL path each such op is a GPU → host → GPU round trip,
and integer values ride f32 slots (exact below 2²⁴). rlx-cpu is built without
BLAS here, so the same pure-Rust code runs natively and in wasm (rlx-web links
rlx-cpu already). `runnable_ops()` — native ∪ host — is what the runtime's
`Device::OpenGl` claims; legalization still lowers against `supported_ops()`, so
an op with a WebGL lowering keeps it.

Status: the planner + numerics for the native kernels **and** the legalized
composites are **verified natively** against RLX's CPU backend
(`tests/{cpu_parity,op_parity,composite_parity,dataflow_parity,ops2_parity,ops3_parity,host_fallback_parity}.rs`;
activation derivatives rlx-cpu has no kernel for are checked against rlxsl's
auto-differentiated gradient). The GLSL is compiled by the browser at
`GlBackend::new`, so it is validated in-browser via `rlx-web`
(`just serve-web --webgl`), not by the native tests.

## License

MIT OR Apache-2.0.

## Profiling

Set `globalThis.__rlxWebglProfile = true` in the page or worker that runs a
plan: every step is then followed by `gl.finish()` and the run logs its time
by step kind to the console. The waits themselves cost, so rank the steps
rather than quoting the totals.
