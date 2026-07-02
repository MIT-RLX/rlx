# rlx-webgl

WebGL2 GPGPU backend for RLX — runs IR graphs (forward **and** backward) in the
browser via render-to-texture, for environments without WebGPU.

## Why render-to-texture

WebGL2 has **no compute shaders** (confirmed in wgpu's own source:
*"WebGL2, and GLES3.0 devices do not support compute"*). So this backend uses
the classic GPGPU approach: every tensor is a single-channel float (`RGBA32F`,
value in `R`) texture, and every op is a **fragment shader** that renders one
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

## Op coverage

`build_plan` first **legalizes** the graph (so a far larger op set works than
the kernels below): dedicated `*Backward` ops are decomposed via
`rlx_autodiff::decompose_backward_ops_except`, and other composites are
rewritten to primitives via `rlx_compile::rewrite_for_backend(graph,
supported_ops())`. So coverage is two layers:

**Native kernels** (`supported_ops()` — what the executor runs directly):

- **Leaves**: `Input`, `Param`, `Constant`, `Cast` (f32), `StopGradient`
- **Linear**: `MatMul`
- **Element-wise** (broadcasting via an inserted gather): `Binary{Add,Sub,Mul,Div,Max,Min,Pow}`, `Compare{Eq,Ne,Lt,Le,Gt,Ge}`, `Where`
- **Activations** (forward + `ActivationBackward`/`ReluBackward`): `Relu, Neg, Exp, Log, Sqrt, Rsqrt, Sigmoid, Tanh, Abs, Sin, Cos, Silu`
- **Reductions**: `Reduce{Sum,Mean,Max,Min,Prod}`, `Softmax` (last axis), `Cumsum` (last axis, via triangular matmul), `ArgMax`/`ArgMin`
- **Normalization**: `LayerNorm`, `RmsNorm` (last axis)
- **Positional**: `Rope` (NeoX half-split, full + partial)
- **Structural**: `Reshape`, `Transpose`, `Expand`, `Narrow`, `Reverse`, `Concat`, `Gather` (runtime indices, embedding-style)
- **Conv/pool (NCHW, forward)**: `Conv` (groups = 1, via im2col-gather + matmul), `Im2Col`, `Pool` (Max/Avg, via reduce-with-window-groups)

**Lowered to the above by the legalization passes** (no native kernel needed):
`GroupNorm`, `BatchNormInference`, `SoftmaxCrossEntropy`, `DotGeneral`,
non-last-axis / multi-axis `Reduce`, control flow (`If`/`While`), fused & RNN
ops (`GatedDeltaNet`/`Lstm`/`Gru`/`Rnn`/`Mamba2`/`SelectiveScan`/`Fused*`/…),
and the dedicated `*Backward` ops for the supported forward ops
(`LayerNormBackward*`, `RmsNormBackward*`, `GroupNormBackward*`, `RopeBackward`,
`MaxPool2dBackward`, `CumsumBackward`, `GatherBackward`,
`SoftmaxCrossEntropyBackward`, …) → so those training graphs work without
per-op backward kernels.

**Not yet supported** (clear planning error): `Conv` with groups > 1,
`ConvTranspose2d`, and **conv `weight`/`input` backward** (its decomposition
mixes flattened-concat patch layouts the flat executor doesn't yet handle — conv
*forward* + `Im2Col` do work); `ScatterAdd`; `TopK`/`Sample`; RNG; quantized I/O
(`QMatMul`/`Dequant*`); complex (`C64`); `Custom`/`Fft`/splat.

Status: the planner + numerics for the native kernels **and** the legalized
composites are **verified natively** against RLX's CPU backend (27 tests in
`tests/{cpu_parity,op_parity,composite_parity,dataflow_parity,ops2_parity}.rs` —
incl. LayerNorm/RmsNorm, GroupNorm/BatchNorm via lowering, LayerNorm backward,
Cumsum/Concat/Pool/Conv2d, Im2Col, ArgMax/ArgMin, runtime Gather, RoPE). WebGL
execution is compile-verified and browser-validated via `rlx-web`
(`just serve-web --webgl`).
