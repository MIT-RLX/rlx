# Extending rlx from a downstream crate

rlx keeps its hot enums closed on purpose — a closed `Op` is what lets fusion,
the optimizer, and autodiff reason exhaustively (per-backend peak perf is the
north star). Everything you *compose* from those primitives, however, is open:
a downstream crate (a model in `rlx-models`, a numerics crate, a research op)
can add blocks, ops, and backends **without editing core and without a
core-crate republish**.

One import gives you the whole surface:

```rust
use rlx_extend::prelude::*;
```

Or scaffold a ready-to-fill template:

```sh
cargo rlx new-op GatedGate     # custom op: OpExtension + lower / kernel + register()
cargo rlx new-model GatedMlp   # LayerStage block composed from FlowCtx primitives
# add --stdout to print instead of writing a file
```

There are four seams. Pick by what you're adding.

---

## 1. A new architecture *block* — implement `LayerStage`

The block seam. Compose primitives through `FlowCtx`'s builders and drop the
block into any flow with `ModelFlow::layer_stage`. No `FlowStage` enum variant,
no core edit; the block lowers to ordinary primitives so it stays visible to
fusion and hits the fast path on every backend.

```rust
use rlx_extend::prelude::*;

struct GatedMlp { gate: String, up: String, down: String }

impl LayerStage for GatedMlp {
    fn name(&self) -> &str { "gated_mlp" }

    fn emit_layer(&self, ctx: &mut FlowCtx<'_>, x: FlowValue)
        -> anyhow::Result<(FlowValue, StageArtifacts)>
    {
        let g = ctx.linear(&x, &self.gate, false)?;   // curated primitive builders —
        let g = ctx.silu(&g);                          // no rlx_ir::hir / HirMut needed
        let u = ctx.linear(&x, &self.up, false)?;
        let h = ctx.mul(&g, &u);
        let out = ctx.linear(&h, &self.down, false)?;
        Ok((out.clone(), StageArtifacts::hidden_only(out.shape().clone())))
    }
}

let flow = ModelFlow::new("my_model")
    .input("x", shape)
    .layer_stage(GatedMlp { /* … */ });
```

`FlowCtx` builders: `param`, `matmul`, `linear`, `add`, `sub`, `mul`,
`residual`, `activation` (+ `relu` / `gelu` / `silu`), `rms_norm`, `cast`. For
anything not covered, drop to `ctx.hir()` (raw HIR) — the same surface the
built-in blocks use.

**Auxiliary outputs** (KV cache taps, aux heads): call
`ctx.publish_side_output("name", &value)`. The flow appends it after the primary
output automatically; its name shows up in `BuiltModel::output_names()`.

Working, tested example: `crates/core/rlx-flow/tests/downstream_layer_stage.rs`.

---

## 2. A new *operation* — implement `OpExtension`

For a genuinely new op (not a composition of existing blocks). Register it once,
then build `Op::Custom` nodes by name. The registry routes shape inference and
autodiff; you choose how it *executes*.

```rust
use rlx_extend::prelude::*;
use std::sync::Arc;

struct MyOp;
impl OpExtension for MyOp {
    fn name(&self) -> &str { "my_op" }
    fn num_inputs(&self) -> usize { 1 }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape { inputs[0].clone() }

    // OPTIONAL — decompose to primitives. If you provide this, the op fuses and
    // runs on EVERY backend with no kernel (the middle tier). Omit it if you
    // ship a hand-tuned native kernel instead.
    fn lower(&self, _node: &Node, ctx: &mut LowerContext) -> Option<NodeId> {
        let x = ctx.inputs[0];
        let s = ctx.out.node(x).shape.clone();
        Some(ctx.out.add_node(Op::Binary(rlx_ir::op::BinaryOp::Add), vec![x, x], s))
    }
}

register_op(Arc::new(MyOp));                 // or register_op_strict → errors on name clash
let y = graph.custom_op("my_op", vec![], vec![x]);   // panics if unregistered
// or, non-panicking:
let y = graph.try_custom_op("my_op", vec![], vec![x])?;  // -> Result<_, CustomOpError>
```

Three execution choices, cheapest first:

| Route | You provide | Runs on |
|---|---|---|
| **Lower** | `OpExtension::lower` (decompose to primitives) | every backend, fuses |
| **Host kernel** | `register_cpu_kernel` / `register_metal_kernel` / `register_mlx_kernel` | that backend, host-delegate; a CPU kernel also runs on CUDA/ROCm/wgpu/Vulkan via host staging |
| **Raw GPU kernel** | `MetalGpuKernel` (MSL) / `WgpuGpuKernel` (WGSL) / `CudaGpuKernel` (CUDA-C·NVRTC) / `RocmGpuKernel` (HIP-C·hipRTC) | Metal / wgpu (Metal·Vulkan·DX12·WebGPU) / CUDA / ROCm, dispatched onto the arena buffer with **no host roundtrip** — takes precedence over a same-named host kernel |
| **Custom AD** | `Graph::custom_fn` (subgraph body + VJP/JVP bodies) | anywhere the body runs |

Four raw-GPU flavors, all dispatching straight against the arena buffer:

- **Metal** — `rlx_metal::op_registry::MetalGpuKernel::encode` hands you the live compute encoder, the shared arena `MTLBuffer`, and per-operand `(byte offset, len, shape)`; compile a pipeline via `rlx_metal::pipeline_cache::load_or_compile_library` and dispatch. Example: `crates/core/rlx-runtime/tests/metal_gpu_custom_op.rs`.
- **wgpu** — `rlx_wgpu::wgpu_gpu_custom::WgpuGpuKernel` returns WGSL following a fixed binding convention (arena storage `@0`, a `params: array<u32>` of element offsets `@1`); the executor windows the operands, builds the bind group, and dispatches. Portable to browser WebGPU (unlike the host-delegate path). Example: `crates/core/rlx-runtime/tests/wgpu_gpu_custom_op.rs`.
- **CUDA** — `rlx_cuda::cuda_gpu_kernels::CudaGpuKernel` returns CUDA-C with a fixed launch signature (`float* arena` + element offsets); the executor NVRTC-compiles + caches it and launches against the arena (≤4 inputs). Example: `crates/core/rlx-runtime/tests/cuda_gpu_custom_op.rs` (validated on an NVIDIA GPU).
- **ROCm** — `rlx_rocm::rocm_gpu_kernels::RocmGpuKernel` is the CUDA seam ported to HIP (same fixed signature; HIP compiles the CUDA-C source shape verbatim via hipRTC). Example: `crates/core/rlx-runtime/tests/rocm_gpu_custom_op.rs` (compile-verified + HW-ready; runtime validation pending an AMD/ROCm GPU).

Use raw kernels for fine-grained ops where a host roundtrip would dominate (coarse ops like Sparse-LU/FFT are fine on the host path; ops expressible as primitives should prefer `lower`).

Autodiff hooks on `OpExtension`: `vjp`, `jvp`, `vmap` (all default to
non-differentiable / panic-on-use, override as needed).

Check registration before building to avoid the panic: `is_op_registered(name)`.

Tested examples: `crates/core/rlx-compile/src/rewrite.rs` (lower unit test) and
`crates/core/rlx-runtime/tests/custom_op_lowering.rs` (end-to-end on CPU with no
kernel). Real consumers: `rlx-linalg`, `rlx-sparse`, `rlx-fdm`.

---

## 2b. A custom *fusion / rewrite pass*

Register a `Pass` (graph → graph) that runs **after** the built-in fusion
pipeline (so core invariants hold) but **before** backend legalization (so its
output is still lowered). Empty by default — zero cost until you register one.

```rust
use rlx_opt::{register_ir_pass, Pass};   // or rlx_fusion::pass::{register_ir_pass, Pass}
use std::sync::Arc;

struct FuseMyPattern;
impl Pass for FuseMyPattern {
    fn name(&self) -> &str { "fuse_my_pattern" }
    fn run(&self, graph: rlx_ir::Graph) -> rlx_ir::Graph {
        // IMPORTANT: fast-path return `graph` unchanged when your pattern is
        // absent — this runs on *every* compiled graph in the process.
        graph
    }
}

register_ir_pass(Arc::new(FuseMyPattern));
```

If your pass emits an `Op::Custom`, give it an `OpExtension::lower` rule (§2) so
it still runs everywhere.

## 3. A new *backend* — register against a `Device`

Runtime backends plug in through a registry — no `match` in core.

```rust
use rlx_runtime::{register_backend, Backend, Device};

register_backend(Device::Cuda, || Box::new(MyCudaBackend));   // existing Device variant
```

Implement `Backend` (one required method, `compile`) and `ExecutableGraph`
(`set_param` + `run`). Declare what you natively support via
`Backend::supported_ops()`; the auto-rewriter decomposes everything else into
primitives you *do* claim, with a host-fallback tier for the rest — so you
implement a native kernel only where it's worth it. See
`crates/core/rlx-runtime/examples/custom_backend.rs`.

The `Device` enum itself is closed. Bind to an existing variant, or — for a
target that isn't a live device — use the codegen pattern below.

---

## 4. An ahead-of-time *codegen target* — consume `rlx_ir::Graph`

The most decoupled seam: anything that can walk an `rlx_ir::Graph` is a
"backend" in the loose sense, invoked directly (not via `Session`). This is how
`rlx-fpga` (→ SystemVerilog) and `rlx-cerebras` (→ CSL) work. Needs nothing from
`rlx-driver`/`rlx-runtime` — only the public `rlx-ir` graph type + your own
IR-recognition and legalize passes.

---

## Local development loop (rlx ↔ rlx-models)

rlx-models pins the published `rlx-*` crates (`^0.2`). To build it against your
local `../rlx` working tree:

```sh
just link-local   # write the gitignored .cargo/config.toml → ../rlx
just unlink        # revert to the published crates (CI / publish state)
```

No committed manifest carries a `[patch.crates-io]`. See the "Local rlx
development" recipes in the rlx-models `Justfile`.

---

## What stays closed (and why)

- **`Op` / `OpKind`** — closed so fusion/optimizer/autodiff are exhaustive. Add
  ops via the `OpExtension` registry (§2), not new variants.
- **`FlowStage`** — closed; extend via `LayerStage` + `layer_stage` (§1).
- **`Device`** — closed; `register_backend` on an existing variant, or a codegen
  target (§3/§4).
- **`Backend` / `ExecutableGraph` traits** — frozen surface; new knobs go on
  `CompileOptions`, not new trait methods.

## Follow-ups

- Runtime-validate the **ROCm** `RocmGpuKernel` path on an AMD GPU (it's
  compile-verified + HW-ready; there's no ROCm rig wired up yet).
## License

MIT OR Apache-2.0.
