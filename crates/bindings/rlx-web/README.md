# rlx-web

WebAssembly entry point for RLX — run models in the browser.

It compiles to `wasm32-unknown-unknown` and exposes a small JS-callable surface
via `wasm-bindgen`.

## Real transformer in the browser

`transformer_next_logits(tokens, vocab, dim, n_layers, n_heads, head_dim, ffn,
seed)` runs a **real decoder-only transformer** — RMSNorm + RoPE + causal
multi-head attention + SwiGLU MLP — on the CPU backend and returns the
next-token logits. Weights are **deterministically synthesized** from `seed`, so
this is a faithful *architecture* run (real graph, real ops), not pretrained
weights. It's verified end-to-end natively (`src/transformer.rs` tests):
determinism, output shape, and the **causal property** (a position's logits
never depend on later tokens), plus isolations for bare attention and
attention+RoPE.

Loading real GGUF pretrained weights is a separate, still-missing piece: the
old `rlx-models` GGUF→Qwen3 runner was deleted and has no in-tree replacement,
and the byte-level GGUF parser (`rlx_gguf::GgufFile::from_reader`) + an
HF↔GGUF name/transpose mapping would need rebuilding to feed real weights into
this graph.

### Two things that had to be fixed for *any* model to run in a browser
- **BLAS:** `rlx-cpu`'s default `blas` feature declared an `extern "C"
  cblas_sgemm` that becomes an unsatisfiable wasm import (instantiate
  `LinkError`). It's now routed to the pure-Rust scalar GEMM on wasm
  (`blas.rs` cfgs gated `not(target_arch="wasm32")`).
- **32-bit overflow:** `rlx-runtime`'s decode memory-budget constants
  (12 GiB / 4 GiB) overflowed 32-bit `usize`; computed in `u64` now.

### CPU fusion bug surfaced here — now fixed
Building this transformer surfaced a real backend bug: the CPU attention-block
fusion (`rlx-cpu`, `Thunk::FusedAttnBlock`) collapsed
`QKV → narrow×3 → RoPE×2 → attention → out-proj` into one kernel but **dropped
the mask kind**, applying only a per-key padding mask. A `MaskKind::Causal`
decoder therefore attended to future tokens (the 1-layer leak was ~0.07;
`skip_fusion` gave 0). The fix threads `mask_kind` through the fused block and
synthesizes the causal / sliding-window mask in-kernel, so `transformer_logits`
now compiles with the **default** fusion pipeline. Guarded by
`rlx-cpu`'s `fused_attn_block_respects_causal_mask` and this crate's
`fused_path_is_causal_like_unfused` / `causal_*` tests.

## Quickstart

```sh
# one command, all platforms — builds the bundle and serves the demo
just serve-web

# same, but also bring up a WebGPU device
just serve-web --webgpu
```

Then open <http://localhost:8000>. (ES modules + wasm must be served over
`http://`, not opened as `file://`.)

Prerequisite: the `wasm-bindgen` **CLI** must match the `wasm-bindgen` crate
version. The build script checks this and prints the exact command if not:

```sh
cargo install wasm-bindgen-cli --version <crate-version>
```

Build without serving: `just build-web` (add `--webgpu`, `--webgl`, or `--all`).
The bundle lands in `crates/rlx-web/web/pkg/`.

## Compute paths

Every path runs **both forward and backward** (the backward pass is the
[`rlx_autodiff`] gradient graph). The graphs are identical across backends —
only the executor differs.

| Path | Mechanism | Sync? | Verification |
|------|-----------|-------|--------------|
| **CPU** (default) | `rlx_runtime::Session` on `rlx-cpu` (single-threaded on wasm) | sync | ✅ native test: grads match finite-differences; SGD reduces loss |
| **WebGPU** (`--webgpu`) | compute shaders via `rlx_wgpu::WgpuExecutable::run_async` | async (GPU→CPU `map_async` can't block the event loop) | native: wgpu tests; in-browser: `just serve-web --webgpu` |
| **WebGL2** (`--webgl`) | render-to-texture via `rlx-webgl` (no compute shaders on WebGL2) | sync (`readPixels`) | ✅ native: planner+numerics vs autodiff; in-browser: `just serve-web --webgl` |

The GPU executors can only be *run* in a browser, so they are compile-verified
here and validated by serving the demo, which cross-checks GPU results against
the CPU reference. The WebGL backend's planner + numerics are additionally
verified natively (see `rlx-webgl`).

## Vision benchmark (browser)

Four classification models run in the browser via [`VisionBench`](src/api.rs):

| Slug | Architecture |
|------|----------------|
| `mnist-cnn` | TinyConv MNIST (matches rlx-cortexm trainer) |
| `mnist-mlp` | Flattened 784→128→10 MLP |
| `cifar-cnn` | 3-block CNN for 32×32 RGB |
| `resnet` | CIFAR-sized ResNet-style (two residual blocks) |

Open <http://localhost:8000/vision-bench.html> after `just serve-web --all`.

### TypeScript / JavaScript SDK

The [`web/rlx.js`](web/rlx.js) wrapper (types in [`web/rlx.d.ts`](web/rlx.d.ts),
source in [`web/rlx.ts`](web/rlx.ts)) provides a developer-friendly API over the
raw wasm-bindgen exports:

```js
import Rlx from "./rlx.js";

const rlx = await Rlx.init({ webgpu: true });
const model = rlx.vision("mnist-cnn");
const params = model.initParams(42);
const { x, label } = model.syntheticBatch(1);
const logits = model.forwardCpu(x, params);
const { loss, params: next } = model.trainStepCpu(x, label, params, 0.01);
const bench = model.benchCpu(50, 42, 0.01);
```

Low-level wasm exports remain available as `rlx.wasm` / `./pkg/rlx_web.js`.
`just build-web` emits `./pkg/rlx_web.d.ts` via wasm-bindgen `--typescript`.

## JS API (low-level)

```js
import init, * as rlx from "./pkg/rlx_web.js";
await init();                                  // loads wasm; installs panic hook

// CPU (sync)
rlx.mlp_forward(x, in, hid, out, w1, b1, w2, b2);             // -> y
rlx.mlp_grads(x, t, in, hid, out, w1, b1, w2, b2);           // -> [loss, ∂w1…, ∂b1…, ∂w2…, ∂b2…]
rlx.mlp_train_step(x, t, in, hid, out, w1, b1, w2, b2, lr);  // -> updated params

// WebGPU (async, --webgpu)
await rlx.init_webgpu();                                      // -> bool
await rlx.mlp_forward_gpu(x, in, hid, out, w1, b1, w2, b2);
await rlx.mlp_grads_gpu(x, t, in, hid, out, w1, b1, w2, b2);

// WebGL2 (sync, --webgl)
rlx.mlp_forward_webgl(x, in, hid, out, w1, b1, w2, b2);
rlx.mlp_grads_webgl(x, t, in, hid, out, w1, b1, w2, b2);
```

## CI gate

`just check-wasm` compiles the CPU + WebGPU + WebGL stack for
`wasm32-unknown-unknown` (all feature configs) and runs `rlx-webgl`'s native
parity tests. It is part of `just ci`.

## License

GPL-3.0-only.
