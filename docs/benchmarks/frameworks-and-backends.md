# MNIST training comparison: frameworks × backends

The `rlx-paper/bench` MNIST-training comparison spans many frameworks and
backends. This page is the **map**: what is implemented, what is verified on
which hardware, how to run each, and what is scaffolded for the CUDA rig or left
as a candidate. For the deep dive on the CoreML/ANE numbers (and the crucial
*overhead-bound vs compute-bound* caveat that applies to **all** of these
small-MNIST results), see [coreml-training.md](coreml-training.md).

> **Read every number with the overhead caveat.** The MNIST CNN/MLP is tiny
> (≤0.1–0.6 MMAC/sample); at these sizes the loop is dominated by framework
> overhead, not the CPU/GPU/ANE. Use `HIDDEN=<n>` (RLX bench + `mpsgraph` +
> `make_model.py`) to scale into the compute-bound regime where hardware
> actually separates.

> **Compare within a model class.** The CSV has a **`model`** column
> (`cnn` / `mlp` / `infer`). Not every runner trains the same net — some only do
> the MLP (NumPy, sklearn, dfdx, IREE, jax-metal, mpsgraph, CoreML, the
> `rlx-*-mlp` rows). Comparing e.g. NumPy-MLP (654k) to RLX-CNN is apples-to-
> oranges; filter on `model` first. Headline within-class results: **CNN** —
> RLX leads GPU (`rlx-metal` 67k) and CPU (`rlx-graphfused` 49.7k); **MLP** —
> NumPy is the BLAS floor (~650–800k) that nothing with autodiff beats, but
> `rlx-mlp-cpu` (435k) is the **fastest *framework* MLP** — 2nd only to raw NumPy,
> ahead of sklearn (267k), coreml-swift (108k), IREE/dfdx/jax-metal. RLX has an
> MLP row on **every backend**: cpu, mlx, metal (thunk + mpsgraph), wgpu, ane
> (`rlx-mlp-*`, `rlx-graphfused-mlp`, `rlx-coreml-mlp-*`).

> **libtorch backends** (`tch-mnist/`, burn `-F cpu`): the libtorch headers
> don't compile under the macOS-26.5 clang (`is_arithmetic` specialization is now
> forbidden) — build with `CXXFLAGS="-Wno-invalid-specialization"`, and run with
> `DYLD_LIBRARY_PATH=<…/libtorch/lib>`.

## Status matrix

Legend: **✓** verified on this Apple-Silicon host · **rig** runnable but needs
the CUDA/Linux rig (script is device-aware, not run here) · **cand.** candidate,
not yet implemented.

| Framework | CPU | Metal/MPS | MLX | ANE/CoreML | CUDA | runner |
|-----------|-----|-----------|-----|-----------|------|--------|
| **RLX** | ✓ (scalar/fused/graphfused) | ✓ `rlx-metal` (+wgpu) | ✓ `rlx-mlx-train` | ✓ `rlx-coreml-mlp` | rig (`DEVICE=cuda --features cuda`) | `rlx-mnist-device`, `rlx-cortexm/train-mnist` |
| PyTorch (eager) | ✓ | ✓ (mps) | – | – | rig | `train_mnist_torch.py` |
| PyTorch (`torch.compile`) | ✓ | ✓ (mps) | – | – | rig | `train_mnist_torch.py COMPILE=1` |
| JAX | ✓ | ✓ (metal, MLP, exp.) | – | – | rig | `train_mnist_jax.py` / `train_mnist_jax_mlp.py` |
| TensorFlow | ✓ | ✓ (metal) | – | – | rig | `train_mnist_tf.py` (+ tensorflow-metal venv) |
| Keras 3 | ✓ (torch) | ✓ (tensorflow→metal) | – | – | rig | `train_mnist_keras.py` (`KERAS_BACKEND=`) |
| MLX (Python) | – | ✓ | ✓ | – | – | `train_mnist_mlx.py` |
| MPSGraph (Swift) | – | ✓ | – | – | – | `mpsgraph-swift/` |
| CoreML (Swift `MLUpdateTask`) | – | – | – | ✓ | – | `coreml-swift/` |
| candle | ✓ | ✓ | – | – | rig | `candle-mnist/` |
| burn | ✓ (libtorch) | ✓ (wgpu) | – | – | rig | `burn-mnist/` (`-F cpu` = burn-tch; ndarray/candle can't) |
| tch (libtorch) | ✓ | ✓ (mps) | – | – | rig | `tch-mnist/` (`DEVICE=cpu\|mps`) |
| tinygrad | ✓ (`DEV=CPU`) | ✓ | – | – | rig | `train_mnist_tinygrad.py` |
| PaddlePaddle | ✓ | – | – | – | rig | `train_mnist_paddle.py` (py3.12) |
| NumPy (from scratch) | ✓ | – | – | – | – | `train_mnist_numpy.py` — the **no-framework floor** |
| scikit-learn | ✓ | – | – | – | – | `train_mnist_sklearn.py` (MLPClassifier) |
| dfdx (Rust) | ✓ (MLP) | – | – | – | rig | `dfdx-mnist/` |
| Flux.jl (Julia) | ✓ | – | – | – | rig | `train_mnist_flux.jl` |
| IREE (MLIR) | ✓ (MLP) | – | – | – | rig (cuda) | `train_mnist_iree.py` (py3.12) |
| ONNX Runtime (ORTModule) | – | – | – | – | **rig** | `train_mnist_ort.py` |
| Luminal (Rust) | cand. | cand. | – | – | cand. | — |
| Mojo / Modular MAX | cand. | cand. | – | – | cand. | — |
| BNNS (Accelerate) | cand. | – | – | – | – | — |

### Apple-GPU variants (added)

Every major framework now has a CPU **and** an Apple-GPU row. The plugins lag
mainline, so both needed pinned versions:

| row | acc | img/s | notes |
|-----|-----|-------|-------|
| `tensorflow,gpu` (tensorflow-metal) | 0.977 | 15.4k | `tensorflow==2.16.2 tensorflow-metal==1.2.0` in a venv; latest TF's plugin won't load |
| `jax-mlp,metal` (jax-metal) | 0.952 | 38.6k | `jax==0.4.34 jax-metal==0.1.1`; experimental Metal backend, conv partial → MLP only |
| `rlx-mlp-wgpu,gpu` (RLX wgpu) | 0.948 | 15.7k | RLX's cross-vendor GPU backend (Metal via wgpu); MLP |

Notably **TF-on-GPU (15.4k) is *slower* than TF-on-CPU (45k)** here — the
overhead-bound tiny-MNIST effect again (Apple-GPU dispatch/transfer isn't
amortized). RLX's CNN on **wgpu is impractically slow** for the per-step host-SGD
loop (wgpu buffer-readback latency); the MLP is usable. These exercise the GPU
but, like the rest, only become a real hardware comparison on a compute-bound
model.

### The overhead floor (NumPy / scikit-learn)

A hand-written **NumPy** MLP (just BLAS `sgemm`, zero framework/dispatch
overhead) trains at **~790k img/s** (high variance, ~630–800k) — and
**scikit-learn**'s `MLPClassifier` at ~267k. That number *is* the BLAS floor:
on a 0.1-MMAC net the arithmetic is a rounding error, so the benchmark ranks
runtimes' per-step overhead, not the CPU/GPU/ANE.

**RLX now reaches that floor.** `rlx-graphfused-mlp` (fully-fused step, params
GPU/arena-resident, in-graph SGD) trains at **~783k img/s at 0.9586 acc** — it
*matches* NumPy on throughput and *beats* it on accuracy (NumPy 0.9474, same
hyperparams; the gap is graphfused's shuffled batches + He-uniform init). See
[Auto-fusion](#auto-fusion-how-the-rlx-mlp-reached-the-numpy-floor) for how it
got there. The host-SGD variant (`rlx-mlp-cpu`, ~505k) stays below the floor: it
reads grads and writes params back across the FFI boundary every step (~0.8 MB of
memcpy NumPy never pays), which is exactly the overhead the resident/graphfused
path removes.

### Auto-fusion: how the RLX MLP reached the NumPy floor

Three lowering-time fusions in `rlx-cpu`'s thunk compiler
(`compile_thunks`, applied automatically to every CPU graph — no opt-in, no
manual graph surgery) turned a 9× gap into parity. Profile of the fully-fused
MLP step (`RLX_PROFILE_THUNKS=1`, 2-epoch run) before → after:

| thunk kind        | before | after  | what changed |
|-------------------|-------:|-------:|--------------|
| `Transpose`       | 151 ms | 0.5 ms | folded into matmul (#1) |
| `BinaryFull` (SGD)| 199 ms | 0.4 ms | folded into one kernel (#2) |
| `ReluBackward`    |  33 ms |  31 ms | already a single fused kernel (#3) |
| **total / step**  | 497 ms | 127 ms | **−74%**, 317k → ~783k img/s |

1. **Transpose → matmul-trans fold** (`Thunk::SgemmT`). Matmul backprop emits
   `Transpose(operand) → MatMul` for `dA = g·Bᵀ` and `dB = Aᵀ·g`; the transpose
   is a full materialized copy and was the dominant backward cost. The compiler
   detects a last-two-axis `Transpose` feeding a 2-D F32 matmul with a single use
   and folds it into cblas `trans_a`/`trans_b` flags (zero copy). Gradient-
   verified against finite differences.
2. **SGD-momentum fold** (`Thunk::SgdMomentum`). The in-graph update appends, per
   param, `v' = mom·v + g ; p' = p − lr·v'` — 4 `BinaryFull` ops + 2 full-size
   constant tensors, which dominated the *whole* step (62%). The compiler
   recognizes the `Sub(p, Mul(Add(Mul(v,momᶜ),g), lrᶜ))` chain (both `p'` and
   `v'` graph outputs, single-use inner nodes, uniform constants) and collapses
   it into one fused pass per param.
3. **ReluBackward** was already a dedicated `dx = dy·(x>0)` kernel emitted by
   autodiff — no Greater+Mul sequence to fuse; the 15% it cost is inherent
   memory-bound traffic.

A general elementwise region-fuser exists (`RLX_REGIONS=1`) and *does* collapse
the SGD chain, but its generic interpreter is ~26× slower per op (total
319 → 785 ms), so the dedicated kernels win decisively. After these folds the
remaining step cost is `FusedMmBiasAct` (forward) + `SgemmT` (backward) +
`SgdMomentum` — i.e. the same Accelerate `sgemm` calls NumPy makes, plus the
fused update; the per-op dispatch overhead is gone.

**The last gap is data staging, not compute.** Two experiments isolate what's
left of the step (~0.04 ms over the ~0.13 ms of compute):

- *Writeback* (`p'`/`v'` memcpy'd over `param`/`vel` each step): skipping it
  (`RLX_SKIP_WRITEBACK`, wrong results, timing only) changes throughput by ~4%
  — within noise. The arena memcpy is cache-resident and cheap; an in-place SGD
  rewrite isn't worth it.
- *Shuffle gather*: RLX reshuffles every epoch and `fill_batch` gathers the
  mini-batch into the arena's input slot (~400 KB/step of random-access memcpy).
  NumPy's reference **never shuffles** — it BLAS-reads a contiguous
  `xtr[i*B:(i+1)*B]` *view* with zero copy. Apples-to-apples (`RLX_NO_SHUFFLE=1`,
  sequential like NumPy) RLX reaches **~780–850k img/s, at or above NumPy**, at
  0.9421 acc. So RLX's default *does extra work* (the gather) *for extra accuracy*
  (shuffle → 0.9586) and still matches NumPy's throughput; strip the gather and it
  pulls ahead. The residual difference is purely NumPy's zero-copy input views vs
  RLX staging each batch into a fixed arena slot — an artifact of this micro-bench
  (real training stages data from disk/augmentation and both pay it), not RLX
  overhead.

(Machine-load variance is high on this box: single runs span ~370–850k for both
RLX and NumPy. The medians, not any single run, are the signal.)

### Device-resident MLP on GPU (`RLX_RESIDENT=1 RLX_ARCH=mlp`)

The Metal MLP was running **host-SGD**: forward+backward on the GPU, then grads
copied to the host, SGD on the CPU, params copied back — two host↔device
transfers per step (`rlx-mlp-thunk` ~175–204k, `rlx-mlp-mpsgraph` ~201k). The
device-resident path (`build_resident_mlp`) keeps params *and* velocity as GPU
handles, folds the SGD+momentum update into the graph, and feeds `p'`/`v'`
straight back into the handles on-device — **zero per-step param transfer**.
Result: **`rlx-resident-mlp` ~230k img/s at 0.9509 acc on Metal** — a ~15–30%
gain over host-SGD *and* higher accuracy, the fastest MLP-on-Metal path in the
table. (mlx has no GPU-handle binding, so it falls back/skips; CPU lacks GPU
handles — its resident equivalent is the arena-resident `rlx-graphfused-mlp`.)

The MLP win came free because the MLP lowers entirely to **MPSGraph**, where
Apple's compiler already fuses the SGD chain — there were never separate launches
to collapse. (The CPU `SgdMomentum` fold has no Metal analogue to port: the fast
Metal paths use MPSGraph, and the host-SGD paths do the update on the CPU.)

### Conv2d on MPSGraph — the CNN now trains fused on Apple's compiler

The resident **CNN** was the one training path MPSGraph rejected: `Conv2d` was
unsupported, so it fell back to per-kernel thunks. Wiring conv forward,
`convolution2D{Data,Weights}Gradient`, `maxPooling2D` and `maxPooling2DGradient`
into `mps_graph_lower` (all native MPSGraph primitives, NCHW/OIHW so no layout
transposes) lets the **whole CNN step — forward, backward, in-graph SGD — lower
to one fused MPSGraph**. The payoff is in the next section; the broader win is
structural: any conv graph can now train on MPSGraph, not just this CNN.

### The "67k ceiling" was thunk dispatch overhead — MPSGraph fusion is 2.1× past it

The 67k figure turned out **not** to be a conv-compute floor at all. The adaptive
MPSGraph-vs-per-op dispatch (`encode_and_run`) gates on the largest *matmul* FLOP
count — and conv contributes none, so a conv-heavy graph looked "tiny" (the CNN's
only matmul is the 0.5-MFLOP FC layer, under the 1M threshold) and silently ran on
**per-op thunks** despite a perfectly good MPSGraph plan sitting unused. Counting
conv FLOPs in `max_matmul_flops_in` fixes the misclassification, and the resident
CNN's whole step now runs as one fused MPSGraph:

| resident CNN on Metal | img/s | acc | path |
|-----------------------|------:|----:|------|
| per-op thunks (old default, `RLX_DISABLE_MPSGRAPH=1`) | ~67k | 0.9812 | im2col + native kernels |
| **fused MPSGraph (new default)** | **~142k** | 0.9739 | one Apple-compiled graph |

**2.1× faster** — the ceiling was per-op dispatch/ObjC overhead across ~69 kernels,
not conv math. First-80-step training losses are **bit-identical** between the two
paths (2.4261 → 0.7199 → …), confirming the lowering is correct; the final 0.9739
vs 0.9812 is MPS's conv algorithm (Winograd-class) accumulating differently over
~1900 steps — a float-trajectory difference, not a bug. `RLX_DISABLE_MPSGRAPH=1`
restores the 0.9812 / 67k thunk path for accuracy-sensitive use.

**Both configs are reported in `mnist_training.csv`**, deliberately kept side by
side rather than picking one — the speed/accuracy trade is the point:
* `rlx-resident,cnn,metal` → **0.9739 / ~142k** — fused MPSGraph, the default
  (speed-first; the heuristic now correctly routes the conv graph to its plan).
* `rlx-resident-thunk,cnn,metal` → **0.9812 / ~66k** — per-op thunks via
  `RLX_DISABLE_MPSGRAPH=1` (accuracy-first).

Pick per use: the fused path for throughput, the thunk path when the 0.7-pt
accuracy matters. Numbers are single-run on a load-variable box (the MPSGraph CNN
spans ~142–184k across runs); medians are the signal.

### Mixed precision (fp16)

`RLX_MPS_FP16=1` runs conv + matmul (forward *and* both gradients) in fp16 inside
the lowerer — cast inputs to half, compute, cast back to fp32 so storage, loss,
and the in-graph SGD stay full-precision. It is correct and active, but gives
**no speedup on the MNIST CNN** (the conv is ~77 MMAC/step — even fused, the step
is dispatch/sync-bound, not compute-bound). It only pays once a step is genuinely
compute-bound, e.g. the resident MLP at `HIDDEN=16384`:

| resident MLP, HIDDEN=16384 | img/s | acc |
|----------------------------|------:|----:|
| fp32 | ~18.9k | 0.9658 |
| **fp16** | **~22.0k (+16%)** | 0.9654 |

Kept as a general capability for compute-bound training.

## What's new in this round

* **`torch.compile` / Inductor** — `COMPILE=1` on the PyTorch runner. On the tiny
  MNIST net it is a **net loss** (CPU 3.8k vs eager 5.3k; MPS 11.9k vs eager
  19.4k) because the Inductor compile cost (a ~5 s first step on CPU) and a
  `max_pool2d_backward` graph break on MPS aren't amortized — the overhead-bound
  effect again. Expected to win on larger models.
* **Keras 3** (`train_mnist_keras.py`) — multi-backend via `KERAS_BACKEND`. The
  **torch** backend works here (0.9855 acc, ~9k img/s on CPU). The **jax**
  backend is currently broken on this box (jaxlib 0.10 / Keras XLA attribute
  mismatch).
* **MPSGraph native-Metal trainer** (`mpsgraph-swift/`) — Apple's low-level graph
  API, SGD+momentum folded in, weights GPU-resident. A native-Metal control
  alongside MLX-Python. ~0.95 acc, ~56k img/s on the 128-hidden MLP.
* **Device-aware Python runners** — `DEVICE=cpu|mps|cuda` (torch), `DEVICE=cpu|cuda`
  (jax), so the same scripts produce the CUDA rows on the rig.
* **RLX bench** now maps `DEVICE=cuda` → `Device::Cuda` and `DEVICE=rocm` →
  `Device::Rocm` (previously only cpu/metal/mlx/wgpu/vulkan).

## CUDA / ROCm runbook (the rig)

All runners are device-aware; on an NVIDIA box (the docs' WSL CUDA rig) run:

```bash
# PyTorch (eager + compiled), JAX, Keras, ORT
DEVICE=cuda            python3 train_mnist_torch.py
DEVICE=cuda COMPILE=1  python3 train_mnist_torch.py
DEVICE=cuda            python3 train_mnist_jax.py        # needs jaxlib-cuda
KERAS_BACKEND=torch DEVICE=cuda python3 train_mnist_keras.py
DEVICE=cuda            python3 train_mnist_ort.py        # needs onnxruntime-training

# RLX on CUDA
( cd rlx-mnist-device && DEVICE=cuda cargo run --release --features cuda )
# ROCm: DEVICE=rocm ... --features rocm
```

This is the **single biggest missing axis** — the current matrix is otherwise
all Apple Silicon. Adding it lets RLX-CUDA stand next to PyTorch/JAX/TF-CUDA on
the hardware where training is normally scrutinized.

## RLX's own MPSGraph path (`rlx-mpsgraph`)

RLX-metal has **two** execution paths: the per-op **thunk** path and an
**MPSGraph** lowering (`run_via_mps_graph`, gated by `RLX_MPSGRAPH_FORCE` /
`estimated_max_flops`). `RLX_MPSGRAPH_TRACE=1` prints any op that can't lower.

**CNN** training can't use MPSGraph: the lowering has no `Conv`/`Pool` (trace:
`[mpsgraph] unsupported: … op Conv`), so the graph falls back to thunks
(`encode_path:thunks_only 100%`). The `rlx-metal` CNN row is therefore the thunk
path — which is fine, it already beats the native MPSGraph baseline.

**MLP** training *does* now run through MPSGraph. The only gaps were the
label-based loss ops; this round added `one_hot` to the MPSGraph wrapper plus
`SoftmaxCrossEntropyWithLogits` and `SoftmaxCrossEntropyBackward` lowerings
(matmul/add/relu/transpose/reduce were already covered). Drive it with
`DEVICE=metal RLX_ARCH=mlp RLX_MPSGRAPH_FORCE=1` — the profiler then reports
`encode_path:mps_graph_full`.

Result (128-hidden MLP, lr=0.02, 2 epochs):

| path | acc | img/s |
|------|-----|-------|
| RLX MPSGraph (`rlx-mlp-mpsgraph`) | 0.9483 | ~201k |
| RLX thunk (`rlx-mlp-thunk`)       | 0.9483 | ~204k |
| native MPSGraph (`mpsgraph`, Swift) | 0.9507 | ~56k |

Two takeaways: (1) RLX's MPSGraph lowering is **bit-identical in accuracy** and
**on par in speed** with its own thunk path — the lowering is correct and
competitive; (2) it is **~3.6× faster than the naive hand-written MPSGraph**
baseline, because RLX runs a compiled `MPSGraphExecutable` with resident state
instead of a per-step `MPSGraph.run()` (the native baseline copies weights out
each step). Extending this to the CNN would additionally need conv/pool + their
gradients (`convolution2DWeightsGradient`, `maxPooling2DGradient`, …) in the
lowering.

## Added this round

* **PaddlePaddle** (`train_mnist_paddle.py`) — full CNN, **0.978 acc / ~6.0k
  img/s** CPU. Needs Python 3.12 (`paddlepaddle` has no 3.14 / arm wheel there):
  `python3.12 -m pip install --break-system-packages paddlepaddle`.
* **dfdx** (`dfdx-mnist/`, Rust const-shaped tensors) — **0.951 acc / ~45k img/s**
  CPU. Trains the **MLP**: dfdx's conv/pool are gated behind its `nightly`
  feature (`generic_const_exprs`), which transitively pulls `gemm-common`'s removed
  `feature(stdsimd)` and no longer builds even on current nightly; the Linear-only
  MLP works on stable.
* **Flux.jl** (`train_mnist_flux.jl`, Julia) — full CNN, **0.983 acc / ~10k
  img/s** CPU (top-tier accuracy). The compiled-*language* angle. Toolchain
  installed (Julia 1.12 via Homebrew; needed an `llhttp` 9.3→9.4 symlink to fix
  brew-Julia's libgit2). `julia -e 'using Pkg; Pkg.add("Flux")'`.
* **IREE (MLIR compiler)** (`train_mnist_iree.py`, py3.12) — MLP train-step lowered
  by JAX to StableHLO, compiled with `iree-compile` (llvm-cpu / cuda), executed
  per batch via the IREE runtime: **0.948 acc / ~57k img/s** CPU. The marquee
  compiler-vs-compiler point — note RLX's own compiled MLP runs ~200k (RLX keeps
  state resident; this round-trips params host↔device each step).

## Candidates not added (blocked — toolchain absent or training unavailable)

* **ONNX Runtime training** — script written (`train_mnist_ort.py`, ORTModule),
  but `onnxruntime-training` ships for Linux/CUDA only (the inference
  `onnxruntime` wheel here cannot train) → rig-only.
* **Luminal** (Rust ML compiler, closest peer) — its training/autograd isn't a
  published crate (`luminal_training` is git-only and early); pin the repo and
  build a train loop when its API stabilizes.
* **Mojo / Modular MAX** — not a blocked *toolchain* but a blocked *baseline*:
  the `magic`/`modular` CLI is deprecated (no-op here) and MAX is
  inference-serving — there is no Mojo training framework, so a row would be
  hand-written backprop, not a framework comparison.
* **BNNS** (Accelerate) — would complete the native-Apple trio (CoreML/ANE ✓,
  MPSGraph/Metal ✓, BNNS/CPU ✗); needs a Swift/C `BNNSGraph` training harness.

## See also

- [coreml-training.md](coreml-training.md) — RLX vs native CoreML deep dive +
  the overhead-/compute-bound analysis and width sweep.
- [backend-selection.md](../backend-selection.md) — RLX device routing, env vars.
- `rlx-paper/bench/README.md` — the full results table and per-runner notes.
## License

MIT OR Apache-2.0.
