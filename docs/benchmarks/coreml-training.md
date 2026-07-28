# CoreML on-device training: RLX vs native `MLUpdateTask`

RLX can train on Apple Silicon through CoreML (gradients lowered onto the device
via `CoremlTrainingSession`). This page documents how that path compares to a
**native CoreML baseline** (Swift, `MLUpdateTask`) on the same MNIST MLP, and —
more importantly — why the naive headline number is misleading and how to read
the benchmark correctly.

**TL;DR.** On a *tiny* model the benchmark is **overhead-bound**, not
compute-bound, so it measures framework overhead rather than the CPU/GPU/ANE.
In that regime RLX looks ~2.6× faster than `MLUpdateTask` and the compute-unit
choice (`cpu`/`gpu`/`ane`/`all`) makes no difference. Grow the model until it is
**compute-bound** and the picture inverts: native CoreML wins ~2–3× and the
ANE/GPU finally pull ahead of CPU. **RLX wins tiny/overhead-bound on-device
training; native CoreML wins compute-bound training.**

## Setup

Both train the identical net and recipe:

* arch: `784 → InnerProduct(H) → ReLU → InnerProduct(10) → Softmax`, `H=128` baseline
* init: uniform He `U(-1,1)·√(2/fan_in)`, biases 0
* loss/optim: categorical cross-entropy, SGD `lr=0.02, momentum=0.9`, batch 128
* data: MNIST, normalized to `[-1, 1]`; accuracy on the 10k test set

Implementations (both under `rlx-paper/bench/`):

| | what runs | code |
|--|-----------|------|
| **RLX** | `grad_with_loss` graph compiled to CoreML; per-step predict → host SGD | `rlx-mnist-device` (`DEVICE=ane RLX_ARCH=mlp`) |
| **native** | updatable CoreML model trained by `MLUpdateTask` | `coreml-swift/` (`make_model.py` + Swift driver) |

On-device training only supports the **legacy NeuralNetwork** format (not ML
Program), so the baseline authors the model with
`NeuralNetworkBuilder.make_updatable()`.

### Compute-unit selector

Both expose all four `MLComputeUnits`, same value names:

* native: `COMPUTE_UNITS=cpu|gpu|ane|all`
* RLX: `RLX_COREML_UNITS=cpu|gpu|ane|all` (→ `MLComputeUnits.cpuOnly /
  .cpuAndGPU / .cpuAndNeuralEngine / .all`, resolved in
  `rlx_coreml::default_compute_units`)

### Width selector (to probe the regime)

* native: `HIDDEN=<n> OUT=model_h<n>.mlmodel .venv/bin/python make_model.py`
* RLX: `HIDDEN=<n>` env on `rlx-mnist-device`

## Results

MNIST MLP, 2 epochs, `lr=0.02`, Apple Silicon. img/s = `epochs·N / train_s`.
Tiny-model runs are sub-second so they swing ±10–15% under load; values below are
clean unloaded samples.

### hidden=128 — overhead-bound

| MLComputeUnits | RLX acc / img/s | native acc / img/s |
|----------------|-----------------|--------------------|
| cpuOnly            | 0.947 / **285k** | 0.956 / 108k |
| cpuAndGPU          | 0.947 / 278k | 0.956 / 107k |
| cpuAndNeuralEngine | 0.947 / 280k | 0.956 / 107k |
| all                | 0.947 / 270k | 0.956 / 108k |

### hidden=4096 — compute-bound

One consistent pass (these multi-second runs are still ±15% load-sensitive; an
unloaded native `all` has been seen up to ~25k):

| MLComputeUnits | RLX acc / img/s | native acc / img/s |
|----------------|-----------------|--------------------|
| cpuOnly            | 0.961 / **12.0k** | 0.970 / 14.3k |
| cpuAndGPU          | 0.961 / 11.8k | 0.970 / 15.8k |
| cpuAndNeuralEngine | 0.961 / 9.0k | 0.970 / **17.6k** |
| all                | 0.961 / 7.7k | 0.970 / 17.3k |

Note the two diagonals: **native gets faster** toward `ane`/`all` (it uses the
accelerators), while **RLX gets slower** (cpu best) — its fp32 graph can't run on
the ANE, so the `ane`/`all` hints only add fallback/placement overhead.

## Why the tiny-model numbers are misleading

The 784-128-10 net is ~0.1 MMAC/sample (~0.6 MFLOP fwd+bwd). At the rates above
that is well under 1% of either engine's FLOP ceiling — the loop is dominated by
per-step *framework overhead*, not arithmetic. Two consequences:

### Why `cpu` ≥ `ane` at small sizes

The compute-unit hint is effectively a no-op for the tiny model:

* The graph is **fp32**; the **ANE is fp16-only**, so CoreML cannot place fp32
  work there and falls back to CPU/GPU regardless of the hint.
* Training ops (gradients, optimizer, softmax-CE backward) have thin ANE
  support, and 0.1 MMAC is far too little to amortize ANE dispatch + the host↔ANE
  copies.

So all four settings land on the same hardware (CPU/GPU); `cpuOnly` is marginally
best because it skips the (unused) placement attempts. **Confirmed** by the
width sweep: at `hidden=4096` the native baseline's `all`/`ane` clearly beat
`cpu` — the accelerators only help once there is real work.

### Why native looks slower than RLX

At `hidden=128` it is pure overhead. `MLUpdateTask` is a general training
runtime: it iterates an `MLBatchProvider` of 60k boxed feature-provider objects,
schedules, and bookkeeps per batch. RLX feeds raw contiguous `f32` slices and
runs a lean predict + host SGD, so its per-step overhead is lower → it wins the
overhead-bound regime.

This **reverses** once compute dominates (`hidden=4096`): native is ~2× faster
because it (a) keeps weights **resident on-device** and (b) uses CoreML's
optimized training kernels (and engages ANE/GPU). RLX's CoreML path, by
contrast:

* **re-feeds every trainable weight as a graph input each step** — negligible at
  128 hidden, but ~13 MB/step at 4096 (`784·4096·4` bytes for `w1` alone);
* runs an **fp32** graph, so it never uses the ANE.

Native is also consistently ~1% more accurate (default epoch shuffling + a
different RNG seed; not a framework-quality gap).

## How to reproduce

```bash
cd rlx-paper/bench/coreml-swift
python3.12 -m venv .venv && .venv/bin/pip install coremltools numpy
.venv/bin/python make_model.py                 # hidden=128 (default)
HIDDEN=4096 OUT=model_h4096.mlmodel .venv/bin/python make_model.py
swift build -c release

# native, all compute units, both widths
for u in cpu gpu ane all; do COMPUTE_UNITS=$u MODEL=mnist_mlp_updatable.mlmodel ./.build/release/coreml-mnist; done
for u in cpu gpu ane all; do COMPUTE_UNITS=$u MODEL=model_h4096.mlmodel        ./.build/release/coreml-mnist; done

# RLX, all compute units, both widths
cd ../rlx-mnist-device && cargo build --release --features coreml
for u in cpu gpu ane all; do DEVICE=ane RLX_ARCH=mlp RLX_LR=0.02 RLX_COREML_UNITS=$u                 ./target/release/rlx-mnist-device; done
for u in cpu gpu ane all; do DEVICE=ane RLX_ARCH=mlp RLX_LR=0.02 RLX_COREML_UNITS=$u HIDDEN=4096 ./target/release/rlx-mnist-device; done
```

The CSV (`rlx-paper/bench/results/mnist_training.csv`) `rlx-coreml-mlp-*` and
`coreml-swift-*` rows are the **128-hidden (overhead-bound) regime** — read them
as "RLX has the leaner training loop," not "RLX out-computes the ANE."

## RLX improvement opportunities surfaced

The compute-bound regime exposes two real (fixable) limits of the CoreML
training path, not bugs:

1. **On-device weight residency.** Re-feeding trainable weights as inputs every
   step costs `O(params)` host↔device traffic per step. Keeping them resident
   (CoreML state / persistent buffers) would remove the dominant cost at scale.
2. **fp16 / ANE path.** The fp32 graph can never use the ANE.
   `CoremlTrainingSession::with_precision_policy(AutoMixed)` lowers an fp16 graph
   that `default_compute_units` routes to `CpuAndNeuralEngine` — wiring that into
   the trainer would let RLX actually use the ANE on compute-bound models.
   (Unset `RLX_COREML_UNITS` keeps **fp32** graphs on CPU+GPU — BNNS-safe for
   large programs; only f16 / `RLX_COREML_UNITS=ane` target the Neural Engine.)

See also: [op-coverage.md](../op-coverage.md) (per-backend op support, incl. ANE)
and [backend-selection.md](../backend-selection.md) (`RLX_COREML_UNITS` and
device routing).
## License

MIT OR Apache-2.0.
