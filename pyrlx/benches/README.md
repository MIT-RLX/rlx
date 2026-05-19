# pyrlx benches

Same workload, every framework, every backend each framework can
reach on this machine. One table at the end.

Frameworks supported: **pyrlx**, **PyTorch**, **JAX**, **tinygrad**.

## Run

```sh
# from pyrlx/
python benches/bench_vs_torch_jax.py
```

Output is a per-workload table — rows are frameworks, columns are
devices — followed by a parity check (max abs diff vs the reference
cell, default `pyrlx/cpu`):

```
== mlp  shape=(128, 768)  (x[B, D] @ W[D, D] + b → gelu) ==
                cpu       metal         mps        cuda
pyrlx          0.245 ms   1.310 ms        —           —
pytorch        0.520 ms      —          0.890 ms      —
jax            0.305 ms      —             —          —

parity (max|Δ| vs pyrlx/cpu):
  mlp        pyrlx/metal:   2.4e-7
  mlp        pytorch/cpu:   1.2e-7
  ...
```

## Workloads

| name        | shape (default)        | description                                    |
| ----------- | ---------------------- | ---------------------------------------------- |
| `mlp`       | `(128, 768)`           | x @ W + b → gelu (transformer FFN-ish)         |
| `layernorm` | `(8, 128, 768)`        | LayerNorm over last dim                        |
| `attention` | `(1, 12, 128, 64)`     | SDPA: softmax(QK^T/√d) @ V, no mask            |

Override with `--shape 256,1024` (interpretation depends on workload).

## Flags

| flag           | default                       | what                                     |
| -------------- | ----------------------------- | ---------------------------------------- |
| `--frameworks` | `pyrlx,pytorch,jax`           | filter to a subset                       |
| `--workloads`  | all                           | filter to a subset                       |
| `--shape`      | per-workload                  | override workload shape                  |
| `--iters`      | `100`                         | timed iterations after warmup            |
| `--warmup`     | `10`                          | warmup iterations (compile, kernel cache)|
| `--seed`       | `0`                           | RNG seed for input data                  |
| `--reference`  | `pyrlx/cpu`                   | parity-check baseline                    |

## Install the comparators

The bench skips frameworks that aren't installed. Each is heavy —
install only what you can run:

```sh
# PyTorch (cpu + cuda + mps as available)
uv pip install torch

# JAX (cpu)
uv pip install -U jax
# JAX (cuda)
uv pip install -U "jax[cuda12]"
# JAX on Apple Silicon GPU is via the experimental `jax-metal` —
# see https://developer.apple.com/metal/jax/ for current status.

# tinygrad — every backend ships with the package; pick at runtime
# via env vars (CLANG=1, METAL=1, CUDA=1, HIP=1/AMD=1, GPU=1
# (=OpenCL), WEBGPU=1). The bench probes each automatically.
uv pip install tinygrad
```

## Reading the numbers

- The harness times one closure call per iteration. Each closure
  syncs the device internally before returning, so the timing is
  *end-to-end host latency*, not lazy dispatch time.
- We report median + p99. p99 is in the per-cell errors block; the
  table itself is medians.
- JIT-warm path only. The bench discards the first `--warmup`
  iterations to skip XLA / TorchInductor / pyrlx compile time.
- Inputs are `float32`; outputs are pulled back to host so the
  parity check sees post-blit values. This is *not* the right
  measurement if you only care about steady-state on-device
  throughput — but it is the right measurement for "how long does
  the user wait for an answer."

## What this bench is and isn't

This is a **forward-pass latency** harness. It does not measure:
- training throughput
- multi-stream / multi-GPU
- tokens/sec on a real serving stack
- peak memory

For those, build a separate harness — this one is small on purpose.
