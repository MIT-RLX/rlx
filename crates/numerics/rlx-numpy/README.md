# rlx-numpy

Host-side numpy numerics reproduced **bit-for-bit**, for ports whose parity tests
compare against numpy at the ULP level:

| function | numpy | why a plain Rust spelling misses |
|---|---|---|
| `sum`, `pairwise_sum` | `np.sum`, and the pairwise loop under it, over a contiguous f64 run | numpy sums in 8-wide blocks, recursing pairwise above 128 |
| `round_half_even` | `np.round` | `f64::round` rounds halves away from zero |
| `linspace` | `np.linspace` | numpy multiplies by one precomputed step and pins the endpoint |
| `sin_cos_separate` | `np.sin`, `np.cos` | LLVM merges them into `sincos`, which is not bit-identical |
| `tanh` | `np.tanh` (f64, NumPy ≥ 2.0 with SIMD+FMA: arm64, AVX-512) | NumPy uses its own SVML-derived kernel, not libm; they differ on 18 % of arguments |

`pairwise_sum_f32` is the f32 analogue used to match torch's CPU reduction kernels.

## `torch` — torch CPU, bit-for-bit

| function | torch | why |
|---|---|---|
| `sum_f32`, `sum_f64`, `sum_rows_f32`, `sum_rows_f64`, `mean_f32` | `Tensor.sum()` / `.mean()` on CPU | ATen's `cascade_sum`: 4 (f32) / 2 (f64) lanes on arm64, ILP 4, a four-level cascade — neither pairwise nor sequential |
| `exp_f32` | `torch.exp` (float32, arm64) | SLEEF's `xexpf`, which differs from libm `expf` in about 1 value in 10 |

Measured against torch 2.11 on macOS arm64. These are the single-thread spellings:
past torch's parallel grain (32768 elements) its result depends on the thread count.

## `npy`

`read_f32` / `write_f32`: `np.load` / `np.save` for C-order little-endian `<f4`
arrays (format 1.0 written, 1.0–3.0 read).

Tests pin values produced by numpy 2.4.3 (`tests/numpy_pins.rs`) and, for `tanh`,
numpy 2.3.5 (`tests/tanh.rs`).
For numpy/torch random streams, see `rlx-rng`.

## License

MIT OR Apache-2.0.
