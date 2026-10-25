# rlx-linalg

Dense linear algebra for RLX — eigh / svd / qr / cholesky /
solve_triangular / diag_extract / diag_set / trace via LAPACK.
**Downstream package**: registers against rlx's custom-op scaffold
without touching `rlx-ir`, `rlx-opt`, or `rlx-runtime` core.

## What's here

- **`cholesky(A, lower)`** — symmetric positive-definite factor.
- **`solve_triangular(A, B, lower, transpose_a)`** — `op(A) · X = B`.
- **`eigh(A)`** — symmetric eigendecomposition; returns `(eigvals, eigvecs)`
  with eigenvalues ascending.
- **`qr(A)`** — economy QR for tall matrices.
- **`svd(A)`** — full / economy SVD via LAPACK `dgesvd`.
- **`logdet`** / **`slogdet`**, **`pinv`**, **`lstsq`**, **`expm`**,
  **`polar`**, **`kron`** — the rest of the graph ops.
- **`diag_extract(A)`** / **`diag_set(v)`** / **`trace(A)`** — diagonal
  utility ops with full forward + JVP rules.
- **`register()`** — call once per process to publish all of the above
  via `OpExtension` (shape inference + autodiff) and `CpuKernel`
  (LAPACK dispatch).

## Forward + reverse + forward-mode AD

Every op above has a closed-form VJP (matrix calculus identities, no
re-decomposition through autograd) and JVP (push-forward through the
same identities). See `rlx-runtime/tests/cpu_jvp_linalg.rs` for
finite-difference parity tests.

## Host-side, bit-for-bit

Plain functions on `&[f64]` for ports that must match a Python reference to the
last bit, not to a tolerance:

- **`numpy`** — `np.dot` and `np.matmul` (`@`) for float64, routed to BLAS the
  way NumPy 2.x routes them (the routing decides the last bits: `gemm` vs `gemv`
  vs `ddot`, and `dsyrk` when both operands are one buffer as `A` and `Aᵀ`).
  `algos::lstsq_numpy` is `np.linalg.lstsq`.
- **`lsap::linear_sum_assignment`** — a line-for-line port of SciPy's
  `rectangular_lsap.cpp`, including its choice among tied optima.
- **`algos::eigh_torch`** / **`eigh_torch_f32`** — `torch.linalg.eigh`: LAPACK
  `?syevd` on the lower triangle, eigenvectors in columns; the f32 version is
  bit-identical to torch 2.11 on macOS. `algos` also exposes the LAPACK-backed
  `cholesky`, `eigh`, `eigvals`, `qr`, `gesv`, `pinv` and `expm` used by the ops.

## Install

```toml
[dependencies]
rlx-linalg = "0.2"
```

## Quickstart

```rust
rlx_linalg::register();   // once per process

let mut g = rlx_ir::Graph::new("eigh");
let a = g.input("a", rlx_ir::Shape::new(&[4, 4], rlx_ir::DType::F64));
let (eigvals, eigvecs) = rlx_linalg::eigh(&mut g, a);
g.set_outputs(vec![eigvals, eigvecs]);
```

## License

MIT OR Apache-2.0.