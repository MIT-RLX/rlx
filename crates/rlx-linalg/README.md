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
- **`diag_extract(A)`** / **`diag_set(v)`** / **`trace(A)`** — diagonal
  utility ops with full forward + JVP rules.
- **`register()`** — call once per process to publish all of the above
  via `OpExtension` (shape inference + autodiff) and `CpuKernel`
  (LAPACK dispatch).

## Forward + reverse + forward-mode AD

Every op above has a closed-form VJP (matrix calculus identities, no
re-decomposition through autograd) and JVP (push-forward through the
same identities). See `cpu_jvp_linalg.rs` for finite-difference parity
tests.

## Install

```toml
[dependencies]
rlx-linalg = "0.1"
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

GPL-3.0-only.