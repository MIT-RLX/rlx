# rlx-sparse

Sparse linear algebra for RLX — CSR LU factor, sparse mat-vec,
Conjugate Gradient. **Downstream package**: registers against rlx's
custom-op scaffold (`rlx_ir::OpExtension` + per-backend `op_registry`)
without touching `rlx-ir`, `rlx-opt`, or `rlx-runtime` core.

This is the JAX-shaped pattern for domain-specific ops: build the
extension surface in core, ship the actual op in a separate crate.

## What's here

- **`SparseTensor`** — CSR (`{row_ptrs, col_indices, values}`) +
  density / nnz / shape metadata.
- **`Op::Custom("rlx_sparse.lu")`** — symbolic + numeric CSR LU.
- **`Op::Custom("rlx_sparse.spmv")`** — sparse mat-vec.
- **`Op::Custom("rlx_sparse.cg")`** — Conjugate Gradient solver.
- **`register()`** — call once per process to publish the
  `OpExtension` (shape inference + autodiff) plus per-backend kernels
  (CPU always, Metal / MLX behind features).

## Install

```toml
[dependencies]
rlx-sparse = "0.1"
```

For Apple GPU acceleration:

```toml
rlx-sparse = { version = "0.1", features = ["metal", "mlx"] }
```

## Quickstart

```rust
rlx_sparse::register();   // once per process

let mut g = rlx_ir::Graph::new("sparse");
let csr = rlx_sparse::SparseTensor::from_dense(&dense, n, n);
let x = rlx_sparse::cg(&mut g, csr, b);
g.set_outputs(vec![x]);
```

## Status

CG + SpMV production-ready on CPU. Metal / MLX kernels behind their
respective features.

## License

GPL-3.0-only.