# rlx for R

An R package for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)): graphs, sessions, device selection, autodiff,
training, weights, text and distributed nodes. `rlx-bindgen` generates the
`.Call` glue (`src/rlx_r.c`), the R API (`R/generated.R`) and `NAMESPACE` from
rlx-bind's tables; `R/support.R` and `src/rlx_r_support.c` are hand-written.

```r
library(rlx)
g <- Graph$new(name = "mlp")
x <- g$input("x", c(1, 4))
w <- g$param("w", c(4, 4))
g$set_outputs(g$matmul(x, w))
m <- Session$new(device = "cpu")$compile(g)   # compile consumes g
m$set_param("w", rlx_flat(diag(4)))
m$run(list(x = c(1, 2, 3, 4)))
```

## Build

```bash
cargo build -p rlx-capi
RLX_CAPI_LIB_DIR=$PWD/target/debug R CMD INSTALL crates/bindings/rlx-r
Rscript crates/bindings/rlx-r/tests/test_rlx.R
Rscript crates/bindings/rlx-r/inst/examples/inference.R
```

The package needs `jsonlite`. `RLX_CAPI_LIB_DIR` names the directory holding
`librlx_capi` at install time (it is linked with an rpath to it).

## Conventions

| | |
|---|---|
| Handles | `Graph`, `Session`, `Compiled`, … — external pointers freed by the GC, or now with `obj$close()` |
| Methods | `obj$method(...)` (`g$matmul(x, w)`) |
| Constructors / statics | `Graph$new(...)`, `Gguf$open(path)`, `Group$connect(rank, world, options = ...)` |
| Free functions | prefixed `rlx_` (`rlx_grad`, `rlx_devices`), so they mask nothing in base R |
| Optional parameters | R defaults (`g$softmax(x, axis = 1)`) |
| Tensors | numeric vectors, flat and **row-major** (`rlx_flat(A)` / `rlx_unflat(v, shape)` convert R's column-major arrays) |
| Named tensors | named lists; tensor maps come back as named lists |
| Errors | conditions of class `rlx_<kind>` and `rlx_error` (`tryCatch(…, rlx_range = …)`), with `$kind` |
| Options in, reports out | lists → JSON (`jsonlite`; wrap a single value in `list()` for an array of one) |
| Hand-written | `Optimizer$step` (returns the updated params), `Trainer$run` (functions as callbacks), in `R/support.R` |

Names: snake_case, as in the Python column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).
R is single-threaded, so a mesh rank is a process (`Rscript distributed.R rank world peers`).

## Examples

| | |
|---|---|
| [`inference.R`](inst/examples/inference.R) | compile a classifier for a device, set weights, run |
| [`training.R`](inst/examples/training.R) | the Trainer with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`loading.R`](inst/examples/loading.R) | GGUF write / read, safetensors into a graph, quantize |
| [`distributed.R`](inst/examples/distributed.R) | one process per rank: collectives, pipelined inference, data-parallel training |
| [`backends.R`](inst/examples/backends.R) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`models.R`](inst/examples/models.R) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples (with
`RLX_TEST_DOCKER=1`, in a container when the toolchain is not installed).

License: MIT OR Apache-2.0.
