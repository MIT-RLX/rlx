# rlx for OCaml

OCaml 4.14+ / 5 bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)): graphs, sessions, device selection, autodiff,
training, weights, text and distributed nodes. `rlx-bindgen` generates the C
stubs (`src/rlx_stubs.c`), the API (`src/generated.ml`) and the copy of
`rlx.h` from rlx-bind's tables; `support.ml`, `rlx.ml` and
`rlx_ml_support.c` are hand-written. No dependencies beyond the compiler's
`unix` and `threads` (JSON is built in: `Rlx.Json`).

```ocaml
let () =
  let open Rlx in
  let g = Graph.create ~name:"mlp" () in
  let x = Graph.input g "x" [| 1; 4 |] in
  let w = Graph.param g "w" [| 4; 4 |] in
  Graph.set_outputs g [| Graph.matmul g x w |];
  let m = Session.compile (Session.create ~device:"cpu" ()) g in   (* compile consumes g *)
  Compiled.set_param m "w" [| 1.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 1. |];
  ignore (Compiled.run m [ ("x", [| 1.; 2.; 3.; 4. |]) ])
```

## Build

```bash
cargo build -p rlx-capi
cd crates/bindings/rlx-ocaml
RLX_CAPI_LIB_DIR=$PWD/../../../target/debug dune build
RLX_CAPI_LIB_DIR=$PWD/../../../target/debug dune test
RLX_CAPI_LIB_DIR=$PWD/../../../target/debug dune exec examples/inference.exe
```

## Conventions

| | |
|---|---|
| Handles | `Graph.t`, `Session.t`, … — custom blocks finalized by the GC, or now with `close` |
| Modules | one per class (`Graph.matmul g a b`, `Compiled.run c inputs`); free functions at top level (`grad`, `devices`) |
| Constructors / statics | `Graph.create ()`, `Gguf.open_ path`, `Group.connect ~options rank world` |
| Optional parameters | labelled optional arguments with the tables' defaults (`Graph.softmax ~axis:1 g x`) |
| Tensors | `float array`; named tensors `(string * float array) list` |
| Errors | `Rlx.Error (kind, message)`, `kind` one of `Type_error`, `Range_error`, `Runtime_error`, `Panic`, `Not_found_error` |
| Options in, reports out | `Rlx.Json.t` |
| Blocking | calls release the runtime lock, so other threads run meanwhile |
| Hand-written | `Optimizer.step` (returns the updated params), `Trainer.run` (closures), in `rlx.ml` |

Names: snake_case, as in the Python column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).
A mesh rank is a process (`distributed.exe rank world peers`).

## Examples

| | |
|---|---|
| [`inference.ml`](examples/inference.ml) | compile a classifier for a device, set weights, run |
| [`training.ml`](examples/training.ml) | the Trainer with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`loading.ml`](examples/loading.ml) | GGUF write / read, safetensors into a graph, quantize |
| [`distributed.ml`](examples/distributed.ml) | one process per rank: collectives, pipelined inference, data-parallel training |
| [`backends.ml`](examples/backends.ml) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`models.ml`](examples/models.ml) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples (with
`RLX_TEST_DOCKER=1`, in a container when the toolchain is not installed).

License: MIT OR Apache-2.0.
