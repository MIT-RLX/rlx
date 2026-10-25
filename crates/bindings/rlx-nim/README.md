# rlx for Nim

Nim 2 bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)): graphs, sessions, device selection, autodiff,
training, weights, text and distributed nodes. `rlx-bindgen` generates
`src/rlx/generated.nim` from rlx-bind's tables; `src/rlx.nim` holds the
hand-written support and includes it.

```nim
import rlx

let g = newGraph(name = "mlp")
let x = g.input("x", [1, 4])
let w = g.param("w", [4, 4])
g.setOutputs([g.matmul(x, w)])
let m = newSession(device = "cpu").compile(g)   # compile consumes g
m.setParam("w", [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
echo m.run({"x": @[1.0, 2, 3, 4]})
```

## Build

The C functions are `dynlib` imports, found at run time:

```bash
cargo build -p rlx-capi
cd crates/bindings/rlx-nim
DYLD_LIBRARY_PATH=../../../target/debug nim r --path:src tests/test_rlx.nim     # LD_LIBRARY_PATH on Linux
DYLD_LIBRARY_PATH=../../../target/debug nim r --path:src examples/inference.nim
```

`-d:rlxLib=<path>` names the library instead of the search path. A function a
build lacks is only looked up if a program uses it.

## Conventions

| | |
|---|---|
| Handles | `Graph`, `Session`, `Compiled`, … — `ref` types freed by `=destroy`, or now with `close()` |
| Methods / constructors / statics | `g.softmax(x, axis = 1)`, `newGraph(…)`, `Gguf.open(path)`, `Group.connect(rank, world, options = …)` |
| Optional parameters | Nim defaults (named arguments work); a parameter with no default is an `Option` (`some("demo")`) |
| Tensors | `float` sequences; named tensors `{"x": @[…]}` |
| Errors | `RlxError` with `kind` (`ekType`, `ekRange`, `ekRuntime`, `ekPanic`, `ekNotFound`) |
| Options in, reports out | `JsonNode` (`%*{…}`); typed reports (`devices()`, shapes) as sequences |
| Renamed | an op named after a keyword gets `Op` (`divOp`, `castOp`) |
| Hand-written | `Optimizer.step` (`var Table` updated in place), `Trainer.run` (procs), in `src/rlx.nim` |

Names: camelCase, as in the JS column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).
A mesh rank is a process (`./distributed rank world peers`).

## Examples

| | |
|---|---|
| [`inference.nim`](examples/inference.nim) | compile a classifier for a device, set weights, run |
| [`training.nim`](examples/training.nim) | the Trainer with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`loading.nim`](examples/loading.nim) | GGUF write / read, safetensors into a graph, quantize |
| [`distributed.nim`](examples/distributed.nim) | one process per rank: collectives, pipelined inference, data-parallel training |
| [`backends.nim`](examples/backends.nim) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`models.nim`](examples/models.nim) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples (with
`RLX_TEST_DOCKER=1`, in a container when the toolchain is not installed).

License: MIT OR Apache-2.0.
