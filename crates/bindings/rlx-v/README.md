# rlx for V

V bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)): graphs, sessions, device selection, autodiff,
training, weights, text and distributed nodes. `rlx-bindgen` generates
`generated.v` (module `rlx`) and the copy of `rlx.h` the module includes from
rlx-bind's tables; `support.v` and `manual.v` are hand-written.

```v
import rlx

g := rlx.Graph.new(name: 'mlp')!
x := g.input('x', [1, 4])!
w := g.param('w', [4, 4])!
g.set_outputs([g.matmul(x, w)!])!
m := rlx.Session.new(device: 'cpu')!.compile(g)! // compile consumes g
m.set_param('w', [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])!
println(m.run({'x': [1.0, 2, 3, 4]})!)
```

## Build

Put this directory on `VMODULES` as `rlx` (a symlink named `rlx` in a modules
directory), then:

```bash
cargo build -p rlx-capi
F="-cflags -L$PWD/target/debug -cflags -Wl,-rpath,$PWD/target/debug"
v $F run crates/bindings/rlx-v/tests/test_rlx.v
v $F run crates/bindings/rlx-v/examples/inference.v
```

A function behind a feature the default library lacks needs `-d rlx_<feature>`.

## Conventions

| | |
|---|---|
| Handles | `&rlx.Graph`, `&rlx.Session`, … — V's GC runs no destructors, so free with `close()` |
| Constructors / statics | `rlx.Graph.new(…)`, `rlx.Gguf.open(path)`, `rlx.Group.connect(rank, world, options: …)` |
| Optional parameters | a `@[params]` struct, so `g.softmax(x, axis: 1)` |
| Results | every call returns `!T`; the error is an `RlxError` with `kind` |
| Optional results | `''` / NaN when absent (V has no result of an option) |
| Tensors | `[]f64` (V's float literals) and `map[string][]f64` |
| Options in, reports out | JSON text in; `json2.Any` out; typed reports as arrays |
| Hand-written | `Optimizer.step(mut params, grads)` (in place), `Trainer.run` (closures), in `manual.v` |

Names: snake_case, as in the Python column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).
A mesh rank is a process (`./distributed rank world peers`).

## Examples

| | |
|---|---|
| [`inference.v`](examples/inference.v) | compile a classifier for a device, set weights, run |
| [`training.v`](examples/training.v) | the Trainer with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`loading.v`](examples/loading.v) | GGUF write / read, safetensors into a graph, quantize |
| [`distributed.v`](examples/distributed.v) | one process per rank: collectives, pipelined inference, data-parallel training |
| [`backends.v`](examples/backends.v) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`models.v`](examples/models.v) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples (with
`RLX_TEST_DOCKER=1`, in a container when the toolchain is not installed).

License: MIT OR Apache-2.0.
