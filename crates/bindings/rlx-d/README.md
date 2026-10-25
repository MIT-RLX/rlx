# rlx for D

D bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)): graphs, sessions, device selection, autodiff,
training, weights, text and distributed nodes. `rlx-bindgen` generates
`source/rlx/capi.d` (the `extern (C)` declarations) and
`source/rlx/generated.d` (the API) from rlx-bind's tables; `support.d` and
`manual.d` are hand-written.

```d
import rlx;

auto g = new Graph(name: "mlp");
immutable x = g.input("x", [1, 4]);
immutable w = g.param("w", [4, 4]);
g.setOutputs([g.matmul(x, w)]);
auto m = new Session(device: "cpu").compile(g);   // compile consumes g
m.setParam("w", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
writeln(m.run(["x": [1.0f, 2, 3, 4]]));
```

## Build

D 2.108+ (named arguments), e.g. LDC 1.38+:

```bash
cargo build -p rlx-capi
cd crates/bindings/rlx-d
L="-L-L../../../target/debug -L-lrlx_capi -L-rpath=../../../target/debug"
ldc2 -i -Isource tests/test_rlx.d $L -of=test_rlx && ./test_rlx
ldc2 -i -Isource examples/inference.d $L -of=inference && ./inference
```

With dub, `dub.json` links `rlx_capi` from `RLX_CAPI_LIB_DIR`. Functions behind
a feature the default library lacks need `-version=rlx_<feature>` (`rlx_fpga`).

## Conventions

| | |
|---|---|
| Handles | `Graph`, `Session`, `Compiled`, … — classes freed by the destructor, or now with `close()` |
| Free functions / statics | `grad(g, [w])`, `devices()`, `version_()` (`version` is a keyword); `Gguf.open(path)`, `Group.connect(rank, world, options: …)` |
| Optional parameters | defaults, with named arguments (`g.softmax(x, axis: 1)`); an optional number with no default is a sentinel (`NONE`, `T.max`, NaN) |
| Tensors | `float[]` slices, passed to C without copying; named tensors `["x": […]]` |
| Errors | `RlxException` with `kind` (`RlxErrorKind.range`, …) |
| Options in, reports out | `JSONValue` (`parseJSON(…)`); typed reports as arrays |
| Several results | a named `Tuple` (`runner.run(x).device`) |
| Hand-written | `opt.step(params, grads)` (in place) and `trainer.run(…)` (delegates), UFCS functions in `manual.d` |

Names: camelCase, as in the JS column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).
A mesh rank is a process (`./distributed rank world peers`).

## Examples

| | |
|---|---|
| [`inference.d`](examples/inference.d) | compile a classifier for a device, set weights, run |
| [`training.d`](examples/training.d) | the Trainer with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`loading.d`](examples/loading.d) | GGUF write / read, safetensors into a graph, quantize |
| [`distributed.d`](examples/distributed.d) | one process per rank: collectives, pipelined inference, data-parallel training |
| [`backends.d`](examples/backends.d) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`models.d`](examples/models.d) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples (with
`RLX_TEST_DOCKER=1`, in a container when the toolchain is not installed).

License: MIT OR Apache-2.0.
