# rlx for Odin

Odin bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)): graphs, sessions, device selection, autodiff,
training, weights, text and distributed nodes. `rlx-bindgen` generates
`rlx/generated.odin` (package `rlx`) from rlx-bind's tables;
`rlx/support.odin` is hand-written.

```odin
import rlx "rlx:rlx"

g := rlx.must(rlx.graph_new(name = "mlp"))
x := rlx.must(rlx.input(g, "x", {1, 4}))
w := rlx.must(rlx.param(g, "w", {4, 4}))
rlx.set_outputs(g, {rlx.must(rlx.matmul(g, x, w))})
m := rlx.must(rlx.session_compile(rlx.must(rlx.session_new(device = "cpu")), g)) // consumes g
rlx.compiled_set_param(m, "w", {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1})
```

## Build

```bash
cargo build -p rlx-capi
cd crates/bindings/rlx-odin
L="-extra-linker-flags:-L$PWD/../../../target/debug -Wl,-rpath,$PWD/../../../target/debug"
odin run tests -collection:rlx=. "$L"
odin run examples/inference -collection:rlx=. "$L"
```

Functions behind a feature the default library lacks need
`-define:RLX_<FEATURE>=true` (`RLX_FPGA`).

## Conventions

| | |
|---|---|
| Handles | `Graph`, `Session`, … — `distinct rawptr`s, freed with `<class>_close` (Odin has no finalizers) |
| Names | the graph's ops unprefixed (`rlx.matmul(g, a, b)`), other members prefixed by class (`rlx.compiled_run`, `rlx.gguf_open`); a builtin or keyword name gets `_` (`abs_`, `clamp_`, `where_`, `cast_`) |
| Results | `(value, Error)`; `rlx.must(…)` unwraps or panics, so calls nest. `Error` is the C status (`.None`, `.Range`, …); `rlx.last_error()` is its message |
| Optional parameters | Odin defaults, so named arguments work (`rlx.softmax(g, x, axis = 1)`) |
| Tensors | `[]f32`, passed without copying; named tensors `map[string][]f32` (`#+feature dynamic-literals` for map literals) |
| Options in, reports out | JSON text in; `json.Value` out; typed reports as slices |
| Allocation | results use `context.allocator`; a `trainer_run` batch must outlive its callback (`make` it — slice literals live on the callback's stack) |
| Hand-written | `optimizer_step` (in place), `trainer_run` (procs), in `rlx/support.odin` |

Names: snake_case, as in the Python column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).

## Examples

| | |
|---|---|
| [`inference`](examples/inference/main.odin) | compile a classifier for a device, set weights, run |
| [`training`](examples/training/main.odin) | the Trainer with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`loading`](examples/loading/main.odin) | GGUF write / read, safetensors into a graph, quantize |
| [`distributed`](examples/distributed/main.odin) | three ranks on threads: collectives, pipelined inference, data-parallel training |
| [`backends`](examples/backends/main.odin) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`models`](examples/models/main.odin) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples (with
`RLX_TEST_DOCKER=1`, in a container when the toolchain is not installed).

License: MIT OR Apache-2.0.
