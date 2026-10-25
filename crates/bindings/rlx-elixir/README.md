# rlx for Elixir

Elixir bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)) through a NIF: graphs, sessions, device selection,
autodiff, training, weights, text and distributed nodes. `rlx-bindgen`
generates the NIF (`c_src/rlx_nif.c`), the API (`lib/rlx/generated.ex`) and the
copy of `rlx.h` from rlx-bind's tables; `lib/rlx/support.ex`,
`lib/rlx/manual.ex` and `c_src/rlx_nif_support.c` are hand-written. Every call
that does work runs on a dirty scheduler, so a blocking call (a mesh rank)
never stalls the VM.

```elixir
alias Rlx.{Compiled, Graph, Session}

g = Graph.new(name: "mlp")
x = Graph.input(g, "x", [1, 4])
w = Graph.param(g, "w", [4, 4])
Graph.set_outputs(g, [Graph.matmul(g, x, w)])
m = Session.compile(Session.new(device: "cpu"), g)   # compile consumes g
Compiled.set_param(m, "w", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
Compiled.run(m, %{"x" => [1, 2, 3, 4]})
```

## Build

```bash
cargo build -p rlx-capi
cd crates/bindings/rlx-elixir
make RLX_CAPI_LIB_DIR=$PWD/../../../target/debug      # priv/rlx_nif.so
mix compile
mix run test/test_rlx.exs
mix run examples/inference.exs
```

The NIF is `RLX_NIF` when set, else `priv/rlx_nif` of the `:rlx` app. Needs
Elixir 1.18+ (the built-in `JSON` module).

## Conventions

| | |
|---|---|
| Handles | resources freed by the GC, or now with `close/1` |
| Modules | one per class (`Rlx.Graph.matmul(g, a, b)`); free functions in `Rlx` (`Rlx.grad/2`, `Rlx.devices/0`) |
| Constructors / statics | `Rlx.Graph.new(name: …)`, `Rlx.Gguf.open(path)`, `Rlx.Group.connect(rank, world, options: …)` |
| Optional parameters | a keyword list with the tables' defaults (`Rlx.Graph.softmax(g, x, axis: 1)`); an unknown key raises |
| Tensors | lists of numbers; named tensors maps |
| Errors | raise `Rlx.Error` with `kind` (`:type`, `:range`, `:runtime`, `:panic`, `:not_found`) |
| Options in, reports out | maps → JSON; reports come back as maps |
| Hand-written | `Optimizer.step/3` (returns the updated params), `Trainer.run/4` (functions), in `lib/rlx/manual.ex` |

Names: snake_case, as in the Python column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).

## Examples

| | |
|---|---|
| [`inference.exs`](examples/inference.exs) | compile a classifier for a device, set weights, run |
| [`training.exs`](examples/training.exs) | the Trainer with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`loading.exs`](examples/loading.exs) | GGUF write / read, safetensors into a graph, quantize |
| [`distributed.exs`](examples/distributed.exs) | three ranks as BEAM processes in one VM: collectives, pipelined inference, data-parallel training |
| [`backends.exs`](examples/backends.exs) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`models.exs`](examples/models.exs) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples (with
`RLX_TEST_DOCKER=1`, in a container when the toolchain is not installed).

License: MIT OR Apache-2.0.
