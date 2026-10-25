# Language bindings

Every rlx binding — JavaScript (`rlx-js`), Python (`pyrlx`), C (`rlx-capi`),
Swift, Go, Java, C#, Ruby, Julia and Zig, and whatever comes next — is
generated from one place:
[`crates/bindings/rlx-bind`](../crates/bindings/rlx-bind). A new op is one row
there; a new language is one emitter or one generator template, not a third
copy of the API.

```
                      rlx-bind
   ┌──────────────────────────────────────────────┐
   │ for_each_graph_op!   the op table (one row   │
   │                      per graph builder)      │
   │ ops::*               a typed fn per row      │
   │ manifest()           the table as data       │
   │ api.idl / api()      the runtime API table:  │
   │                      every class, method and │
   │                      fn, named per language  │
   │ runtime::*           Session, Compiled,      │
   │                      Trainer, Optimizer,     │
   │                      Tokenizer, Sampler,     │
   │                      check, devices, GGUF,   │
   │                      RLXP, weight loaders    │
   │ enums, ApiError      one parser per option,  │
   │                      one error type          │
   └──────┬──────────────┬───────────────┬────────┘
   macro  │       macro  │               │ macro + rlx-bindgen
          ▼              ▼               ▼
       rlx-js          pyrlx          rlx-capi ──► rlx.h
     (quickrs)        (PyO3)        (extern "C")    │
                                                    ▼  rlx-bindgen (ir.rs: both tables,
                                    Swift · Go · Java · C# · Ruby · Julia · Zig
```

## The op table

`crates/bindings/rlx-bind/src/table.rs` holds one row per graph builder:

```rust
/// NCHW convolution, output shape inferred.
conv2d(input: node, weight: node;
       { kernel_size: pair = [1, 1], stride: pair = [1, 1], padding: pair = [0, 0],
         dilation: pair = [1, 1], groups: usize = 1 }) => |g| {
    g.conv2d(input, weight, kernel_size, stride, padding, dilation, groups)
};
```

* **Name**: snake_case, Python's spelling. JS camelCases it (`conv2d`,
  `attentionKindVdim`; a trailing `_` drops: `where_` → `where`). `[js = "…"]`,
  `[py_also = "…"]`, `[c = "…"]`, `[swift = "…"]` rename per language.
* **Arguments**: `name: kind = default`. The kind fixes the wire type every
  binding converts to, the decoding every binding shares (node ids checked,
  enums parsed, pairs validated) and how the generators spell it — see
  `rlx-bind/src/kinds.rs`. Defaults are neutral literals (`-1`, `1e-5`,
  `"causal"`, `[1, 1]`, `None`) so every generator can print them.
* **Options**: the `; { … }` group is trailing options — one object in JS
  (`conv2d(x, w, {stride: 2})`), keyword-or-positional parameters in Python
  (`conv2d(x, w, stride=2)`), flattened parameters in C, labeled defaulted
  parameters in Swift. `name | alias` lets JS also read a legacy key.
* **Returns**: `-> node` (the default), `pair`, `nodes`, `unit`, `shape`,
  `strings`, `usize`, `string`.
* **Body**: a block over the decoded arguments, run under the panic guard,
  inside a closure returning `ApiResult` — `?` and `bail!` work.

The table expands, at compile time, into `rlx_bind::ops::<name>` (the typed
core function every binding calls), the reflected `manifest()`, the JS methods
(`rlx-js/src/api/bind.rs`) and the Python methods (`pyrlx/src/bind.rs`).

## Adding an op

1. Add the IR builder in `rlx-ir` (see the op checklist in `AGENTS.md`).
2. Add one row to `rlx-bind/src/table.rs`.
3. Run `just gen-bindings` — it regenerates the C glue, `rlx.h`, Swift's
   `GraphOps.swift`, the `Graph` section of `_pyrlx.pyi`, and
   [`bindings-api.md`](bindings-api.md).

rlx-js and pyrlx pick the row up with no edit. Two checks keep it that way:

* `rlx-bind`'s `ir_drift` test fails when `rlx-ir` gains a public builder the
  table neither calls nor lists in `NOT_BOUND` with a reason.
* `rlx-bindgen`'s `up_to_date` test (and `just check-bindings`, part of
  `just ci`) fails when a generated file is stale.

## The runtime API table

Everything past the graph builders — `Session`, `FlexibleSession`, `Compiled`,
`Runner`, `Router`, autodiff, `check`, devices, `Gguf` / `GgufWriter`, weights
and RLXP, `Optimizer`, `Trainer`, `Tokenizer`, `Sampler`, distributed `Group`s
— is one row per member in
[`rlx-bind/src/api.idl`](../crates/bindings/rlx-bind/src/api.idl):

```
/// Routes each run to a device by policy, benchmarking and falling back.
class Router [py=DeviceRouter]
    new(graph: graph, policy: value?)
    run_on(device: str, inputs: inputs) -> outputs
    run_chain(inputs: inputs, hint: str?) -> (device: str, outputs: outputs)
end
fn loopback_peers(world: u32) -> strings
```

Each row has argument kinds (`str`, `json?`, `f32s`, `inputs`, `graph`, …), a
return kind, and its name in each language — Python's spelling under the
Python class, camelCase for JS and Swift, `rlx_<class>_<name>` in C, unless the
row overrides one (`py=`, `js=`, `c=`, `swift=`, plus `py_also` / `js_also` for
extra spellings). A language without a row says so (`js=-`) with a `why=`.
The grammar and the C expansion of every kind are at the top of the file.

The table is a contract, checked from every side:

* `rlx-bindgen` parses rlx-capi and fails unless every row names a C function
  whose parameter types are exactly what the row's kinds expand to, every class
  has its `_free`, and every exported C function is a row (or buffer/output
  plumbing). It then **generates** `rlx-swift/Sources/RLX/API.swift` — one
  Swift member per row; only rows marked `manual` (a callback loop, an
  in-place optimizer step) are hand-written, in `Manual.swift`.
* `rlx-js/tests/api_contract.rs` and `pyrlx/tests/test_api_contract.py` check
  that every row (for the features the build has) resolves to an export, and
  that every export is a row or a `host` helper the table lists (the JS file
  system, NumPy dtype helpers, typed option builders).
* `docs/bindings-api.md` prints the name map of every row.

## Adding a runtime function

1. Implement it once in `rlx-bind/src/runtime/` (plain Rust over
   `rlx-runtime`, options and reports as JSON).
2. Add its row to `api.idl`.
3. Add the C function to `rlx-capi` (a few lines over the `rlx-bind` call;
   `arg::` readers in, `out::` writers out), and the JS and Python methods.
4. `just gen-bindings` — the header, Swift and docs follow. `cargo test -p
   rlx-bindgen`, the contract tests and `just check-bindings` fail until all
   four languages have it.

## Runtime classes

Sessions, compiled graphs, training, text, GGUF, RLXP, the weight loaders and
distributed nodes are plain Rust in `rlx-bind/src/runtime/`, behind features
(`runtime`, `training`, `text`, `gguf-io`, `gguf-convert`, `weights`, `fpga`). Tensors cross as
slices; structured options and reports cross as JSON (`serde_json::Value`),
with keys matched ignoring case and `_` — JS's `weightDecay` and Python's
`weight_decay` are the same option. Unknown options are an error in every
language.

JS and Python wrap these in their own classes by hand — those wrappers are a
few lines per method, and are where language idioms live (zero-copy NumPy views and
reshaped outputs in Python, reusable typed-array scratch in JS, closures in
Swift). The training *loop* (`run`) is the binding's, a callback loop over
`Trainer::step`, so no handle stays borrowed while user code runs; `RunStats`
keeps its summary identical everywhere. Core reports use snake_case keys; JS
camelCases them when it converts one (`fused_ops` → `fusedOps`).

Errors are `ApiError { kind, message }`:

| kind | JS | Python | C |
|------|----|--------|---|
| `Type` | `TypeError` | `ValueError` | `RLX_ERR_TYPE` |
| `Range` | `RangeError` | `ValueError` | `RLX_ERR_RANGE` |
| `Runtime` | `InternalError` | `RuntimeError` | `RLX_ERR_RUNTIME` |
| `Panic` | `InternalError` | `RuntimeError` | `RLX_ERR_PANIC` |
| `NotFound` | `InternalError` | `FileNotFoundError` | `RLX_ERR_NOT_FOUND` |

A `Compiled` checks its boundary in every language, as `CompiledGraph::try_run`
does in Rust ([troubleshooting.md](troubleshooting.md)): an unknown input
name or a buffer of the wrong length is a `Type` error naming it, not a run on
zeros. Compile failures map by cause: an invalid graph is `Type`, an absent
device or an op it cannot lower is `Runtime`, a backend compile panic is
`Panic`. `inputs()` / `params()` / `outputs()` return `{name, dtype, shape}`
rows (`outputs()` in `run` order, `name` empty when the graph gives none);
`output_index(name)` is that output's position, a `Range` error naming the
closest names otherwise. Graph builders name outputs with
`name_output(index, name)`.
`check_bindings(names)` fails on anything declared but not supplied, and
`allow_unbound(names)` marks inputs meant to read zeros.

## The C ABI

`rlx-capi` is the hub every FFI language links (`staticlib` for Swift / iOS,
`cdylib` for runtimes that load a shared library). Conventions, all in
`rlx-capi/src/lib.rs`: an `int32_t` status with `rlx_last_error()`; opaque
handles freed with `rlx_*_free`; arrays as pointer + length; `RLX_NONE` for no
node, `NULL` for any other optional (and for a defaulted string); buffers rlx
allocates freed with `rlx_u32s_free` / `rlx_f32s_free` / `rlx_bytes_free` /
`rlx_string_free`; structured results as JSON text.

For Apple apps, `ios/build-xcframework.sh --capi [--apple]` packages the
static library and `rlx.h` (module `CRLX`) as `RLX.xcframework`; the Swift
package builds on it with `RLX_XCFRAMEWORK=1`
([`rlx-swift/README.md`](../crates/bindings/rlx-swift/README.md)).

The `rlx_graph_<op>` functions are generated (`src/graph_ops.rs`); the runtime
functions are hand-written over `rlx-bind::runtime`, one per `api.idl` row. `include/rlx.h` is
generated by parsing the crate's own sources, so it lists exactly what the
library exports. `examples/train_linear.c` drives it end to end from C.

## Adding a language

**Through the C ABI** (Fortran, Crystal, Pascal, Ada, Racket, …): every FFI
language here is one emitter in `crates/tooling/rlx-bindgen/src/` over
[`ir::Ir`](../crates/tooling/rlx-bindgen/src/ir.rs) — both tables lowered to
functions whose parameters and results use one kind vocabulary (`str`,
`json?`, `f32s`, `inputs`, `rows2`, a handle, …), each already checked against
the C signature. An emitter is:

1. a type per argument / result kind, and how to marshal it (about 25 kinds);
2. a class (or struct, or module) per handle type, freed deterministically and
   by the language's finalizer;
3. how optional parameters read: keyword arguments (Ruby, Julia, C#, Elixir),
   defaults with named arguments (Kotlin, Dart, PHP, Nim, D, Odin), labelled
   arguments (OCaml, Swift), an options struct with defaulted fields (Zig,
   Java, Go, C++, V), an options record with a `def…` value (Haskell),
   name-value pairs (MATLAB);
4. a small hand-written support file: loading the library, the error type, the
   buffer helpers, JSON if the language lacks it, and the two `manual` rows
   (`Optimizer.step` updates arrays in place; `Trainer.run` takes callbacks —
   most languages drive it over `Trainer.step`).

`go.rs`, `java.rs`, `csharp.rs`, `ruby.rs`, `julia.rs`, `zig.rs`, `cpp.rs`,
`node.rs`, `kotlin.rs`, `dart.rs`, `php.rs`, `nim.rs`, `dlang.rs`, `vlang.rs`,
`haskell.rs`, `odin.rs` are 300–600 lines each. A language with no FFI of its
own gets a generated C layer driven by C types alone — `jni.rs` (Kotlin),
`lua.rs`, `matlab.rs` (MEX), `r.rs` (`.Call`), `ocaml.rs` (stubs) and
`elixir.rs` (NIF) each print one C function per rlx-capi function plus the
language side. Add the output to `rlx_bindgen::outputs` so `check-bindings` covers
it, write the six examples (`inference`, `training`, `loading`, `distributed`,
`backends`, `models` — their expected lines are in `rlx-capi/tests/ffi_languages.rs`),
and add the language there so `cargo test -p rlx-capi` runs its tests and
examples. `native_or_docker` runs a language's script with its local
toolchain, or (`RLX_TEST_DOCKER=1`) in its container image against a Linux
build of the library (`just linux-capi-volume`); a language with no threads
runs the distributed example as three processes (`each_example_ranks`).

**Natively** (a language with a Rust embedding story, like PyO3 or quickrs):
write a callback macro over `rlx_bind::for_each_graph_op!` that emits one
method per row — `rlx-js/src/api/bind.rs` and `pyrlx/src/bind.rs` are two
complete examples, each about 200 lines — wrap `rlx_bind::runtime` for the
classes, and add a contract test over `rlx_bind::api::contract` like rlx-js's.

## Models and model plugins

`Model` (feature `models`, on by default in every binding) opens a model by
path or id and runs it by name. It is the binding view of
`rlx_runtime::model`:

- **The built-in `rlxp` provider** runs a graph saved with
  `save_model(graph, path, weights, {task, family, outputs})` on any device.
- **Plugins** add providers at run time. A plugin is a `cdylib` that exports
  `rlx_model_plugin_v1`, a C table at ABI version 1. Plugins load from
  `~/.rlx/plugins` (or `$RLX_HOME/plugins`) with no setting needed, from
  `RLX_MODEL_PLUGINS` (libraries or directories), or with
  `load_model_plugin(path)`. This is how a model collection downstream of rlx
  (LLMs, speech, vision) reaches every language, without rlx depending on it.
- **Linked plugins** cover platforms that can't load a library at run time:
  iOS, Android, wasm and consoles. There the same plugin crate links into the
  app's library (`link_model_plugins!`, feature `linked-plugins` of
  rlx-runtime), and the app calls `rlx_link_model_plugins()` once (Swift:
  `linkModelPlugins()`). `ios/build-xcframework.sh --capi-crate` packages it
  for Swift. See
  [Link one](../crates/bindings/rlx-model-plugin/README.md#link-one-where-nothing-loads).

`open` picks the provider that names itself in `options.provider`, or else the
one whose `probe` scores highest. Each task has its own method on `Model`, in
every language:

| Task | Method |
|---|---|
| `text-generation` | `chat(messages)`, `generate_text(prompt)`, `generate(ids, max_tokens)`, `predict_logits(ids)` |
| `speech-recognition` | `transcribe(audio, sample_rate = 16000)` |
| `text-to-speech` | `synthesize(text)` → samples and their sample rate |
| `embedding` | `embed(texts)` → one vector per text |
| any | `run(inputs)`, `run_typed(inputs)` with named tensors |

`chat` applies the model's chat template. The generation methods take sampling
options (`max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop_ids`). A
method called on the wrong kind of model fails with the name of the method that
fits. The task methods travel over `run` as named tensors (`audio`, `text`,
`embeddings`), so a plugin built for ABI 1 needs nothing new. To write a
plugin, see [`rlx-model-plugin`](../crates/bindings/rlx-model-plugin/README.md),
which is the reference plugin and the guide.

## Not bound

Backward-only kernels (reach them through `grad` / `jvp`), lowering forms the
compiler produces, raw graph surgery, and the Gaussian-splat ops (an
out-of-tree seam: the IR keeps them, the bindings do not expose them). The
full list, with reasons, is `NOT_BOUND` in `rlx-bind/tests/ir_drift.rs`.

## License

MIT OR Apache-2.0.
