# rlx-bind

The language-neutral core every rlx binding is generated from.

* **`for_each_graph_op!`** — the op table: one row per graph builder
  (`src/table.rs`). rlx-js and pyrlx expand it into their native methods at
  compile time; `rlx-bindgen` prints the C ABI, `rlx.h`, Swift and `.pyi` from
  it.
* **`ops::*`** — a typed function per row (`ops::conv2d(graph, input, weight,
  kernel_size, …)`), with node ids checked, enums parsed and panics caught.
* **`manifest()`** — the table as data, for generators and drift checks.
* **`runtime::*`** (feature `runtime`, plus `training`, `text`, `gguf-io`,
  `gguf-convert`, `weights`, `models`, `fpga`) — `Session`, `Compiled`,
  `FlexibleSession`, `Runner` / `Router`, `Trainer`, `Optimizer`, `Tokenizer`,
  `Sampler`, `Model` (packages), `check`, devices, autodiff, graph composition
  (`compose`), distributed nodes (`dist::Group`), GGUF, RLXP and the weight
  loaders, as plain Rust. Options and reports cross as JSON.
* **`api.idl`** (`api::Api`) — the runtime API table: one row per class,
  method and function, named in every language. `rlx-bindgen` checks each row
  against the C function `rlx-capi` exports and generates Swift from it; rlx-js
  and pyrlx test that every row resolves and that they export nothing else, so
  a missing method fails a contract test in the language that lacks it.
* **`enums`, `ApiError`** — one parser per string option, one error type.

Adding an op is one row in `src/table.rs` and `just gen-bindings`; adding a
runtime call is one row in `src/api.idl`, implemented in `runtime` and wired
in rlx-capi, rlx-js and pyrlx. The
`ir_drift` test fails when rlx-ir gains a builder the table neither reaches
nor lists in `NOT_BOUND`. See [`docs/bindings.md`](../../../docs/bindings.md).

License: MIT OR Apache-2.0.
