# rlx-macros

Procedural macros for RLX. Compile-time-only crate; nothing here ships
in the runtime.

## What's here

- **`#[rlx_model]`** — expands a struct into a model registration block
  (AOT-compiled-graph integration; the AOT path is incremental — most
  consumers will use the JIT `Session` flow in `rlx-runtime` first).
- **`pipeline_schedule!`** — compile-time pipeline scheduler grammar.
- **`register_lm_runner!`** — register a model family's LM runner (`family`,
  `description`, `arches`) through `inventory`, so
  `rlx_runtime::auto_runner_name` can route a weights file to it.
- **`rlx_runner_main!`** — the `fn main()` of a per-family runner binary
  (`rlx_runner_main!(rlx_qwen3::cli::run, "rlx-qwen3")`).
- **`__rlx_build` / `__rlx_expr`** — the parser and code generator behind
  `rlx_tensor::rlx!` and `rlx_expr!`. Call those, not these: the declarative
  wrappers bring the names the output references into scope.

## Install

```toml
[dependencies]
rlx-macros = "0.2"
```

Usually pulled in transitively via [`rlx-runtime`](https://crates.io/crates/rlx-runtime)
or [`rlx`](https://crates.io/crates/rlx).

## Build / test

```sh
cargo build -p rlx-macros
cargo test  -p rlx-macros
```

## Status

Surface intentionally narrow. Downstream op and architecture registration
goes through `rlx-ir`'s `OpExtension` registry and `register_lm_runner!`, not
dedicated proc macros.

## Gotchas

- Proc macros run at compile time of the *consumer* crate, so error
  messages can be confusing. When adding diagnostics, use `syn::Error`
  with a `Spanned` location, not `panic!`.
- Don't pull large deps into this crate; everything here is a build-time
  dep of every downstream user.

## License

MIT OR Apache-2.0.
