# rlx-macros

Procedural macros for RLX. Compile-time-only crate; nothing here ships
in the runtime.

## What's here

- **`#[rlx_model]`** — expands a struct into a model registration block
  (AOT-compiled-graph integration; the AOT path is incremental — most
  consumers will use the JIT `Session` flow in `rlx-runtime` first).
- **`pipeline_schedule!`** — compile-time pipeline scheduler grammar
  (plan #11).

## Install

```toml
[dependencies]
rlx-macros = "0.1"
```

Usually pulled in transitively via [`rlx-runtime`](https://crates.io/crates/rlx-runtime)
or [`rlx`](https://crates.io/crates/rlx).

## Build / test

```sh
cargo build -p rlx-macros
cargo test  -p rlx-macros
```

## Status

Surface intentionally narrow at 0.1.0. `#[rlx_op]` and `#[rlx_arch]`
proc macros for downstream extensibility (plan #25 / #82) are not yet
implemented.

## Gotchas

- Proc macros run at compile time of the *consumer* crate, so error
  messages can be confusing. When adding diagnostics, use `syn::Error`
  with a `Spanned` location, not `panic!`.
- Don't pull large deps into this crate; everything here is a build-time
  dep of every downstream user.

## License

GPL-3.0-only.
