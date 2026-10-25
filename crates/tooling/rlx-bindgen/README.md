# rlx-bindgen

Generates rlx's foreign-language bindings from [`rlx-bind`](../../bindings/rlx-bind)'s
two tables — the graph-builder op table (`table.rs`) and the runtime API table
(`api.idl`). Not published; it writes files into the workspace.

| output | from |
|--------|------|
| `rlx-capi/src/graph_ops.rs` — the `rlx_graph_<op>` C functions | the op table |
| `rlx-capi/include/rlx.h` — the C header | parsing rlx-capi's sources, so it matches the library |
| `rlx-swift/Sources/RLX/GraphOps.swift` — `Graph` methods | the op table |
| `rlx-swift/Sources/RLX/API.swift` — every runtime class and function | the runtime API table, checked against the C ABI |
| `pyrlx/python/pyrlx/_pyrlx.pyi` — the `Graph` stub section | the op table |
| `docs/bindings-api.md` — every op and runtime row in every language | both tables |
| the FFI-language bindings (Go, Java, C#, Ruby, Julia, Zig, C++, Node, Kotlin, Dart, Lua, MATLAB/Octave, R, PHP, Nim, D, V, Haskell, OCaml, Elixir, Odin) | `src/ir.rs`: both tables lowered into one typed view of the C ABI, each function checked against its exported signature, then one printer per language |

rlx-js and pyrlx don't need generated glue: they expand the op table at compile
time.

```sh
just gen-bindings      # cargo run -p rlx-bindgen — write every output
just check-bindings    # cargo run -p rlx-bindgen -- --check — fail if one is stale
```

`--check` (and `tests/up_to_date.rs`) fails when a checked-in output is stale,
so an op added to the table can't ship without its C, Swift and stub
counterparts. It also fails when an `api.idl` row and the C function rlx-capi
exports disagree. See [`docs/bindings.md`](../../../docs/bindings.md) and
[`crates/bindings/README.md`](../../bindings/README.md).

## License

MIT OR Apache-2.0.
