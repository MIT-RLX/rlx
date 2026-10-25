# Combining and splitting graphs

`rlx_ir::compose` combines several `Graph`s into one and splits one `Graph`
into several. It lives in `rlx-ir` (works under `no_std` too), so every crate
that holds a graph can use it — model builders, autodiff, the distributed
pipeline, export tools — and every binding exposes it ([below](#bindings)).

## Combining

`Graph::splice` is the one copy-in primitive. `parallel`, `sequential`,
`Partition::merge`, `rlx_compile::inline_into` and autodiff's
`merge_subgraph` are all calls to it.

| API | Result |
|-----|--------|
| `Graph::splice(&src, &Splice)` | Copies `src` into this graph. Returns `Spliced { outputs, map, vars }`. |
| `compose::parallel(name, &[&a, &b, …])` | Side by side: outputs are `a`'s, then `b`'s, … |
| `compose::sequential(name, &[&a, &b, …])` | Chained: `b`'s free inputs (declaration order) are fed by `a`'s outputs. Arity is checked. |

`Splice` options:

| Option | Effect |
|--------|--------|
| `.bind(name, node)` / `.bind_all(pairs)` | Feed `src`'s leaf `name` from `node`, a node of the target. The leaf is an input — or a **param**: bind one to tie it to a weight of another name, or to compute it. Names are `src`'s own, before any prefix. Binding a name `src` lacks, or the same name twice, is an error. |
| `.prefix("enc.")` | Namespace `src`'s unbound leaves and node names, so they don't share with same-named leaves of the target. |
| `.adapt_shapes()` | Accept a bound node whose shape differs from its leaf, and re-infer every copied node's shape from its new operands (autodiff splices per-step bodies this way). Without it a differing shape is `ShapeMismatch`. |

Leaves are bound by name, so the combined graph keeps **one leaf per name**:
an unbound input or param whose name the target already has is *shared* (the
same input fed once, a tied weight loaded once) after a kind and shape check.
Use `prefix` to keep a graph's leaves apart instead.

```rust
use rlx_ir::{Graph, Splice};

// Three layers with their own weights: l0.w, l1.w, l2.w …
let mut g = Graph::new("stack");
let mut h = None;
for l in 0..3 {
    let mut s = Splice::new().prefix(format!("l{l}."));
    if let Some(h) = h {
        s = s.bind("x", h);
    }
    h = Some(g.splice(&layer, &s)?.outputs[0]);
}
g.set_outputs(vec![h.unwrap()]);

// The same layer applied twice with tied weights:
let twice = rlx_ir::compose::sequential("twice", &[&layer, &layer])?;

// A decoder whose `w_out` is the encoder's `w`:
let w = g.param_id("w").unwrap();
g.splice(&decoder, &Splice::new().bind("h", h_enc).bind("w_out", w))?;
```

`splice` checks everything before adding a node, so on error the target is
unchanged.

## Splitting

Each of these returns a `Partition`: self-contained part graphs that run in
order, and `merge()` to put them back.

| API | Result |
|-----|--------|
| `Graph::partition(n, \|id\| part)` | `n` parts. Any assignment works as long as every edge runs to the same or a later part (checked: `BackwardEdge`). |
| `Graph::split_at(&[node, …])` | Two parts: the given nodes and everything they read, then the rest. The cut nodes are outputs of the first part. |
| `Graph::components()` | One part per independent subgraph; the parts import nothing, so they run in any order — on different devices at once. Sharing an input does not connect two subgraphs. |
| `Partition::merge()` | One graph again. |
| `Graph::extract(&[node, …])` | The single subgraph computing those nodes (their backward cone), with them as outputs. |

A value crossing parts becomes a named boundary, `boundary_name(id)` =
`__boundary.<node id>`: an output of its producer's part (`Part::exports`)
and an `Op::Input` of each consumer's part (`Part::imports`). Inputs, params
and constants are copied into every part that reads them, so each part is a
self-contained graph that compiles on its own — on another device or another
machine. Run the parts in order, feeding each part's imports with earlier
parts' exports; `Partition::outputs[k] == (part, j)` locates original output
`k`.

```rust
let p = g.partition(2, |id| if id.0 < cut { 0 } else { 1 })?;
let mut env = feeds.clone();
for part in &p.parts {
    let outs = run(&part.graph, &env);              // any device
    for (name, v) in part.exports.iter().zip(outs) {
        env.insert(name.clone(), v);
    }
}
```

`rlx_distributed::partition` remains the contiguous, node-count-balanced
pipeline splitter with its own stage format for the cluster transport; use
`Graph::partition` for arbitrary placement.

## Variables and symbols

Variables (`Graph::vars`, decode-step caches) follow their nodes:
`parallel` offsets their output indices, `sequential` threads every graph's
variables and appends their next values after the last graph's outputs, and
`extract` keeps a variable whose input and next value both survive. A part
carries a variable when it holds both ends; `Partition::vars` records every
variable by input name, so `merge` restores them wherever the ends landed.
Two graphs writing the same variable is an error (`SharedVar`). Symbolic op
attributes move with their nodes; symbol ids are not renumbered, so graphs
combined must agree on what each symbol means (conflicting `SymDef`s are an
error).

## Bindings

Rows in `rlx-bind/src/api.idl`, implemented once in
`rlx-bind/src/runtime/compose.rs`. Graphs are borrowed, never consumed; a
graph may be spliced into itself.

| | Python | JS | C |
|-|--------|----|---|
| Splice | `splice(target, source, bind=…, prefix=…, adapt_shapes=…)` → node ids | `rlx.splice(target, source, {bind, prefix, adaptShapes})` | `rlx_splice(target, source, options_json, &ids)` |
| Combine | `parallel(a, b, name=None)`, `sequential(a, b, name=None)` | `rlx.parallel`, `rlx.sequential` | `rlx_parallel`, `rlx_sequential` |
| Extract | `extract(graph, outputs)` | `rlx.extract` | `rlx_extract` |
| Split | `Partition(graph, assign)`, `Partition.split_at(graph, cut)`, `Partition.components(graph)` | `new rlx.Partition`, `Partition.splitAt`, `Partition.components` | `rlx_partition_new`, `_split_at`, `_components` |
| Parts | `len(p)` / `num_parts()`, `graph(i)`, `imports(i)`, `exports(i)`, `outputs()`, `merge()` | `numParts()`, `graph(i)`, … | `rlx_partition_num_parts`, `_graph`, … |

`parallel` / `sequential` take two graphs; chain calls for more (both are
associative). `Partition(graph, assign)` takes one part per node id
(`node_count()` entries; leaves' entries are ignored). Swift and the other
C-ABI languages are generated from the same rows (`just gen-bindings`).

```python
import pyrlx as rlx

g = rlx.Graph("stack")
[h0] = rlx.splice(g, layer, prefix="l0.")
[h1] = rlx.splice(g, layer, prefix="l1.", bind={"x": h0})
g.set_outputs([h1])

p = rlx.Partition.split_at(g, [h0])
env = {"l0.x": x}
for i in range(len(p)):
    feed = {n: env[n] for n in p.imports(i)}
    if i == 0:
        feed["l0.x"] = x
    outs = run(p.graph(i), feed)            # any device
    env.update(zip(p.exports(i), outs))
```

Tests: `rlx-ir` (`compose::tests`), `rlx-bind` (`runtime::compose`),
`rlx-runtime/tests/graph_compose.rs` (combined graphs, parts run in order,
merged partitions and components on CPU against the original graph),
`rlx-js/tests/compose.rs`, `pyrlx/tests/test_compose.py`, and
`rlx-capi/examples/compose.c` (run by `rlx-capi/tests/c_program.rs`).
## License

MIT OR Apache-2.0.
