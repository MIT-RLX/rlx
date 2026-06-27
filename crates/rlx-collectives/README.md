# rlx-collectives

In-graph collective ops for tensor-parallel RLX execution. Adds a
`collective.all_reduce` custom op that sums a tensor across a process
group from **inside a compiled graph** — the primitive a tensor-parallel
layer needs right after its row-sharded `o_proj` / `down_proj`.

The op carries a `u64` **group id** in its `attrs`; each rank registers
its [`rlx_driver::ProcessGroup`] handle under an id, and the kernel
resolves it at execution time. An id-in-attrs (rather than a
thread-local) is deliberate: it stays correct under the backend's
threaded executor, and lets one process host several groups (e.g. a
tensor-parallel group and a pipeline-parallel group).

## Usage

```rust,ignore
use std::sync::Arc;
use rlx_ir::{Graph, Shape, DType};

// Once per process: install the op (IR shape extension + CPU kernel)
// and register this rank's process group under an id.
rlx_collectives::register();
rlx_collectives::register_group(rank, group); // group: Arc<ProcessGroup>

// In the graph: sum a row-sharded partial product across ranks.
let mut g = Graph::new("tp");
let x = g.input("x", Shape::new(&[2, 4], DType::F32));
let w = g.param("W", Shape::new(&[4, 8], DType::F32));
let y_partial = g.matmul(x, w, Shape::new(&[2, 8], DType::F32));
let y = rlx_collectives::all_reduce(&mut g, y_partial, rank); // shape unchanged
g.set_outputs(vec![y]);
```

## API

| Item                       | Role                                                        |
|----------------------------|-------------------------------------------------------------|
| `register()`               | Install the op (IR shape ext + CPU kernel). Idempotent.     |
| `register_group(id, grp)`  | Bind a rank's `ProcessGroup` to an id (typically `id = rank`). |
| `unregister_group(id)`     | Drop the group registered under `id`.                       |
| `all_reduce(g, x, id)`     | Insert a sum-across-`id` over `x`; result has `x`'s shape.   |
| `ALL_REDUCE`               | Registry name, `"collective.all_reduce"`.                   |

## Design notes

* **Sum reduce, `f32`.** The CPU kernel blocks until every rank reaches
  the collective, then delegates to `ProcessGroup::all_reduce(_, Sum)`
  (the transport — `LocalTransport` for in-process threads, or
  `NetTransport` for multi-node — lives in [`rlx-driver`]).
* **Shape-preserving.** All-reduce is elementwise across ranks, so the
  shape extension returns the input shape unchanged; the op fuses into
  graphs like any other.
* **GPU.** This crate ships the CPU kernel; device-resident all-reduce
  (no host round-trip) is registered by the backend — see
  `rlx_mlx::distributed` for the MLX `collective.all_reduce` kernel.

## Status

| Property              | Notes                                                        |
|-----------------------|--------------------------------------------------------------|
| Reduce ops            | Sum (the tensor-parallel case)                               |
| Dtype                 | `f32`                                                        |
| Validation            | Tensor-parallel matmul, Megatron SwiGLU MLP, and a Qwen3 decoder-layer shard, each vs. a hand-computed reference |
| Transport             | In-process threads + TCP `NetTransport` (multi-node hardware not exercised) |

## License

GPL-3.0-only.
