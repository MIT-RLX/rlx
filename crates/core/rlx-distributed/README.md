# rlx-distributed

Multi-node **distributed inference** (pipeline / tensor parallel) for RLX models. It layers on `rlx-driver`'s transports (`TcpTransport`, `ThunderboltTransport`, and `rlx-mlx`'s `MlxTransport`) and `ProcessGroup` to run one model split across several hosts, coordinating per-rank layer blocks with a model-agnostic relay.

## Placing a model on the cluster

Machines can be **listed, discovered, or both** — hand-written `[[node]]` entries
are kept verbatim and keep their place in the pipeline; discovered hosts are
appended and deduped by `addr`. See `cluster.example.toml`.

```rust,no_run
# use rlx_distributed::cluster::*;
# fn tensor_index() -> Vec<TensorEntry> { unimplemented!() }
let mut cx = Cluster::from_path("cluster.toml")?;
// discover -> probe -> derive cost from the checkpoint -> plan
cx.autoplan(&tensor_index(), &CostOptions::default().with_experts(8, 288), worker_bin)?;
print!("{}", cx.plan_table());
# Ok::<(), anyhow::Error>(())
```

```text
node                 device    layers      resident      disk   stage
10.0.0.5:9100        cuda        0..14     8.9/34.0G    28.0G   0.042s paged
node-b:9100          metal      14..30     9.9/58.0G    32.0G   0.031s paged
node-c:9100          cpu        30..45    10.1/60.0G    30.0G   0.088s paged
critical path: node-c:9100 at 0.088s/token
```

`policy = "auto"` decides capacity as the better of *experts resident in
RAM/VRAM* and *dense weights resident, expert banks on disk*, then shares layers
out by predicted per-layer time — compute plus the disk time to stream in the
experts each token touches. For a fine-grained MoE that is the difference
between a plan that fits and one that says the model is too big: GLM-5.3-Flash
needs ~10 GB resident per node, not the 93 GB a RAM-only view reports.

### Objectives, KV and precision

`[placement.objective]` says what to make small, separately from how to split:
`latency` uses the fewest nodes that hold the model (each stage boundary is a hop
on the critical path), `throughput` evens out stage times, `memory` spreads thin.
`context` reserves KV-cache bytes **before** weights, so a plan that fits at
`seq = 8` cannot OOM at 128 k.

The cache width is **inferred from the checkpoint's tensor shapes** — latent
(MLA) layers cache only the compressed latent, ordinary attention caches K and
V, and linear-attention layers have a fixed state that does not grow with
context. On a hybrid stack like GLM-5.3-Flash (34 recurrent + 11 latent) that is
the difference between pricing the cache at 3 GB and at 12 GB. The inference is
anchored on `hidden_size`, so it reads GGUF and HF dim orders alike.

It is kept **per block**, not just averaged, because on a hybrid stack the
average is not what any node pays: a stage holding GLM-5.3-Flash layers 0..3
caches nothing, while one holding 3..7 caches a full layer's worth. Each
assignment reports its exact `est_kv_bytes`, and capacity — decided before the
range is — budgets for the worst contiguous window of that length, so the plan
fits wherever the range lands. `CostOptions::kv_dtype` scales the whole thing
for an f16 or int8 cache.

When it *cannot* be read, planning stops rather than reserving nothing — a silent
zero looks exactly like "no cache" and fails much later, far from its cause. The
error names the three ways out: declare `kv_bytes_per_layer_token`, set
`ModelCost::kv` from the model crate, or `on_unknown_kv = "assume"` for a
pessimistic MHA-width bound. `assume` needs the model's real width and errors
without one: estimating width from byte counts is not conservative, because the
"≥1 byte per parameter" premise inverts under quantization and the bound comes
out at *half* the real cache. `precision_ladder = ["f16", "mxfp4"]` steps down
until the model fits and reports which width it settled on — including any
`fNeXmY` minifloat, whose width is parsed from the name rather than looked up in
a table.

`device` and `precision` are validated when the config loads, so `device =
"cuda+cpu"` or a mistyped precision fails naming the file rather than surfacing
as a puzzling compile flag on the third node. The TOML is unchanged: both are
still written as plain strings.

### Standby nodes and retirement

A node is `active`, `standby` or `draining`. A standby is planned with the *same
layer range* as the node it backs, so it holds the same weights and stays warm:

```rust,no_run
# use rlx_distributed::cluster::*;
# fn cost() -> ModelCost { unimplemented!() }
# let mut cx: Cluster = unimplemented!();
cx.retire("10.0.0.5:9100", cost())?;   // promote its standby, or re-plan
# Ok::<(), anyhow::Error>(())
```

Promotion is a pointer swap rather than a cold load — for a 90 GB stage that is
seconds instead of minutes. With no standby, `retire` still works by re-planning
across the remaining nodes; it just reloads weights, which is why a cluster you
intend to drain should carry spares.

`ModelCost::from_tensor_index` derives the per-layer numbers from a checkpoint's
tensor listing (a GGUF header is enough — no weights are read), keeping routed
expert bytes separate from resident ones. With the `gguf` feature,
`ModelCost::from_gguf` takes the shard paths directly.

## Modules

- `config` — `hosts.json` parsing + process-group construction ([`DistConfig::connect`]).
- `partition` — pipeline-parallel layer assignment ([`pipeline_layer_range`], [`block_role`]).
- `pipeline` — the model-agnostic relay ([`PipelineCoordinator`]) driving per-rank [`BlockRunner`]s.
- `launch` — local multi-process cluster helpers ([`LocalCluster`], [`worker_args`]).

A model family plugs in by implementing [`BlockRunner`] for its layer block.

## Public API

```rust
use rlx_distributed::{DistConfig, ParallelMode, PipelineCoordinator, pipeline_layer_range};

let cfg = DistConfig::from_hostfile("hosts.json", /*rank*/ 0)?;
let group = cfg.connect()?;                     // ProcessGroup over the chosen transport
let (start, end) = pipeline_layer_range(rank, world, num_layers);
// build a BlockRunner for [start,end) and hand it to PipelineCoordinator
# anyhow::Ok(())
```

## Quick start

```bash
cargo run -p rlx-distributed --example transport_bench
```

## How it fits

Built on `rlx-driver` (`ProcessGroup`, transports). Model crates such as [rlx-qwen3](https://github.com/MIT-RLX/rlx-models) provide the per-rank block runners.
## License

MIT OR Apache-2.0.
