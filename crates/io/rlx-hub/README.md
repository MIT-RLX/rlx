# rlx-hub

Shard-aware HuggingFace download and verification for RLX. Fetches only the
part of a checkpoint a node actually needs, and proves what arrived is intact.
Model-agnostic: which repo and which layers go where is the caller's business.

## What's here

- **`index`** — parse `model.safetensors.index.json` and map a pipeline
  stage's contiguous layer range onto the shard **files** that hold its
  tensors (`plan_layer_stages`). Boundary shards are assigned to adjacent
  stages so every node ends up with *complete* layers. This is what lets three
  machines download a third of a model each instead of the whole thing.
- **`download`** — resumable `download_file` / `download_files` (via
  `curl -C -`), plus a three-tier integrity check: exact byte size from the HF
  API, a structural `.safetensors` header/data-length check that catches
  truncated or interrupted transfers, and — where the API exposes it
  (`fetch_sha256s`) — full content SHA-256.
- **`error`** — a `HubError` taxonomy, so callers match on structured variants
  (size / sha256 / structural mismatch, missing `curl`, …) rather than parsing
  strings.

## Requirements

`curl` on `PATH`. The resumable transfer and range requests are delegated to
it rather than reimplemented, which keeps this crate free of a TLS stack and
of an async runtime.

## Install

```toml
[dependencies]
rlx-hub = "0.2"
```

## Quickstart

```rust,no_run
use rlx_hub::{HfRepo, download_files, fetch_index, fetch_sha256s, fetch_sizes, plan_layer_stages};

let repo  = HfRepo::new("mlx-community/DeepSeek-V4-Flash-2bit-DQ");
let index = fetch_index(&repo)?;
let sizes = fetch_sizes(&repo)?;
let shas  = fetch_sha256s(&repo).unwrap_or_default();

// this node owns layers 18..35, and no embedding / head tensors
let stage = &plan_layer_stages(&index, &[18..35], &[vec![]])[0];

let report = download_files(
    &repo, &stage.shards, "/models/ckpt".as_ref(),
    &sizes, Some(&shas), |file, status| println!("{status}: {file}"),
);
assert!(report.failed.is_empty());
# Ok::<(), rlx_hub::HubError>(())
```

## See also

- [`docs/distributed.md`](../../../docs/distributed.md) — how layer stages are
  assigned across a multi-node run.

## License

MIT OR Apache-2.0.
