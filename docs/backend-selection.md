# Backend selection and runtime switching

RLX separates three concerns:

1. **What you ship** — Cargo features (`cpu`, `metal`, `cuda`, …) baked into the binary.
2. **What you allow** — `DevicePolicy` (allow / deny / prefer lists).
3. **What you pick per request** — `GraphDevices` or `DeviceRouter`.

The single-backend path (`Session::new(device).compile(graph)`) is unchanged. The multi-backend
helpers below sit on top of it.

## Quick pick (Rust)

```rust
use rlx::prelude::*;

// Fastest backend for this graph (cost model + calibration caches)
let device = fastest_device_for(&g);

// Or restrict what the process may use
let policy = DevicePolicy::only([Device::Cpu, Device::Metal, Device::Mlx]);
let device = resolve_device(&g, None, &policy)?;
```

## `DevicePolicy`

Controls which backends are considered. Intersected with compile-time features and runtime
driver probes (`is_available`).

| Method | Purpose |
|--------|---------|
| `DevicePolicy::all()` | Default — every compiled-in backend |
| `DevicePolicy::only([…])` | Allow-list |
| `.with_deny([…])` | Block-list |
| `.with_prefer([…])` | Tie-break order when cost models tie |
| `.with_benchmark_pick(n)` | One-time micro-benchmark (needs inputs at resolve time) |
| `DevicePolicy::from_env()` | Read `RLX_*` env vars (see below) |

Pick strategy (`DevicePickStrategy`):

- **`CostModel`** (default) — rank via calibrated throughput + platform priority.
- **`Benchmark { runs }`** — run `benchmark_devices` once, cache winner on this `GraphDevices` instance.

## `GraphDevices`

Owns one `Graph` and a lazy per-device compile cache. Params uploaded via `set_param` /
`set_param_typed` are mirrored to every cached backend.

```rust
let mut runner = GraphDevices::with_policy(g, policy);
runner.set_param("w", &weights);

// Explicit backend
let out = runner.run(Device::Metal, &inputs)?;

// Hint → RLX_DEVICE → cost model / benchmark
let out = runner.run_resolved_with_inputs(None, &inputs)?;

// Fallback chain (RLX_DEVICE_CHAIN or explicit list)
let (device, out) = runner.run_chain(None, &inputs)?;

// Try backends in order until one succeeds
let (device, out) = runner.run_try(&[Device::Cuda, Device::Gpu, Device::Cpu], &inputs)?;

// Warm / benchmark
runner.warm_all()?;
let report = runner.benchmark(&inputs, 20)?;
runner.sync_params_to_all();
```

Key methods:

| Method | Behavior |
|--------|----------|
| `devices()` | Backends that support this graph under the policy |
| `report()` | Per-backend blockers + recommended pick |
| `fastest()` | Cost-model winner (no compile) |
| `resolve(hint)` | Hint → env → cost model |
| `compile(device)` | Lazy compile; returns `&mut CompiledGraph` |
| `invalidate_cache()` | Drop all cached executables |

## `FlexibleSession`

Like `Session`, but backend is chosen at compile time via policy + optional hint:

```rust
let session = FlexibleSession::from_env();
let compiled = session.compile_resolved(g, None)?; // picks via policy
let compiled = session.compile_on(g, Device::Cpu)?; // explicit
```

## `DeviceRouter`

Serving-oriented wrapper: **`warm_all()` on construction**, optional throttle-aware re-warm
(via `HwSnapshot::is_throttled()`), then `run` / `run_chain` / `run_on`.

```rust
let mut router = DeviceRouter::from_env(g)?;
router.set_rebench_on_throttle(false);
let (device, out) = router.run(&inputs, None)?;
```

Use `DeviceRouter` when you want all viable backends compiled up front (lower tail latency on
first request per device). Use `GraphDevices` when you want lazy compile-on-first-use.

## Introspection

| API | Returns |
|-----|---------|
| `is_available(device)` | Driver + feature probe |
| `available_devices()` | All compiled-in backends that pass `is_available` |
| `devices_for(&graph)` | Graph-compatible backends |
| `device_report(&graph, &policy)` | Blockers + recommended device |
| `device_label(device)` | Stable string (`"metal"`, `"gpu"`, …) |
| `parse_device(&str)` | String → `Device` (aliases: `nvidia`→cuda, `wgpu`→gpu) |
| `BackendsManifest::current()` | Compile-time feature manifest |
| `register_backends! { … }` | Register companion crates at startup (prelude macro) |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `RLX_DEVICES` | Allow-list (`cpu,metal,cuda`) |
| `RLX_DENY_DEVICES` | Block-list |
| `RLX_PREFER_DEVICES` | Tie-break order |
| `RLX_DEVICE` | Default hint for resolved runs |
| `RLX_DEVICE_CHAIN` | Fallback order (`cuda,gpu,cpu`) |
| `RLX_BENCHMARK_PICK` | If set to `N`, enable benchmark pick with `N` runs |

Prefix variants: `DevicePolicy::from_env_key("MYAPP")` reads `MYAPP_DEVICES`, etc.

## Python (`pyrlx`)

```python
import json
import pyrlx as rlx
import numpy as np

# Introspection
print(json.loads(rlx.backends_manifest()))
print(rlx.parse_device("metal"))
print(rlx.fastest_device_for(g))
print(rlx.device_report(g)[0].label)

# Multi-backend runner
policy = rlx.DevicePolicy.only(["cpu", "metal"]).with_benchmark_pick(20)
runner = rlx.GraphDevices(g, policy=policy)
runner.set_param("w", weights)
runner.set_param_typed("w", weights.tobytes(), "f32")
out = runner.run("cpu", {"x": x})
out = runner.run_resolved({"x": x}, device=None)
device, outs = runner.run_chain({"x": x})

# Deferred compile
session = rlx.FlexibleSession(rlx.DevicePolicy.from_env())
compiled = session.compile_resolved(g, device="cpu")

# Serving wrapper (warm-all on init)
router = rlx.DeviceRouter(g, policy=rlx.DevicePolicy.only(["cpu"]))
device, outs = router.run({"x": x})
device, outs = router.run_chain({"x": x})
outs = router.run_on("cpu", {"x": x})
router.with_rebench_on_throttle(False)
```

Python classes and functions: `DevicePolicy`, `GraphDevices`, `FlexibleSession`, `DeviceRouter`,
`DeviceCandidate`, `DeviceBenchResult`, `fastest_device_for`, `device_report`, `parse_device`,
`backends_manifest`.

Build with the backends you need:

```sh
maturin develop --features cpu,metal,mlx,gpu
```

See also [`crates/pyrlx/docs/backends.md`](../crates/pyrlx/docs/backends.md).

## Custom backends

Builtins register on first `Session` use. Register companion crates before compiling:

```rust
rlx::register_backends! {
    splat => rlx::splat::register,
}
```

## Calibration caches

Metal, MLX, CUDA, ROCm, and wgpu run a one-time matmul micro-benchmark and write JSON under
`~/.cache/rlx/`:

| File pattern | Backend |
|--------------|---------|
| `metal-calib-*.json` | Apple Metal |
| `mlx-calib-*.json` | Apple MLX |
| `cuda-calib-*.json` | NVIDIA CUDA |
| `rocm-calib-*.json` | AMD ROCm |
| `wgpu-calib-*.json` | wgpu (`Device::Gpu`) |

Fields: `sgemm_gflops`, `roundtrip_overhead_ns`, `memory_bw_gbps`. These feed
`fastest_device_for` / `CudaCostModel` / `RocmCostModel` / `WgpuCostModel`. Delete a cache
file to force re-measurement.

Implementation: `rlx-cuda/src/calibrate.rs`, `rlx-rocm/src/calibrate.rs`, `rlx-wgpu/src/calibrate.rs`.

## Tests

```sh
cargo test -p rlx-runtime --test graph_devices_parity
cargo test -p rlx-runtime device_router --lib
cd crates/pyrlx && pytest tests/test_graph_devices.py -q
```

## When to use which API

| Use case | API |
|----------|-----|
| Fixed backend, hot loop | `Session` + `CompiledGraph` |
| Pick backend once per graph | `fastest_device_for` / `FlexibleSession` |
| Switch backend per request | `GraphDevices` |
| Serving with warm-all + fallback | `DeviceRouter` |
| Ops / deploy introspection | `BackendsManifest`, `device_report` |
## License

GPL-3.0-only.
