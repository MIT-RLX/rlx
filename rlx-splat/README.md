# rlx-splat

3D Gaussian splatting for RLX — scene types, CPU reference pipeline, strict IR ops, and backend hooks.

## Architecture

| Module | Role |
|--------|------|
| `rlx_splat::core` | Scene, camera, SH math |
| `rlx_splat::reference` | CPU project → bin → sort → raster |
| `rlx_splat::prep_layout` | Packed prepare buffer (`prep_packed_len` aliases `rlx_ir`) |
| `rlx_splat::pipeline` | Graph helpers (common IR + decomposed) |
| `rlx_splat::shaders` | RLX-owned MSL source (`SPLAT_RASTER_MSL`) |
| `rlx_splat::backends::metal` | Native tile raster dispatch (feature `metal`) |
| `rlx_ir::Op::GaussianSplat*` | First-class ops (render / prepare / rasterize / backward) |

Call [`rlx_splat::register()`] once so `rlx-cpu` can execute splat ops.

## Quick start (monolithic op)

```rust
use rlx_ir::infer::GraphExt;
use rlx_runtime::{Device, Session};
use rlx_splat::{core::make_parity_scene, graph::gaussian_splat_render_scene, register};

register();
// build graph with gaussian_splat_render_scene, Session::compile, run
```

## Decomposed strict IR (prepare → rasterize)

```rust
use rlx_ir::infer::GraphExt;
use rlx_ir::ops::splat::{GaussianSplatInputs, GaussianSplatRenderParams};
use rlx_runtime::{Device, Session};
use rlx_splat::{pipeline::gaussian_splat_render_decomposed, register};

register();
let mut g = rlx_ir::Graph::new("splat");
// … inputs + meta …
let rgba = gaussian_splat_render_decomposed(&mut g, inputs, params);
g.set_outputs(vec![rgba]);
let out = Session::new(Device::Cpu).compile(g).run(&bindings);
```

Example: `cargo run -p rlx-splat --example decomposed_session --features test-support`

Packed prepare size is shared with the IR:

```rust
use rlx_ir::ops::splat::gaussian_splat_prep_packed_len;
```

## Autodiff / training

- **Forward (decomposed):** `GaussianSplatPrepare` + `GaussianSplatRasterize` for staging / GPU prepare buffers.
- **Training backward:** monolithic `Op::GaussianSplatRenderBackward` (explicit graph or autodiff VJP).
- **Autodiff on decomposed forward:** `rlx_autodiff::prepare_graph_for_ad` fuses prepare→rasterize into `GaussianSplatRender` before the VJP walk, so `grad()` on a decomposed graph uses the same backward as monolithic render.
- Helpers: `pipeline::gaussian_splat_backward_decomposed` (explicit packed grads), `graph::gaussian_splat_backward_scene`.

## Features

| Feature | Description |
|---------|-------------|
| `core` | Scene/camera types (default) |
| `reference` | CPU reference pipeline (default) |
| `execute` | Arena executors → `rlx-cpu` |
| `cpu` | `register()` + graph/pipeline |
| `io` | PLY/COLMAP (`load_colmap_training_bundle`, `resolve_colmap_init_hparams`, frame downscale) |
| `metal` | `backends::metal` raster dispatch (macOS) |
| `session` / `test-support` | `rlx-runtime` integration tests |

Avoid enabling `session` on `rlx-metal`’s dependency edge — use `reference` + `metal` only to prevent `rlx-runtime` cycles.

## Tests

```bash
cargo test -p rlx-splat --features test-support
cargo test -p rlx-splat --features test-support session_decomposed
```
