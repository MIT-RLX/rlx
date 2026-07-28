# rlx-qnn FFI runtime backend (`Device::Hexagon`)

Status: **dynamic C-API path implemented + validated across a broad op surface**
(see *Milestones → M-breadth*). A graph dispatched through
`Session::new(Device::Hexagon)` executes on the real QNN `libQnnCpu.so` with
bit-exact parity vs the oracle, validated in Docker (`rust:1-bookworm` + the
Community SDK, x86_64 under emulation) — 17/17 FFI ops green, covering a complete
decoder-LLM forward pass plus embedding and vision models. The context-binary
perf path (M3) and HTP/on-device (M4) remain future work.

This documents the in-process FFI runtime backend that `../rlx-models` consumes,
and how it sits next to the codegen path (`qnn_model.cpp` + `qnn-net-run`, also
real-validated on `libQnnCpu.so` — see `../README.md` and `docker/`).

## What shipped (M1/M2)

- `runtime/rlx_qnn_shim.{c,h}` — C shim: `dlopen` the backend lib → `QnnInterface`
  → backend/context/graph create → tensors + MatMul node → finalize → execute.
  Compiled against the real SDK headers (no hand-transcribed vtable).
- `src/runtime.rs` (feature `runtime`) — `matmul_f32` / `QnnExecutable` Rust FFI
  binding + `build.rs` (cc against `$QNN_SDK_ROOT/include/QNN`).
- `rlx-runtime`: `Device::Hexagon` (rlx-driver), the `qnn_backend` adapter
  (`backend.rs`), `qnn` feature, registry + `device_parse` wiring, and
  `tests/qnn_hexagon_matmul.rs` (Session-level parity, skips without the SDK).

## Why FFI is required (the codegen path can't serve rlx-models)

rlx-models runs inference through the standard rlx `Session` / `Device` API —
the same dispatch every other backend uses (Metal, CUDA, ANE). The codegen path
is a *side tool*: it emits C++ + shell scripts and shells out to the SDK; there
is no `Device` variant, so a model crate cannot dispatch to it.

To run a model on Snapdragon *through rlx the way every other backend works*,
QNN must be a real **FFI `Device` backend** — `dlopen` the vendor runtime and
execute in-process, exactly like `rlx-cuda` / `rlx-coreml`.

This is also why QNN differs from `rlx-cerebras` / `rlx-fpga` (deliberately
codegen-only, no `Device`): those have no live runtime to link — a wafer
simulator / bitstream synthesis step. QNN ships `dlopen`-able
`libQnn{Htp,Cpu,Gpu}.so`, a real runtime, so it belongs as a `Device` peer.

The codegen work is **not** wasted: it is the graph-construction reference + the
numerical oracle, and it feeds the context-binary perf path (below).

## Where the code lands (mirrors `rlx-coreml` / `Device::Ane` exactly)

Concrete touch points (verified against the tree at time of writing):

| Change | File | Note |
| --- | --- | --- |
| Add `Hexagon` variant | `crates/rlx-driver/src/device.rs:23` (`enum Device`) | New `// ── Qualcomm ──` section. Name TBD (see open questions). |
| FFI runtime (the real work) | `crates/rlx-qnn/src/runtime/` (new) | `QnnExecutable`: `dlopen` libQnn, build/load graph, `set_param`/`finalize`/`run`. |
| Runtime adapter | `crates/rlx-runtime/src/backend.rs` (new `pub mod qnn_backend`, mirror `coreml_backend` at :2366) | `QnnBackend: Backend` + `QnnExecutableWrapper: ExecutableGraph` delegating to `rlx_qnn::QnnExecutable`. |
| Feature + optional dep | `crates/rlx-runtime/Cargo.toml` (`[features]`, deps) | `qnn = ["dep:rlx-qnn"]`; `rlx-qnn = { path = "../rlx-qnn", optional = true }`. Stays out of any default (empty-prelude rule). |
| Registration | `crates/rlx-runtime/src/registry.rs:65` (`register_builtin`) | `#[cfg(feature = "qnn")] map.insert(Device::Hexagon, …)`. |
| String parse | `crates/rlx-runtime/src/device_parse.rs:42` | `"hexagon" \| "htp" \| "qnn" => Device::Hexagon` and the reverse arm. |
| Capability/legalize | `crates/rlx-runtime/src/device_ext.rs` + a `QNN_SUPPORTED_OPS` slice in `backend.rs` | Like `COREML_SUPPORTED_OPS`; `LegalizeForBackend` refuses unsupported ops instead of mis-dispatching. |

The `Backend` trait the adapter implements (`backend.rs:422`): `compile(graph,
opts) -> Box<dyn ExecutableGraph>`, `compile_lir`, `supported_ops`. The
`ExecutableGraph` it returns (`backend.rs:148`): `set_param(name, &[f32])`,
`finalize_params()`, `run(&[(&str, &[f32])]) -> Vec<Vec<f32>>`, optional
`clone_box`. This is the entire surface `QnnExecutable` must satisfy — same
shape `CoremlExecutable` already satisfies.

## Two integration styles (and which to lead with)

QNN exposes two ways to get a runnable graph; both go through the same
`dlopen` + `QnnInterface` vtable:

1. **Dynamic C-API build (online).** `QnnContext_create` → `QnnGraph_create` →
   `QnnTensor_*` / `QnnGraph_addNode` (built from rlx-ir) → `QnnGraph_finalize`
   → `QnnGraph_execute`. Flexible for any graph, simplest to wire to the
   existing `model::Model::from_graph` recognizer. Downside: on HTP,
   `finalize` (graph prepare) is expensive at runtime → poor cold start.

2. **Context binary (offline-prepare, online-load).** Build a serialized,
   HTP-optimized context `.bin` once (`qnn-context-binary-generator`, fed by the
   model lib the codegen path already produces), then at runtime
   `QnnContext_createFromBinary` → `QnnGraph_execute`. Peak perf + fast cold
   start — the deployment shape rlx-models actually wants, and the one aligned
   with the per-backend-peak-perf north star.

**Recommendation:** lead with (1) to stand up and *validate* the FFI plumbing on
`libQnnCpu.so` in Docker now, then add (2) for HTP perf. The codegen path is
exactly what produces the (2) artifact, so the two halves compose rather than
compete.

## QNN FFI lifecycle (what `QnnExecutable` does)

```text
dlopen(libQnn<Backend>.so)                         # Cpu / Htp / Gpu, by Device + env
QnnInterface_getProviders(...)                     # grab the versioned vtable
  → backendCreate → deviceCreate → contextCreate
  → graphCreate
  → for each rlx-ir op: tensorCreateGraphTensor + graphAddNode   (style 1)
    OR contextCreateFromBinary(<rlx-qnn .bin>)                    (style 2)
  → graphFinalize
run(inputs):  wrap input &[f32] as Qnn_Tensor_t (APP_WRITE) →
              graphExecute → copy APP_READ outputs back to Vec<Vec<f32>>
drop:         graphFree / contextFree / deviceFree / backendFree / dlclose
```

`set_param` stages static weights (style 1: `QNN_TENSOR_TYPE_STATIC` with a
client buffer; style 2: already baked into the `.bin`). `finalize_params` maps
to `graphFinalize`. Reuse `reference::matmul_f32` as the parity oracle, exactly
as the codegen path does.

## Build & gating posture

- **Driverless build.** `dlopen` the backend lib at runtime (don't link it) so
  `cargo build -p rlx-qnn --features runtime` compiles on any host with no SDK
  present — the same approach `rlx-vulkan` uses for `libvulkan`. Path/dlopen
  name resolves from `QNN_SDK_ROOT` / env, else a clear runtime error.
- **Optional everywhere.** `qnn` feature is off by default in rlx-runtime; the
  `rlx-qnn` runtime module sits behind a `runtime` feature in this crate so the
  codegen tool stays dependency-free. Honors the empty-prelude / optional-
  backend rules — no workspace default flips.
- **`rlx-qnn-sys`?** Decide between hand-rolled `extern "C"` decls for the
  `QnnInterface` vtable vs a vendored `bindgen` sys-crate (like `rlx-mlx-sys`).
  The vtable is large and versioned; bindgen reduces drift but adds a build dep.
  Lean hand-rolled for the matmul milestone, revisit if the op surface grows.

## Validation plan (same Docker harness, no new hardware)

1. **Style-1 FFI on CPU ref backend, in Docker.** A `--features runtime` test /
   example that `dlopen`s `libQnnCpu.so` (x86_64), builds+executes the matmul
   in-process, and checks vs `reference::matmul_f32` + numpy — the FFI analog of
   what `docker/validate.py run` already does for the codegen path. Add a
   `validate.py ffi` mode + a `just qnn-docker-ffi` recipe.
2. **Context binary** path validated the same way (load `.bin` → execute).
3. **HTP / on-device.** `libQnnHtp.so` needs real Snapdragon silicon — the only
   step the Docker loop cannot cover. Final milestone; gate behind a device CI
   runner.

## Milestones

- **M1 — interface bring-up. ✅ done.** `dlopen` + `QnnInterface_getProviders` +
  backend/context/graph create/free on `libQnnCpu.so`.
- **M2 — single matmul, style 1. ✅ done.** Graph build via the QNN C API + the
  `ExecutableGraph` wrapper + `Device::Hexagon` registration; Session-level
  parity in Docker. rlx-models can target `Device::Hexagon` for a matmul today.
- **M3 — context binary (perf). ✅ done (CPU-validated).** Persistent session
  (`rlx_qnn_session_*`: finalize once, execute many). Save via
  `contextGetBinary` / load via `createFromBinary` + `libQnnSystem` metadata +
  `graphRetrieve`. Round-trip bit-exact on `libQnnCpu.so`. HTP still needs
  silicon (M4).
- **M-breadth — op surface. ✅ done** (see earlier notes; 39 FFI tests).
- **M4 — HTP / on-device.** `libQnnHtp.so` on Snapdragon silicon (Docker can't
  cover). Same session + context-binary path.

## Risks / open questions

- **vtable layout fidelity.** The `QnnInterface` / `Qnn*_Config` struct layouts
  must match the SDK headers for the pinned version; a mismatch is UB, not a
  compile error. Pin a version (2.42.x validated here) and assert
  `coreVersion` / `apiVersion` at load.
- **`Send`/`Sync`.** The QNN context handle is owned exclusively; mark the
  wrapper `unsafe impl Send` like `CoremlExecutableWrapper` and never share a
  context across threads (clone via fresh context for parallel dispatch).
- **Published-rlx coupling.** `Device::Hexagon` lands in `rlx-driver` +
  `rlx-runtime` and must be **published** before rlx-models can pin it
  (rlx-models tracks published rlx). This is a core-repo change, not a
  models-repo one.
- **Naming.** `Device::Hexagon` (the NPU) vs `Device::Qnn` (the runtime, which
  also drives CPU/GPU). Leaning `Hexagon` to match the `Ane`/`Cuda`
  hardware-named variants, with the backend lib (`Cpu`/`Htp`/`Gpu`) selected via
  env — but `Qnn` is defensible if we want one variant spanning all QNN targets.
```
## License

MIT OR Apache-2.0.
