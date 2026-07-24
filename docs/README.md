# RLX documentation

| Document | Contents |
|----------|----------|
| [extending.md](extending.md) | **Extending rlx from downstream** (no core edit): the four seams — `LayerStage` blocks (`ModelFlow::layer_stage` + `FlowCtx` primitive builders + side outputs), custom ops (`OpExtension` + `register_op` + `lower`/kernel/`custom_fn`), backends (`register_backend`), codegen targets (consume `rlx_ir::Graph`); the `rlx-extend` prelude + `just link-local` dev loop |
| [fk-fusion.md](fk-fusion.md) | FKL-style region fusion: prologue, batch regions, env/session toggles, kernel tuning |
| [backend-selection.md](backend-selection.md) | Multi-backend runtime: `DevicePolicy`, `GraphDevices`, `DeviceRouter`, env vars, Python API, calibration |
| [distributed.md](distributed.md) | **Distributed computing**: the layers (Transport→ProcessGroup→Node), collectives, node discovery (static/mDNS/rendezvous), the `dist_job` + `dist_node` examples, and the `rlx_runtime::dist` ship-graph worker |
| [iroh-transport.md](iroh-transport.md) | **NAT-traversing distributed** (`IrohTransport`, feature `iroh`): QUIC + n0 relays + pkarr/DNS discovery — reach peers by `EndpointId` (no `ip:port`/port-forwarding), the per-edge FIFO wire protocol, `connect_discovered` / `process_group_from_env`, the `TOPOLOGY=iroh` launcher topology + `RLX_DEVICE` / `RLX_DETERMINISTIC_REDUCE`, and hybrid GPU+CPU run recipes |
| [op-coverage.md](op-coverage.md) | **Single source of truth** for every IR op: descriptions, per-backend coverage matrix (CPU/Metal/MLX/WGPU/ANE/CUDA/ROCm/TPU — all **153/`OpKind`**; Vulkan/oneAPI also at 153 as EXTRA backends), and op variations (Activation/Binary/Quant schemes/…) |
| [gguf-backend-paths.md](gguf-backend-paths.md) | **GGUF / `DequantMatMul` execution paths** — shared scheme ids, per-backend GPU/host/ANE/TPU lowering, Metal fused IQ GEMV, pyrlx convert/load, env toggles, P0–P5 + backlog code map |
| [scaled-matmul-fp8.md](scaled-matmul-fp8.md) | **Native low-precision GEMM (`Op::ScaledMatMul`)** — FP8/FP6/FP4 + the parameterized `fNeXmY` minifloat family (all 28 formats), the `ScaledFormat` API/DX, specifying a format across compose-ops / Tensor DSL / `CompileOptions` / pyrlx, per-backend status (CPU/CUDA/ROCm/Metal/Vulkan) + hardware validation |
| [nan-debugging.md](nan-debugging.md) | **Localizing NaN/Inf** at the compiler/IR level — static lint (`RLX_LINT_NUMERICS`) for provable constant blow-ups + runtime localizer (`RLX_DEBUG_NANS`) that names the culprit op vs a propagator, with provenance and a fix hint; where it lives and how to extend it to a backend |
| [weight-compute-caching.md](weight-compute-caching.md) | **Computing weight-derived tensors once, not every forward** — compile-time fold (`param_bindings` / offline [`rlx-bake`](../crates/io/rlx-bake/) → `*.rlx`), and runtime hoist (`cache_param_invariant` / `RLX_CACHE_PARAM_INVARIANT`) |
| [rlx-bake.md](rlx-bake.md) | **`rlx-bake` walkthrough** — what bake is (vs model+weights), pipeline, format / encrypt, `.rlxp` export, full MNIST train→bake→encrypt→run steps and how to read the stats |
| [rlxp.md](rlxp.md) | **`.rlxp` package format** — flat mmap (default) + hybrid hot/warm/cold, optional ZIP/dir, optional executable MIR graph, sidecars, dist placement, GGUF/ONNX import |
| [rlx-env-vars.md](rlx-env-vars.md) | **Exhaustive `RLX_*` inventory** — every env / option identifier in the tree, grouped by backend/area, with curated-catalog and code-read marks (`just gen-rlx-env-vars`) |
| [development.md](development.md) | Dev workflow: `just` recipes, pyrlx, tests, dispatch probes, `Op::Scan` unroll / host contract |
| [fpga-export.md](fpga-export.md) | **FPGA / SystemVerilog export**: `ExportTarget`, `FpgaExportConfig`, target-agnostic RTL, `HwTarget` matrix |
| [benchmarks/higher-order-ad.md](benchmarks/higher-order-ad.md) | Higher-order autodiff benchmarks |
| [benchmarks/mlx-linux.md](benchmarks/mlx-linux.md) | MLX on Linux/WSL: compile, CPU vs CUDA, vs `rlx-cpu` matmul benches |
| [benchmarks/coreml-training.md](benchmarks/coreml-training.md) | CoreML on-device training: RLX vs native `MLUpdateTask`, compute-unit sweep, overhead- vs compute-bound regimes (why `cpu`≥`ane`, and why the result flips with model size) |
| [benchmarks/frameworks-and-backends.md](benchmarks/frameworks-and-backends.md) | MNIST-training comparison **matrix**: every framework × backend, verified/rig/candidate status, `torch.compile`/Keras/MPSGraph/ORT runners, and the CUDA runbook |

Release notes: [`CHANGELOG.md`](../CHANGELOG.md) (workspace **0.2.13**).

Related repo docs:

- [`llms.txt`](../llms.txt) — workspace map for agents
- [`PLAN.md`](../PLAN.md) — roadmap and landed items
- [`crates/bindings/pyrlx/README.md`](../crates/bindings/pyrlx/README.md) — Python install and quickstart
- [`crates/bindings/pyrlx/docs/dsl.md`](../crates/bindings/pyrlx/docs/dsl.md) — `graph()` / `Node` DSL, scalar literals, operators
- [`crates/bindings/pyrlx/docs/quickstart.md`](../crates/bindings/pyrlx/docs/quickstart.md) — explicit vs DSL builders
- [`crates/bindings/pyrlx/docs/backends.md`](../crates/bindings/pyrlx/docs/backends.md) — maturin feature matrix
- [`crates/bindings/pyrlx/examples/dsl_quickstart.py`](../crates/bindings/pyrlx/examples/dsl_quickstart.py) — runnable DSL demo
## License

GPL-3.0-only.
