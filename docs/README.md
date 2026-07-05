# RLX documentation

| Document | Contents |
|----------|----------|
| [fk-fusion.md](fk-fusion.md) | FKL-style region fusion: prologue, batch regions, env/session toggles, kernel tuning |
| [backend-selection.md](backend-selection.md) | Multi-backend runtime: `DevicePolicy`, `GraphDevices`, `DeviceRouter`, env vars, Python API, calibration |
| [distributed.md](distributed.md) | **Distributed computing**: the layers (Transport→ProcessGroup→Node), collectives, node discovery (static/mDNS/rendezvous), the `dist_job` + `dist_node` examples, and the `rlx_runtime::dist` ship-graph worker |
| [op-coverage.md](op-coverage.md) | **Single source of truth** for every IR op: descriptions, per-backend (CPU/Metal/MLX/WGPU/ANE/CUDA/ROCm/TPU) coverage matrix, and op variations (Activation/Binary/Quant schemes/…) |
| [gguf-backend-paths.md](gguf-backend-paths.md) | **GGUF / `DequantMatMul` execution paths** — shared scheme ids, per-backend GPU/host/ANE/TPU lowering, Metal fused IQ GEMV, pyrlx convert/load, env toggles, P0–P5 + backlog code map |
| [scaled-matmul-fp8.md](scaled-matmul-fp8.md) | **Native low-precision GEMM (`Op::ScaledMatMul`)** — FP8/FP6/FP4 + the parameterized `fNeXmY` minifloat family (all 28 formats), the `ScaledFormat` API/DX, specifying a format across compose-ops / Tensor DSL / `CompileOptions` / pyrlx, per-backend status (CPU/CUDA/ROCm/Metal/Vulkan) + hardware validation |
| [development.md](development.md) | Dev workflow: `just` recipes, pyrlx, tests, dispatch probes |
| [benchmarks/higher-order-ad.md](benchmarks/higher-order-ad.md) | Higher-order autodiff benchmarks |
| [benchmarks/mlx-linux.md](benchmarks/mlx-linux.md) | MLX on Linux/WSL: compile, CPU vs CUDA, vs `rlx-cpu` matmul benches |
| [benchmarks/coreml-training.md](benchmarks/coreml-training.md) | CoreML on-device training: RLX vs native `MLUpdateTask`, compute-unit sweep, overhead- vs compute-bound regimes (why `cpu`≥`ane`, and why the result flips with model size) |
| [benchmarks/frameworks-and-backends.md](benchmarks/frameworks-and-backends.md) | MNIST-training comparison **matrix**: every framework × backend, verified/rig/candidate status, `torch.compile`/Keras/MPSGraph/ORT runners, and the CUDA runbook |

Release notes: [`CHANGELOG.md`](../CHANGELOG.md) (workspace **0.2.10**).

Related repo docs:

- [`llms.txt`](../llms.txt) — workspace map for agents
- [`PLAN.md`](../PLAN.md) — roadmap and landed items
- [`crates/pyrlx/README.md`](../crates/pyrlx/README.md) — Python install and quickstart
- [`crates/pyrlx/docs/dsl.md`](../crates/pyrlx/docs/dsl.md) — `graph()` / `Node` DSL, scalar literals, operators
- [`crates/pyrlx/docs/quickstart.md`](../crates/pyrlx/docs/quickstart.md) — explicit vs DSL builders
- [`crates/pyrlx/docs/backends.md`](../crates/pyrlx/docs/backends.md) — maturin feature matrix
- [`crates/pyrlx/examples/dsl_quickstart.py`](../crates/pyrlx/examples/dsl_quickstart.py) — runnable DSL demo
## License

GPL-3.0-only.
