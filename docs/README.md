# RLX documentation

| Document | Contents |
|----------|----------|
| [fk-fusion.md](fk-fusion.md) | FKL-style region fusion: prologue, batch regions, env/session toggles, kernel tuning |
| [backend-selection.md](backend-selection.md) | Multi-backend runtime: `DevicePolicy`, `GraphDevices`, `DeviceRouter`, env vars, Python API, calibration |
| [development.md](development.md) | Dev workflow: `just` recipes, pyrlx, tests, dispatch probes |
| [benchmarks/higher-order-ad.md](benchmarks/higher-order-ad.md) | Higher-order autodiff benchmarks |
| [benchmarks/mlx-linux.md](benchmarks/mlx-linux.md) | MLX on Linux/WSL: compile, CPU vs CUDA, vs `rlx-cpu` matmul benches |

Release notes: [`CHANGELOG.md`](../CHANGELOG.md) (workspace **0.2.7**).

Related repo docs:

- [`llms.txt`](../llms.txt) — workspace map for agents
- [`PLAN.md`](../PLAN.md) — roadmap and landed items
- [`pyrlx/README.md`](../pyrlx/README.md) — Python install and quickstart
- [`pyrlx/docs/dsl.md`](../pyrlx/docs/dsl.md) — `graph()` / `Node` DSL, scalar literals, operators
- [`pyrlx/docs/quickstart.md`](../pyrlx/docs/quickstart.md) — explicit vs DSL builders
- [`pyrlx/docs/backends.md`](../pyrlx/docs/backends.md) — maturin feature matrix
- [`pyrlx/examples/dsl_quickstart.py`](../pyrlx/examples/dsl_quickstart.py) — runnable DSL demo
## License

GPL-3.0-only.
