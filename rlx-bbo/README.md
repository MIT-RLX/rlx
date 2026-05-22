# rlx-bbo

Generic black-box optimization and **Flow Map Q-Guidance (FMQ)** / **QGBS** for any `f64` objective — no simulator bindings.

Graph-based offline/online FMQ training (compiled MLP actor + twin critic on `rlx-ir`) lives in [`rlx-rl`](../rlx-rl/).  
Reference JAX code: [`/Users/Shared/q-guided-flow-map-policies`](/Users/Shared/q-guided-flow-map-policies).

## Modules

| Module | Role |
|--------|------|
| `q_guidance` | Trust-region step, `q_steered_search`, `q_guided_beam_search` |
| `twin` | Cheap vs expensive twin-η search |
| `surrogate` | Linear ridge critic from trajectory JSONL |
| `flow_map` | Diagonal linear flow-map offline training |
| `trajectory` | JSONL logging for offline datasets |
| `cmaes` | Separable CMA-ES |

## rlx-eda

Depend on `rlx-bbo` directly (workspace path `../rlx/rlx-bbo`). EDA-specific prescreen↔ngspice twins and harness JSON helpers stay in `eda-fmq`.
