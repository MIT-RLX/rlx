# Reference: Python → rlx-rl

Official JAX implementation: [`/Users/Shared/q-guided-flow-map-policies`](/Users/Shared/q-guided-flow-map-policies)  
Paper: [arxiv:2605.12416](https://arxiv.org/abs/2605.12416)

## Module map

| Python | rlx-rl | Status |
|--------|--------|--------|
| `agents/flow_utils.py` | `flow_curriculum.rs` | `sample_r_t` |
| `agents/flow_map_policy.py` | `trainer::offline_pretrain` | L_Diag + L_ESD (mf/lsd/psd) + critic |
| `agents/fmq.py` | `trainer::fmq_actor_update` | L_FMQ + optional `esd_weight` / `diag_weight` |
| `agents/qgbs.py` | `qgbs.rs` | Algorithm 2 (renoise + diagonal velocity + beam + η projection) |
| `utils/networks.py` `ActorVectorFieldMFM` | `graph/actor.rs` | `concat(s, a_r, r, t)` MLP |
| `utils/networks.py` `Value` | `graph/critic.rs` | Twin Q |
| `envs/*` | `env::RlEnv` | Plug your simulator |
| `main.py` | `FmqTrainer` | CPU `Session` |

## Training pipeline

1. **Offline** — `offline_pretrain`: L_Diag every step; L_ESD after `flow_map_warmup_steps`; critic TD on dataset transitions.
2. **Freeze** — `freeze_offline_anchor` (end of offline).
3. **Online** — `online_finetune`: L_FMQ + optional auxiliary ESD/diag + critic on replay.
4. **Eval** — `EvalConfig::best_of_n(M)` or `EvalConfig::with_qgbs(QgbsConfig::from_spec(&spec))`.

## Distillation types (`distillation_type`)

| Value | Python | rlx-rl |
|-------|--------|--------|
| `mf` | Eulerian JVP (Eq. 5) | `distillation::esd_teacher_mf` |
| `lsd` | Lagrangian JVP on jump | `distillation::esd_lsd_sample` |
| `psd` | Progressive blend | `distillation::esd_teacher_psd` |

ESD teachers use host `CompiledFlowMapAgent::velocity` + finite-difference JVP (`JVP_EPS`).

## Config fields (`RlSpec`)

Maps to `configs/config.yaml`: `flow_map_warmup_steps`, `flow_map_anneal_end_step`, `fmq_*`, `qgbs_eta`, `actor_num_samples`, `esd_weight`, `diag_weight`, `distillation_type`.

## Not ported

- Fourier features / encoders (`use_fourier_features`, Impala)
- Action chunking / `horizon_length` stacking
- OGBench / RoboMimic loaders (`envs/`, `datasets/`)
- WandB / HF checkpoint layout
- `log_alpha` adaptive FMQ α
