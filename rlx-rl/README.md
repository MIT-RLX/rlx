# rlx-rl

Flow-map generative policies with **Flow Map Q-Guidance (FMQ)** and **Q-Guided Beam Search (QGBS)** on RLX ([arxiv:2605.12416](https://arxiv.org/abs/2605.12416)).

**Reference implementation (JAX):** [`/Users/Shared/q-guided-flow-map-policies`](/Users/Shared/q-guided-flow-map-policies) — see [`REFERENCE.md`](REFERENCE.md) for a module-by-module map.

## Design

| Principle | Implementation |
|-----------|----------------|
| **MLP actor/critic** | `rlx-ir` graphs in [`graph/`](src/graph/) — not [`rlx-flow`](../rlx-flow/) |
| **CPU + autodiff** | [`Session::new(Device::Cpu)`](src/trainer.rs) + `legalize_broadcast` → `grad_with_loss` |
| **No sim bindings** | Implement [`RlEnv`](src/env.rs); store [`Transition`](src/buffer.rs) in [`ReplayBuffer`](src/buffer.rs) |
| **Optional QGBS at eval** | [`EvalConfig::with_qgbs`](src/policy.rs) → Algorithm 2 over [`CompiledFlowMapAgent`](src/graph/actor.rs) |
| **Offline ESD + curriculum** | [`flow_curriculum`](src/flow_curriculum.rs) + [`distillation`](src/distillation.rs) (`mf` / `lsd` / `psd`) |

## Plug in your environment

```rust
use rlx_rl::{
    buffer::Transition, dataset::OfflineDataset, env::RlEnv, policy::EvalConfig,
    spec::RlSpec, FmqTrainer, QgbsConfig,
};

struct MyEnv { /* your state */ }

impl RlEnv for MyEnv {
    fn reset(&mut self) -> Vec<f32> { /* state */ }
    fn step(&mut self, action: &[f32]) -> Transition {
        // fill state, action, reward, next_state, done
        todo!()
    }
}

let spec = RlSpec { state_dim: 12, action_dim: 7, batch: 32, hidden: vec![256, 256], ..RlSpec::toy(32) };
let mut trainer = FmqTrainer::new(spec);

// Offline CFM from demonstrations
trainer.offline_pretrain(&offline_dataset, 10_000);

// Online FMQ (no simulator inside RLX)
let mut env = MyEnv::default();
trainer.online_finetune(&mut env, 50_000);

// Eval: one-step (default)
let r0 = trainer.eval_rollout(&mut env, &EvalConfig::one_step());

// Eval: optional QGBS
let eval = EvalConfig::with_qgbs(QgbsConfig::default());
let r1 = trainer.eval_rollout(&mut env, &eval);
```

Custom online loop without `RlEnv`:

```rust
let tr: Transition = /* from your stack */;
trainer.online_step_from_transition(&tr);
```

## Toy example (feature `toy`)

```bash
cargo run -p rlx-rl --example fmq_toy --features "compile,toy"
cargo test -p rlx-rl --features "compile,toy"
```

## Flow map + FMQ

\[
X_{r,t}(a_r \mid s) = a_r + (t-r)\, u_{r,t}(a_r \mid s), \quad a_1 = X_{0,1}(a_0 \mid s)
\]

Online FMQ: project \(a_1\) with \(\nabla_a Q\) inside a trust region, then regress \(u_{0,1}\) toward \(a_1^* - a_0\).
