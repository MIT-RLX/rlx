// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! On-device (ANE) training for the CoreML backend.
//!
//! Apple's `MLUpdateTask` only updates Core ML **updatable models**, which
//! require the legacy *NeuralNetwork* spec format plus a fixed set of updatable
//! layers (innerProduct / convolution), a built-in loss (categorical
//! cross-entropy or MSE) and a built-in optimizer (SGD / Adam). `rlx-coreml`
//! emits **ML Program** (MIL) models — which `MLUpdateTask` cannot update — so
//! on-device training takes the **gradient path**: the autodiff backward graph
//! runs on [`Device::Ane`] to produce gradients, and a host-side optimizer step
//! updates the weights. This matches the "best-effort `MLUpdateTask`, fall back
//! to gradients" contract: [`CoremlTrainingSession::mlupdatetask_eligibility`]
//! reports *why* the task path is unavailable and [`step`](CoremlTrainingSession::step)
//! transparently uses gradients.
//!
//! Reaching [`UpdatePath::MlUpdateTask`] would need a NeuralNetwork-format
//! updatable-model lowering (a separate emitter from the MIL one) plus an
//! `MLUpdateTask` FFI; that is the Path-A follow-up. The gradient path here is
//! the genuine, validated on-device training capability today.

use std::collections::HashMap;

use rlx_driver::Device;
use rlx_ir::op::BinaryOp;
use rlx_ir::{Graph, NodeId, Op, Shape};

use crate::{CompiledGraph, Session};

/// A `Constant` node holding `value` repeated to fill `shape` — a broadcast-free
/// scalar for the in-graph optimizer update.
fn const_full(g: &mut Graph, value: f32, shape: &Shape) -> NodeId {
    let n = shape.num_elements().unwrap_or(1);
    let data: Vec<u8> = std::iter::repeat_n(value, n)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    g.add_node(Op::Constant { data }, vec![], shape.clone())
}

/// Host-side optimizer applied to the ANE-computed gradients.
///
/// A lightweight `Copy` descriptor; the actual stateful stepper (which owns the
/// per-parameter moment/preconditioner buffers) comes from the [`rlx-optim`]
/// crate and is built by [`Optimizer::build`] when the session is constructed.
#[derive(Debug, Clone, Copy)]
pub enum Optimizer {
    /// `v ← μ·v + g;  w ← w − lr·v`  (μ = 0 ⇒ plain SGD).
    Sgd { momentum: f32 },
    /// Bias-corrected Adam.
    Adam { beta1: f32, beta2: f32, eps: f32 },
    /// AdamW — Adam with decoupled weight decay. The default for transformers.
    AdamW {
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    },
    /// Lion — sign-of-EMA update; memory-light, large effective LR.
    Lion {
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
    },
    /// Muon — Newton–Schulz-orthogonalized momentum (2-D weights).
    Muon { momentum: f32, ns_steps: u32 },
    /// Sophia-H — diagonal-Hessian-preconditioned (2nd-order-ish) step.
    Sophia {
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
    },
}

impl Optimizer {
    /// Plain SGD (no momentum).
    pub fn sgd() -> Self {
        Self::Sgd { momentum: 0.0 }
    }
    /// Adam with the usual defaults.
    pub fn adam() -> Self {
        Self::Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
    /// AdamW (β=(0.9,0.999), ε=1e-8, weight decay 0.01).
    pub fn adamw() -> Self {
        Self::AdamW {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
    /// Lion (β=(0.9,0.99), weight decay 0.01).
    pub fn lion() -> Self {
        Self::Lion {
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.01,
        }
    }
    /// Muon (momentum 0.95, 5 Newton–Schulz steps).
    pub fn muon() -> Self {
        Self::Muon {
            momentum: 0.95,
            ns_steps: 5,
        }
    }
    /// Sophia-H (β=(0.965,0.99), weight decay 0.1).
    pub fn sophia() -> Self {
        Self::Sophia {
            beta1: 0.965,
            beta2: 0.99,
            weight_decay: 0.1,
        }
    }

    /// Build the stateful `rlx-optim` stepper at the given base learning rate.
    fn build(&self, lr: f32) -> Box<dyn rlx_optim::Optimizer> {
        match *self {
            Optimizer::Sgd { momentum } => {
                Box::new(rlx_optim::Sgd::new(lr).with_momentum(momentum, false))
            }
            Optimizer::Adam { beta1, beta2, eps } => Box::new(
                rlx_optim::Adam::new(lr)
                    .with_betas(beta1, beta2)
                    .with_eps(eps),
            ),
            Optimizer::AdamW {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => Box::new(
                rlx_optim::AdamW::new(lr)
                    .with_betas(beta1, beta2)
                    .with_eps(eps)
                    .with_weight_decay(weight_decay),
            ),
            Optimizer::Lion {
                beta1,
                beta2,
                weight_decay,
            } => Box::new(
                rlx_optim::Lion::new(lr)
                    .with_betas(beta1, beta2)
                    .with_weight_decay(weight_decay),
            ),
            Optimizer::Muon { momentum, ns_steps } => Box::new(
                rlx_optim::Muon::new(lr)
                    .with_momentum(momentum)
                    .with_ns_steps(ns_steps),
            ),
            Optimizer::Sophia {
                beta1,
                beta2,
                weight_decay,
            } => Box::new(
                rlx_optim::Sophia::new(lr)
                    .with_betas(beta1, beta2)
                    .with_weight_decay(weight_decay),
            ),
        }
    }
}

/// Learning rate + optimizer for a [`CoremlTrainingSession`].
///
/// **Speed vs precision** is chosen by compute units (`RLX_COREML_UNITS`), not a
/// field here: gradients default to fp32 CPU+GPU (accurate / BNNS-safe), and
/// `RLX_COREML_UNITS=ane` (or an f16 AFP graph) runs on the **Neural Engine**
/// (`f16+f16` hardware) for faster training. See [`rlx_coreml::default_compute_units`].
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub lr: f32,
    pub optimizer: Optimizer,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            optimizer: Optimizer::sgd(),
        }
    }
}

/// Which on-device update path actually ran for a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdatePath {
    /// Core ML `MLUpdateTask` updated the model in place (NeuralNetwork models).
    MlUpdateTask,
    /// ANE-computed gradients + host optimizer step (the path for ML Program models).
    Gradient,
}

/// Whether `MLUpdateTask` can drive this model, and if not, why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlUpdateEligibility {
    /// The model is an updatable NeuralNetwork that `MLUpdateTask` can train.
    Eligible,
    /// `MLUpdateTask` is inapplicable; the gradient path is used instead.
    Unsupported { reason: String },
}

/// Outcome of a single [`CoremlTrainingSession::step`].
#[derive(Debug, Clone, Copy)]
pub struct StepReport {
    /// Loss measured on this batch *before* the weight update.
    pub loss: f32,
    /// Which update path ran.
    pub path: UpdatePath,
}

/// A gradient-based on-device training loop for the CoreML / ANE backend.
///
/// Build it from a forward graph whose single output is a scalar loss and the
/// names of the [`Param`](rlx_ir::Op::Param)s to train; each [`step`](Self::step)
/// runs forward+backward on the configured device (default [`Device::Ane`]) and
/// applies the host optimizer to the held weights.
pub struct CoremlTrainingSession {
    /// `grad_with_loss(forward, trainable)` — outputs `[loss, grad₀, grad₁, …]`.
    backward: Graph,
    /// Trainable param names, in the same order as the backward graph's grads.
    trainable: Vec<String>,
    /// Current values of every param the graph references (trainable + fixed).
    params: HashMap<String, Vec<f32>>,
    cfg: TrainConfig,
    /// Steps taken (reported via [`steps`](Self::steps)).
    t: u64,
    /// The stateful host optimizer (owns per-param moments / preconditioners),
    /// built from `cfg.optimizer`. Stepping is delegated to the `rlx-optim` crate
    /// so the full AdamW/Lion/Muon/Sophia/… suite is available on-device.
    opt: Box<dyn rlx_optim::Optimizer>,
    /// Logical shapes of the trainable params — needed by matrix-aware optimizers
    /// (Muon/Adafactor/SOAP); ignored by elementwise ones (SGD/Adam/Lion).
    shapes: HashMap<String, Vec<usize>>,
    /// Summed gradients across [`accumulate`](Self::accumulate) micro-batches,
    /// per trainable param. Empty unless gradient accumulation is in use.
    accum: HashMap<String, Vec<f32>>,
    /// Number of micro-batches summed into `accum` since the last step.
    accum_count: usize,
    /// When set (via [`with_fused_optimizer`](Self::with_fused_optimizer)) and the
    /// optimizer is momentum-SGD, the weight update runs **on-device**: the graph is
    /// augmented to output new weights + new velocities, computed on the ANE.
    fused: bool,
    /// On-device optimizer state (momentum velocity), per trainable param. Fed as a
    /// graph input and read back from a graph output each fused step.
    vel: HashMap<String, Vec<f32>>,
    /// Velocity input/output names (`"{param}__vel"`), in `trainable` order.
    vel_names: Vec<String>,
    device: Device,
    /// Automatic-Floating-Point policy. `None` = fp32 (precise, CPU+GPU). Set to
    /// `AutoMixed`/`AlwaysF16` for f16 mixed precision → CoreML runs it on the
    /// Neural Engine (fast). See [`Self::with_precision_policy`].
    policy: Option<crate::PrecisionPolicy>,
    /// Backward graph compiled **once** and reused across steps. The trainable
    /// weights are graph `Input`s (not baked `Param`s), so the CoreML model is
    /// static and each step is a plain prediction — no per-step recompile.
    /// Built lazily on the first [`step`](Self::step) (after any
    /// [`with_precision_policy`](Self::with_precision_policy)).
    compiled: Option<CompiledGraph>,
}

impl CoremlTrainingSession {
    /// Train `trainable` params of `forward` on the ANE. `initial` must hold a
    /// value for **every** param the graph references (trainable and fixed).
    pub fn new(
        forward: Graph,
        trainable: &[&str],
        initial: HashMap<String, Vec<f32>>,
        cfg: TrainConfig,
    ) -> Self {
        Self::new_on(forward, trainable, initial, cfg, Device::Ane)
    }

    /// [`new`](Self::new) with an explicit device (e.g. `Device::Cpu` to compute
    /// the same gradient path off the ANE, used by parity tests).
    pub fn new_on(
        forward: Graph,
        trainable: &[&str],
        initial: HashMap<String, Vec<f32>>,
        cfg: TrainConfig,
        device: Device,
    ) -> Self {
        let wrt: Vec<NodeId> = trainable
            .iter()
            .map(|n| {
                forward
                    .param_id(n)
                    .unwrap_or_else(|| panic!("trainable param `{n}` is not a Param in the graph"))
            })
            .collect();
        // Trainable param shapes (from the forward graph, before the rebind below)
        // for matrix-aware optimizers.
        let shapes: HashMap<String, Vec<usize>> = trainable
            .iter()
            .map(|n| {
                let id = forward.param_id(n).expect("trainable param exists");
                let sh = forward.shape(id);
                let dims = (0..sh.rank()).map(|i| sh.dim(i).unwrap_static()).collect();
                (n.to_string(), dims)
            })
            .collect();

        let mut backward = rlx_opt::grad_with_loss(&forward, &wrt);
        // Rebind each trainable weight from a baked `Param` to a runtime `Input`.
        // CoreML bakes `Param`s as model constants, so leaving them as params
        // forces a full re-lower + recompile every step (the weights change each
        // step). As `Input`s the model is compiled once and the current weights
        // are fed per step — turning each step from a recompile into a predict.
        let names: Vec<&str> = trainable.to_vec();
        for node in backward.nodes_mut() {
            if let Op::Param { name } = &node.op {
                if names.contains(&name.as_str()) {
                    node.op = Op::Input { name: name.clone() };
                }
            }
        }
        Self {
            backward,
            trainable: trainable.iter().map(|s| s.to_string()).collect(),
            params: initial,
            opt: cfg.optimizer.build(cfg.lr),
            cfg,
            t: 0,
            shapes,
            accum: HashMap::new(),
            accum_count: 0,
            fused: false,
            vel: HashMap::new(),
            vel_names: trainable.iter().map(|n| format!("{n}__vel")).collect(),
            device,
            policy: None,
            compiled: None,
        }
    }

    /// Train in mixed precision via an Automatic-Floating-Point policy (e.g.
    /// [`PrecisionPolicy::AutoMixed`](crate::PrecisionPolicy)): compute ops run
    /// f16, boundaries stay f32, and CoreML schedules the f16 graph on the
    /// **Neural Engine** for faster (lower-precision) training. Default (unset) is
    /// fp32 on CPU+GPU.
    pub fn with_precision_policy(mut self, policy: crate::PrecisionPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Run the optimizer **on-device**: the backward graph is augmented to compute
    /// the weight update (and updated optimizer state) on the ANE, so a step is a
    /// single fused device dispatch with no host-side optimizer loop. Currently
    /// applies to momentum-SGD (`Optimizer::Sgd { momentum > 0 }`) — the velocity
    /// buffer becomes device-resident graph I/O. For other optimizers this is a
    /// no-op and the host `rlx-optim` path runs (see [`fused_active`](Self::fused_active)).
    pub fn with_fused_optimizer(mut self) -> Self {
        self.fused = true;
        self
    }

    /// Whether the on-device fused-optimizer path is active for this config
    /// (`with_fused_optimizer()` set **and** the optimizer is momentum-SGD).
    pub fn fused_active(&self) -> bool {
        self.fused && matches!(self.cfg.optimizer, Optimizer::Sgd { momentum } if momentum > 0.0)
    }

    /// Whether `MLUpdateTask` can update this model. Always `Unsupported` for an
    /// `rlx-coreml`-emitted graph: those are ML Program (MIL) models, and
    /// `MLUpdateTask` only updates legacy NeuralNetwork updatable models.
    pub fn mlupdatetask_eligibility(&self) -> MlUpdateEligibility {
        MlUpdateEligibility::Unsupported {
            reason: "rlx-coreml emits ML Program (MIL) models; MLUpdateTask only updates \
                     legacy NeuralNetwork updatable models — using the gradient path"
                .to_string(),
        }
    }

    /// The path [`step`](Self::step) will take for this model.
    pub fn update_path(&self) -> UpdatePath {
        match self.mlupdatetask_eligibility() {
            MlUpdateEligibility::Eligible => UpdatePath::MlUpdateTask,
            MlUpdateEligibility::Unsupported { .. } => UpdatePath::Gradient,
        }
    }

    /// Run one training step over a batch. `inputs` are the non-trainable feeds
    /// (the forward graph's `Input`s, e.g. features + targets); the scalar loss
    /// seed is supplied internally. Returns the pre-update loss and the path taken.
    pub fn step(&mut self, inputs: &[(&str, &[f32])]) -> StepReport {
        if self.fused_active() {
            return self.fused_sgd_step(inputs);
        }
        // Best-effort MLUpdateTask, else gradients. The task path is unreachable
        // for ML Program models (see `mlupdatetask_eligibility`); kept explicit so
        // a future NeuralNetwork updatable path slots in here.
        match self.update_path() {
            UpdatePath::MlUpdateTask => self.gradient_step(inputs), // no NN path yet
            UpdatePath::Gradient => self.gradient_step(inputs),
        }
    }

    /// Build the on-device momentum-SGD graph: clone the backward graph and, for
    /// each trainable param, append `vel' = μ·vel + grad; w' = w − lr·vel'` as IR
    /// ops, exposing `vel` as an input and `(w', vel')` as outputs. The returned
    /// graph outputs are `[loss, w'₀, vel'₀, w'₁, vel'₁, …]`.
    fn build_fused_sgd_graph(&self, mu: f32, lr: f32) -> Graph {
        let mut g = self.backward.clone();
        let mut outs = vec![g.outputs[0]];
        for (i, name) in self.trainable.iter().enumerate() {
            let grad = g.outputs[1 + i];
            let w = g
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, Op::Input { name: n2 } if n2 == name))
                .map(|n| n.id)
                .expect("trainable weight is a graph input");
            let shape = g.node(w).shape.clone();
            let vel = g.input(format!("{name}__vel"), shape.clone());
            let mu_c = const_full(&mut g, mu, &shape);
            let lr_c = const_full(&mut g, lr, &shape);
            let mu_v = g.binary(BinaryOp::Mul, vel, mu_c, shape.clone());
            let vel_new = g.binary(BinaryOp::Add, mu_v, grad, shape.clone());
            let lr_vn = g.binary(BinaryOp::Mul, vel_new, lr_c, shape.clone());
            let w_new = g.binary(BinaryOp::Sub, w, lr_vn, shape.clone());
            outs.push(w_new);
            outs.push(vel_new);
        }
        g.set_outputs(outs);
        g
    }

    /// On-device momentum-SGD step. Feeds weights + velocities, runs the fused graph
    /// on the ANE (which emits new weights + velocities), and stores them back — no
    /// host-side optimizer math. Outputs are `[loss, w'₀, vel'₀, …]`.
    fn fused_sgd_step(&mut self, inputs: &[(&str, &[f32])]) -> StepReport {
        let (mu, lr) = match self.cfg.optimizer {
            Optimizer::Sgd { momentum } => (momentum, self.cfg.lr),
            _ => unreachable!("fused_active gates on Sgd"),
        };
        if self.compiled.is_none() {
            let fg = self.build_fused_sgd_graph(mu, lr);
            let session = Session::new(self.device);
            let session = match self.policy.clone() {
                Some(p) => session.with_policy(p),
                None => session,
            };
            let mut compiled = session.compile(fg);
            for (name, data) in &self.params {
                if !self.trainable.iter().any(|t| t == name) {
                    compiled.set_param(name, data);
                }
            }
            self.compiled = Some(compiled);
        }
        // Velocities start at zero.
        for name in self.trainable.clone() {
            let n = self.params[&name].len();
            self.vel.entry(name).or_insert_with(|| vec![0.0; n]);
        }

        let seed = [1.0f32];
        let outs = {
            let mut feeds: Vec<(&str, &[f32])> = inputs.to_vec();
            feeds.push(("d_output", &seed));
            for name in &self.trainable {
                feeds.push((name.as_str(), self.params[name].as_slice()));
            }
            for (name, vname) in self.trainable.iter().zip(&self.vel_names) {
                feeds.push((vname.as_str(), self.vel[name].as_slice()));
            }
            self.compiled.as_mut().unwrap().run(&feeds)
        };

        let loss = outs
            .first()
            .and_then(|o| o.first())
            .copied()
            .unwrap_or(f32::NAN);
        // outputs = [loss, w'₀, vel'₀, w'₁, vel'₁, …]
        let names = self.trainable.clone();
        for (i, name) in names.iter().enumerate() {
            if let Some(w_new) = outs.get(1 + 2 * i) {
                self.params.insert(name.clone(), w_new.clone());
            }
            if let Some(vel_new) = outs.get(2 + 2 * i) {
                self.vel.insert(name.clone(), vel_new.clone());
            }
        }
        self.t += 1;
        StepReport {
            loss,
            path: UpdatePath::Gradient,
        }
    }

    /// Run forward+backward on the device once; return the pre-update loss and each
    /// trainable's gradient (in `self.trainable` order). Compiles the static backward
    /// model on first use and reuses it. Shared by [`step`](Self::step) and
    /// [`accumulate`](Self::accumulate).
    fn compute_grads(&mut self, inputs: &[(&str, &[f32])]) -> (f32, Vec<Vec<f32>>) {
        // Compile the static backward model **once** (trainable weights are now
        // graph `Input`s — see `new_on`). A precision policy makes the graph f16
        // (→ ANE, fast); without one it's fp32 (→ CPU+GPU, precise) — resolved by
        // `rlx_coreml::default_compute_units`.
        if self.compiled.is_none() {
            let session = Session::new(self.device);
            let session = match self.policy.clone() {
                Some(p) => session.with_policy(p),
                None => session,
            };
            let mut compiled = session.compile(self.backward.clone());
            // Bake only the *non-trainable* params (fixed across steps); trainable
            // weights are fed as inputs below.
            for (name, data) in &self.params {
                if !self.trainable.iter().any(|t| t == name) {
                    compiled.set_param(name, data);
                }
            }
            self.compiled = Some(compiled);
        }

        // Backward feeds: forward inputs + scalar loss cotangent + current
        // trainable weights (as runtime inputs).
        let seed = [1.0f32];
        let outs = {
            let mut feeds: Vec<(&str, &[f32])> = inputs.to_vec();
            feeds.push(("d_output", &seed));
            for name in &self.trainable {
                let w = self.params.get(name).expect("trainable param has a value");
                feeds.push((name.as_str(), w.as_slice()));
            }
            self.compiled.as_mut().unwrap().run(&feeds)
        };

        // outputs = [loss, grad(trainable[0]), grad(trainable[1]), …]
        let loss = outs
            .first()
            .and_then(|o| o.first())
            .copied()
            .unwrap_or(f32::NAN);
        let grads: Vec<Vec<f32>> = (0..self.trainable.len())
            .map(|i| outs.get(1 + i).cloned().unwrap_or_default())
            .collect();
        (loss, grads)
    }

    fn gradient_step(&mut self, inputs: &[(&str, &[f32])]) -> StepReport {
        let (loss, grads) = self.compute_grads(inputs);
        let names = self.trainable.clone();
        for (name, grad) in names.iter().zip(&grads) {
            self.apply_update(name, grad);
        }
        // One global iteration per training step (advances the optimizers'
        // bias-correction counter once, regardless of how many params were stepped).
        self.opt.end_iteration();
        self.t += 1;
        StepReport {
            loss,
            path: UpdatePath::Gradient,
        }
    }

    /// Accumulate one micro-batch's gradients **without** stepping. Run it over
    /// several micro-batches, then [`step_accumulated`](Self::step_accumulated) to
    /// apply the optimizer to their mean — an effective batch larger than one ANE
    /// prediction can hold, at no extra weight-update cost. Returns the micro-batch
    /// loss (pre-update). The optimizer state and weights are untouched until the step.
    pub fn accumulate(&mut self, inputs: &[(&str, &[f32])]) -> f32 {
        let (loss, grads) = self.compute_grads(inputs);
        let names = self.trainable.clone();
        for (name, grad) in names.iter().zip(&grads) {
            let acc = self
                .accum
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; grad.len()]);
            if acc.len() == grad.len() {
                for (a, &g) in acc.iter_mut().zip(grad) {
                    *a += g;
                }
            }
        }
        self.accum_count += 1;
        loss
    }

    /// Apply the optimizer to the **mean** of the accumulated gradients, then clear
    /// the accumulators. No-op (returns 0) if nothing was accumulated. Returns the
    /// number of micro-batches that were averaged.
    pub fn step_accumulated(&mut self) -> usize {
        let n = self.accum_count;
        if n == 0 {
            return 0;
        }
        let inv = 1.0 / n as f32;
        let names = self.trainable.clone();
        for name in &names {
            let mean: Vec<f32> = match self.accum.get(name) {
                Some(acc) => acc.iter().map(|&a| a * inv).collect(),
                None => continue,
            };
            self.apply_update(name, &mean);
        }
        self.opt.end_iteration();
        self.t += 1;
        self.accum.clear();
        self.accum_count = 0;
        n
    }

    /// Number of micro-batches accumulated since the last step (0 when idle).
    pub fn pending_accumulation(&self) -> usize {
        self.accum_count
    }

    /// Apply the `rlx-optim` stepper to one trainable param. `opt` / `params` /
    /// `shapes` are distinct fields, so the borrows don't conflict.
    fn apply_update(&mut self, name: &str, grad: &[f32]) {
        let shape = self
            .shapes
            .get(name)
            .cloned()
            .unwrap_or_else(|| vec![grad.len()]);
        let w = self
            .params
            .get_mut(name)
            .expect("trainable param has a value");
        debug_assert_eq!(w.len(), grad.len(), "grad shape must match weight `{name}`");
        self.opt.step(name, &shape, w, grad);
    }

    /// Current value of a trained (or fixed) parameter.
    pub fn param(&self, name: &str) -> Option<&[f32]> {
        self.params.get(name).map(|v| v.as_slice())
    }

    /// All current parameter values.
    pub fn params(&self) -> &HashMap<String, Vec<f32>> {
        &self.params
    }

    /// Number of steps taken.
    pub fn steps(&self) -> u64 {
        self.t
    }

    /// The learning-rate + optimizer descriptor this session was built with.
    pub fn config(&self) -> TrainConfig {
        self.cfg
    }
}
