// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Training: `Optimizer` (rlx-optim over caller-owned arrays) and `Trainer`
//! (differentiate once, compile once, keep the params, step).
//!
//! The forward graph must have exactly one output, the scalar loss — `grad`
//! seeds `d_output` with `1`, which only means "differentiate the loss" when
//! the loss is what the graph ends on.
//!
//! # Where the update runs
//!
//! The **host** path runs the optimizer on the host and re-uploads the weights
//! each step — correct everywhere. The **resident** path fuses AdamW into the
//! compiled graph (`rlx_runtime::train::ResidentTrainer`): parameters and
//! moments stay on the device, worth 1.4–2.1× above ~200 k parameters. It is
//! offered only where it is verified correct and faster (Metal, wgpu); `cpu`
//! is refused (no device buffers — measured 3.3× slower), and other GPUs need
//! an explicit `resident = "unverified"`.
//!
//! # Checkpoints
//!
//! GGUF, so every rlx tool reads them: `param/<name>` (trainable),
//! `fixed/<name>` (no gradient), `opt/<slot>/<name>` (optimizer accumulators);
//! the header carries the step, lr and algorithm. Both paths write the same
//! layout, so a host checkpoint resumes on the resident path and back.
//!
//! The training *loop* (`run`) is the binding's: a callback-driven loop over
//! [`Trainer::step`], so no handle stays borrowed while user code runs.
//! [`RunStats`] keeps the summary the same in every language.

use std::collections::HashMap;

use rlx_ir::{Dim, Graph, NodeId, Op};
use rlx_optim::{
    Adafactor, Adam, AdamW, Lamb, Lion, Mars, Muon, NAdamW, OptItem, Optimizer as OptimizerTrait,
    QHAdamW, RAdam, Sgd, Soap, Sophia,
};
use rlx_runtime::train::ResidentTrainer;
use rlx_runtime::{CompiledGraph, Device, PrecisionPolicy};
use serde_json::{Value, json};

use super::opts::Opts;
use crate::panics::catch;
use crate::{ApiError, ApiResult};

// ── optimizer construction ──────────────────────────────────

/// Every optimizer kind, for error messages and generators.
pub const KINDS: &[&str] = &[
    "sgd",
    "adam",
    "adamw",
    "lion",
    "muon",
    "radam",
    "nadamw",
    "lamb",
    "mars",
    "soap",
    "sophia",
    "qhadamw",
    "adafactor",
];

/// What an optimizer config resolved to.
pub struct OptimizerSpec {
    pub optimizer: Box<dyn OptimizerTrait>,
    /// Lowercased algorithm name, recorded in checkpoints.
    pub kind: String,
    /// The learning rate it starts at.
    pub lr: f32,
    /// The raw options, for the resident path's AdamW hyperparameters.
    options: Value,
}

/// Build one of rlx-optim's algorithms. `kind` wins over `opts.kind`; both
/// absent means `adamw`. `opts` may also be just the kind's name. Options: `lr`, `beta1`, `beta2`, `eps`,
/// `weight_decay`, `momentum`, `nesterov`, `f32_math`, `gamma`, `rho`.
///
/// An unknown `kind` or option is an error: silently falling back would make a
/// config typo look like a bad learning rate.
pub fn optimizer(kind: Option<&str>, opts: &Value) -> ApiResult<OptimizerSpec> {
    // A bare name is the kind with every hyperparameter at its default.
    if let (None, Value::String(name)) = (kind, opts) {
        return optimizer(Some(name), &Value::Null);
    }
    let o = Opts::new("optimizer", opts)?;
    let kind = kind
        .map(str::to_string)
        .or(o.str("kind")?.map(str::to_string))
        .unwrap_or_else(|| "adamw".to_string())
        .trim()
        .to_ascii_lowercase();
    let lr_given = o.f64("lr")?;
    let lr = lr_given.map(|v| v as f32).unwrap_or(1e-3);
    let num = |name: &str, default: f32| o.f32(name, default);
    let optimizer: Box<dyn OptimizerTrait> = match kind.as_str() {
        "sgd" => {
            let mut x = Sgd::new(lr);
            x.momentum = num("momentum", x.momentum)?;
            x.nesterov = o.bool("nesterov", x.nesterov)?;
            x.weight_decay = num("weight_decay", 0.0)?;
            Box::new(x)
        }
        "adam" => {
            let mut x = Adam::new(lr);
            x.beta1 = num("beta1", x.beta1)?;
            x.beta2 = num("beta2", x.beta2)?;
            x.eps = num("eps", x.eps)?;
            x.weight_decay = num("weight_decay", 0.0)?;
            // f32 moment arithmetic is 2.1x faster than the f64-intermediate
            // default and is what PyTorch does.
            x.f32_math = o.bool("f32_math", x.f32_math)?;
            Box::new(x)
        }
        "adamw" => {
            let mut x = AdamW::new(lr);
            x.beta1 = num("beta1", x.beta1)?;
            x.beta2 = num("beta2", x.beta2)?;
            x.eps = num("eps", x.eps)?;
            x.weight_decay = num("weight_decay", x.weight_decay)?;
            x.f32_math = o.bool("f32_math", x.f32_math)?;
            Box::new(x)
        }
        "lion" => {
            let mut x = Lion::new(lr);
            x.beta1 = num("beta1", x.beta1)?;
            x.beta2 = num("beta2", x.beta2)?;
            x.weight_decay = num("weight_decay", x.weight_decay)?;
            Box::new(x)
        }
        "muon" => {
            let mut x = Muon::new(lr);
            x.momentum = num("momentum", x.momentum)?;
            x.nesterov = o.bool("nesterov", x.nesterov)?;
            x.weight_decay = num("weight_decay", x.weight_decay)?;
            Box::new(x)
        }
        "radam" => {
            let mut x = RAdam::new(lr);
            x.beta1 = num("beta1", x.beta1)?;
            x.beta2 = num("beta2", x.beta2)?;
            x.eps = num("eps", x.eps)?;
            x.weight_decay = num("weight_decay", 0.0)?;
            Box::new(x)
        }
        "nadamw" => {
            let mut x = NAdamW::new(lr);
            x.beta1 = num("beta1", x.beta1)?;
            x.beta2 = num("beta2", x.beta2)?;
            x.eps = num("eps", x.eps)?;
            x.weight_decay = num("weight_decay", x.weight_decay)?;
            Box::new(x)
        }
        "lamb" => {
            let mut x = Lamb::new(lr);
            x.beta1 = num("beta1", x.beta1)?;
            x.beta2 = num("beta2", x.beta2)?;
            x.eps = num("eps", x.eps)?;
            x.weight_decay = num("weight_decay", 0.0)?;
            Box::new(x)
        }
        "mars" => {
            let mut x = Mars::new(lr);
            x.beta1 = num("beta1", x.beta1)?;
            x.beta2 = num("beta2", x.beta2)?;
            x.eps = num("eps", x.eps)?;
            x.gamma = num("gamma", x.gamma)?;
            x.weight_decay = num("weight_decay", 0.0)?;
            Box::new(x)
        }
        "soap" => {
            let mut x = Soap::new(lr);
            x.beta1 = num("beta1", x.beta1)?;
            x.beta2 = num("beta2", x.beta2)?;
            x.eps = num("eps", x.eps)?;
            x.weight_decay = num("weight_decay", 0.0)?;
            Box::new(x)
        }
        "sophia" => {
            let mut x = Sophia::new(lr);
            x.gamma = num("gamma", x.gamma)?;
            x.rho = num("rho", x.rho)?;
            x.eps = num("eps", x.eps)?;
            x.weight_decay = num("weight_decay", 0.0)?;
            Box::new(x)
        }
        "qhadamw" => {
            let mut x = QHAdamW::new(lr);
            x.beta1 = num("beta1", x.beta1)?;
            x.beta2 = num("beta2", x.beta2)?;
            x.eps = num("eps", x.eps)?;
            x.weight_decay = num("weight_decay", x.weight_decay)?;
            Box::new(x)
        }
        "adafactor" => {
            let mut x = Adafactor::new();
            // A relative step size means "no lr" is a real choice, not a
            // missing one — only override when the caller says so.
            if let Some(lr) = lr_given {
                x.lr = Some(lr as f32);
            }
            x.weight_decay = num("weight_decay", 0.0)?;
            Box::new(x)
        }
        other => {
            return Err(ApiError::type_(format!(
                "unknown optimizer '{other}' ({})",
                KINDS.join(", ")
            )));
        }
    };
    o.finish()?;
    Ok(OptimizerSpec {
        optimizer,
        kind,
        lr,
        options: opts.clone(),
    })
}

/// A standalone optimizer over arrays the caller owns.
pub struct Optimizer {
    pub inner: Box<dyn OptimizerTrait>,
    pub kind: String,
}

/// One parameter of an [`Optimizer::step`].
pub struct StepItem {
    pub name: String,
    pub shape: Vec<usize>,
    pub param: Vec<f32>,
    pub grad: Vec<f32>,
}

impl Optimizer {
    pub fn new(kind: Option<&str>, opts: &Value) -> ApiResult<Self> {
        let spec = optimizer(kind, opts)?;
        Ok(Self {
            inner: spec.optimizer,
            kind: spec.kind,
        })
    }

    /// Update every `param` in place from its `grad`. The optimizer keys its
    /// state by name, so pass the same names every step.
    pub fn step(&mut self, items: &mut [StepItem]) -> ApiResult<()> {
        for it in items.iter() {
            if it.param.len() != it.grad.len() {
                return Err(ApiError::type_(format!(
                    "optimizer.step('{}'): param has {} values but grad has {}",
                    it.name,
                    it.param.len(),
                    it.grad.len()
                )));
            }
        }
        let mut refs: Vec<OptItem<'_>> = items
            .iter_mut()
            .map(|it| OptItem {
                name: it.name.as_str(),
                shape: it.shape.as_slice(),
                param: it.param.as_mut_slice(),
                grad: it.grad.as_slice(),
            })
            .collect();
        catch("optimizer.step", || {
            self.inner.step_batch(&mut refs);
            self.inner.end_iteration();
        })
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }
}

// ── Trainer ─────────────────────────────────────────────────

/// Whether to fuse the update into the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Resident {
    #[default]
    Off,
    /// Where verified (Metal, wgpu).
    On,
    /// Also on backends nobody has verified it on — compare a few steps first.
    Unverified,
}

impl Resident {
    /// `true` / `false` / `"unverified"` / null.
    pub fn from_json(v: &Value) -> ApiResult<Self> {
        Ok(match v {
            Value::Null | Value::Bool(false) => Resident::Off,
            Value::Bool(true) => Resident::On,
            Value::String(s) if s.eq_ignore_ascii_case("unverified") => Resident::Unverified,
            other => {
                return Err(ApiError::type_(format!(
                    "resident must be true, false or \"unverified\", got {other}"
                )));
            }
        })
    }
}

/// Everything a [`Trainer`] is built from, besides the graph.
pub struct TrainerConfig {
    /// Parameters to train, by name.
    pub wrt: Vec<String>,
    /// Initial values for every parameter; those not in `wrt` are uploaded
    /// once and never updated (how LoRA freezes a base weight).
    pub init: Vec<(String, Vec<f32>)>,
    pub device: String,
    /// The optimizer config (see [`optimizer`]): `{kind, lr, …}`.
    pub optimizer: Value,
    pub policy: Option<PrecisionPolicy>,
    /// Global gradient-norm clip across all trainable params together.
    pub clip_norm: Option<f32>,
    pub resident: Resident,
}

/// Differentiate-once / compile-once training over a scalar-loss graph.
pub struct Trainer {
    /// The host path's compiled backward graph; `None` on the resident path.
    compiled: Option<CompiledGraph>,
    resident: Option<ResidentTrainer>,
    optimizer: Box<dyn OptimizerTrait>,
    /// Recorded and checked in checkpoints — Adam moments loaded into Lion
    /// would be nonsense.
    optimizer_kind: String,
    /// Trainable parameters, in the order `grad` emits their gradients.
    names: Vec<String>,
    shapes: Vec<Vec<usize>>,
    /// Host-path weights. Empty on the resident path (they live on the device).
    values: Vec<Vec<f32>>,
    /// Per trainable param: skip its update. The gradient is still computed —
    /// `wrt` is baked into the compiled graph — so this changes what moves,
    /// not what it costs.
    frozen: Vec<bool>,
    /// `init` entries not in `wrt`: uploaded once, kept so a checkpoint is
    /// complete and `params()` reports them.
    fixed: Vec<(String, Vec<f32>)>,
    device: Device,
    steps: u64,
    lr: f32,
    clip_norm: Option<f32>,
    last_grad_norm: f32,
    /// A trainer of a package ([`Trainer::from_package`]): what writes its
    /// results back as a package.
    package: Option<PackageSaver>,
}

#[cfg(feature = "models")]
type PackageSaver = rlx_runtime::model::trainer::Saver;
#[cfg(not(feature = "models"))]
type PackageSaver = std::convert::Infallible;

#[cfg(feature = "models")]
impl Trainer {
    /// Train a function of the package at `path`, whatever the model is
    /// (`rlx_runtime::model::trainer::prepare`). options: `{function, dims,
    /// loss, train, optimizer}` (over the package's own `train`) and the
    /// trainer's `{device, policy, clip_norm, resident}`.
    pub fn from_package(path: &str, opts: &Value) -> ApiResult<Self> {
        let prepared = rlx_runtime::model::trainer::prepare(path, opts)
            .map_err(|e| ApiError::type_(format!("Trainer.from_package: {e:#}")))?;
        let (graph, wrt, init, optimizer, saver) = prepared.split();
        let mut optimizer = optimizer;
        if optimizer.get("kind").is_none() {
            optimizer["kind"] = json!("adamw");
        }
        let device = opts
            .get("device")
            .and_then(Value::as_str)
            .unwrap_or("cpu")
            .to_string();
        let policy = match opts.get("policy") {
            Some(p) => super::policy::parse(p)?,
            None => None,
        };
        let clip_norm = opts
            .get("clip_norm")
            .and_then(Value::as_f64)
            .map(|v| v as f32);
        let resident = Resident::from_json(opts.get("resident").unwrap_or(&Value::Null))?;
        let mut t = Self::new(
            graph,
            TrainerConfig {
                wrt,
                init: init.into_iter().collect(),
                device,
                optimizer,
                policy,
                clip_norm,
                resident,
            },
        )?;
        t.package = Some(saver);
        Ok(t)
    }

    /// Write what a package trainer trained at `path`: a LoRA trainer's
    /// factors an adapter package over the package (`name` its name), an
    /// SAE a package of its own, else the package with the trained values.
    pub fn save_package(&mut self, path: &str, name: Option<&str>) -> ApiResult<()> {
        if self.package.is_none() {
            return Err(ApiError::type_(
                "Trainer.save_package: only a trainer from a package saves one",
            ));
        }
        let names = self.names.clone();
        let trained: HashMap<String, Vec<f32>> = self
            .params()
            .into_iter()
            .filter(|(n, _)| names.contains(n))
            .collect();
        let saver = self.package.as_ref().expect("checked above");
        saver
            .save(
                &trained,
                std::path::Path::new(path),
                name.unwrap_or("trained"),
            )
            .map_err(|e| ApiError::runtime(format!("Trainer.save_package: {e:#}")))
    }
}

/// `name → (node, static dims)` for every `Param`.
fn param_nodes(graph: &Graph) -> HashMap<String, (NodeId, Vec<usize>)> {
    let mut out = HashMap::new();
    for (index, node) in graph.nodes().iter().enumerate() {
        if let Op::Param { name } = &node.op {
            let dims = node
                .shape
                .dims()
                .iter()
                .map(|d| match d {
                    Dim::Static(n) => *n,
                    Dim::Dynamic(_) => 0,
                })
                .collect();
            out.insert(name.clone(), (NodeId(index as u32), dims));
        }
    }
    out
}

impl Trainer {
    /// Build a trainer; the forward graph is consumed.
    pub fn new(forward: Graph, cfg: TrainerConfig) -> ApiResult<Self> {
        let device = super::session::device(&cfg.device)?;
        if cfg.wrt.is_empty() {
            return Err(ApiError::type_(
                "Trainer: `wrt` must name at least one parameter to train",
            ));
        }
        if let Some(p) = cfg.policy.as_ref() {
            super::policy::check_supported(device, p)?;
        }
        let spec = optimizer(None, &cfg.optimizer)?;
        if forward.outputs.len() != 1 {
            return Err(ApiError::type_(format!(
                "Trainer: the forward graph must have exactly one output (the scalar loss), got {}",
                forward.outputs.len()
            )));
        }
        let nodes = param_nodes(&forward);
        let mut wrt = Vec::with_capacity(cfg.wrt.len());
        let mut shapes = Vec::with_capacity(cfg.wrt.len());
        for name in &cfg.wrt {
            match nodes.get(name) {
                Some((node, dims)) => {
                    wrt.push(*node);
                    shapes.push(dims.clone());
                }
                None => {
                    let mut known: Vec<&str> = nodes.keys().map(String::as_str).collect();
                    known.sort();
                    return Err(ApiError::type_(format!(
                        "Trainer: '{name}' is not a Param in this graph — it has [{}]",
                        known.join(", ")
                    )));
                }
            }
        }
        if cfg.resident != Resident::Off {
            return Self::new_resident(forward, cfg, spec, device, wrt, shapes);
        }

        // A `wrt` the loss does not reach, or an op with no VJP rule, is a
        // panic inside rlx — reported rather than aborting the process.
        let backward = catch("Trainer: differentiating the loss", || {
            rlx_opt::autodiff::grad_with_loss(&forward, &wrt)
        })?;
        let session = rlx_runtime::Session::new(device);
        let mut compiled = match cfg.policy.clone() {
            Some(p) => {
                let opts = rlx_runtime::CompileOptions::new().policy(p);
                catch("Trainer: compiling", || {
                    session.compile_with(backward, &opts)
                })?
            }
            None => catch("Trainer: compiling", || session.compile(backward))?,
        };

        let count = cfg.wrt.len();
        let mut values = vec![Vec::new(); count];
        let mut fixed = Vec::new();
        for (name, data) in cfg.init {
            match cfg.wrt.iter().position(|n| *n == name) {
                Some(i) => values[i] = data,
                None => {
                    compiled.set_param(&name, &data);
                    // Kept: a checkpoint without the frozen base weights
                    // cannot reproduce the model it came from.
                    fixed.push((name, data));
                }
            }
        }
        for (i, name) in cfg.wrt.iter().enumerate() {
            if values[i].is_empty() {
                return Err(ApiError::type_(format!(
                    "Trainer: no initial value for trainable parameter '{name}' — pass it in `init`"
                )));
            }
            let expected: usize = shapes[i].iter().product();
            if expected != 0 && values[i].len() != expected {
                return Err(ApiError::type_(format!(
                    "Trainer: '{name}' is declared {:?} ({expected} values) but `init` gave {}",
                    shapes[i],
                    values[i].len()
                )));
            }
        }
        Ok(Self {
            compiled: Some(compiled),
            resident: None,
            optimizer: spec.optimizer,
            optimizer_kind: spec.kind,
            names: cfg.wrt,
            shapes,
            values,
            frozen: vec![false; count],
            fixed,
            device,
            steps: 0,
            lr: spec.lr,
            clip_norm: cfg.clip_norm,
            last_grad_norm: 0.0,
            package: None,
        })
    }

    fn new_resident(
        forward: Graph,
        cfg: TrainerConfig,
        spec: OptimizerSpec,
        device: Device,
        wrt: Vec<NodeId>,
        shapes: Vec<Vec<usize>>,
    ) -> ApiResult<Self> {
        let label = rlx_runtime::device_label(device);
        if matches!(device, Device::Cpu) {
            return Err(ApiError::type_(format!(
                "Trainer(resident) on '{label}': the CPU backend has no device buffers, so the \
                 fused update feeds the parameters and moments as ordinary inputs and reads three \
                 buffers back per parameter — measured 3.3x slower than the host optimizer. Drop \
                 `resident` to use the host optimizer."
            )));
        }
        let verified = matches!(device, Device::Metal | Device::Gpu);
        if !verified && cfg.resident != Resident::Unverified {
            return Err(ApiError::type_(format!(
                "Trainer(resident) on '{label}': the fused update is not verified on this backend. \
                 Pass resident=\"unverified\" to try it, and compare a few steps against the host \
                 path before trusting the numbers."
            )));
        }
        let kind = spec.kind.clone();
        if kind != "adam" && kind != "adamw" {
            return Err(ApiError::type_(format!(
                "Trainer(resident): only 'adam' and 'adamw' are fused into the graph; '{kind}' \
                 runs on the host. Drop `resident` to use it."
            )));
        }
        if cfg.clip_norm.is_some() {
            return Err(ApiError::type_(
                "Trainer(resident): `clip_norm` / `clipNorm` needs the gradients on the host, and \
                 the fused path never brings them back. Drop one of the two.",
            ));
        }
        // Defaults come from the host optimizer this replaces: `AdamW::new`
        // defaults `weight_decay` to 0.01 (as PyTorch does), and hardcoding 0
        // here once trained a different objective on the two paths.
        let (d_beta1, d_beta2, d_eps, d_wd) = if kind == "adam" {
            let d = Adam::new(spec.lr);
            (d.beta1, d.beta2, d.eps, d.weight_decay)
        } else {
            let d = AdamW::new(spec.lr);
            (d.beta1, d.beta2, d.eps, d.weight_decay)
        };
        let o = Opts::new("optimizer", &spec.options)?;
        let weight_decay = o.f32("weight_decay", d_wd)?;
        // Classic Adam folds L2 into the gradient; the fused graph implements
        // AdamW's decoupled decay — different algorithms when decay is nonzero.
        if kind == "adam" && weight_decay != 0.0 {
            return Err(ApiError::type_(
                "Trainer(resident) with 'adam' and a nonzero weight_decay / weightDecay: classic \
                 Adam folds \
                 decay into the gradient, while the fused update applies AdamW's decoupled \
                 decay. Use 'adamw', or drop `resident`.",
            ));
        }
        let adam = rlx_runtime::train::AdamSpec {
            lr: spec.lr,
            beta1: o.f32("beta1", d_beta1)?,
            beta2: o.f32("beta2", d_beta2)?,
            eps: o.f32("eps", d_eps)?,
            weight_decay,
        };
        let slots: Vec<rlx_runtime::train::TrainableParam> = cfg
            .wrt
            .iter()
            .zip(wrt.iter())
            .map(|(name, node)| rlx_runtime::train::TrainableParam {
                name: name.clone(),
                node: *node,
            })
            .collect();
        let seed: HashMap<String, Vec<f32>> = cfg.init.iter().cloned().collect();
        let mut compile_opts = rlx_runtime::CompileOptions::new();
        if let Some(p) = cfg.policy.clone() {
            compile_opts = compile_opts.policy(p);
        }
        let trainer = catch("Trainer(resident)", || {
            ResidentTrainer::with_options(&forward, &slots, &seed, &adam, device, &compile_opts)
        })?
        .map_err(|e| ApiError::type_(format!("Trainer(resident): {e}")))?;
        let fixed = cfg
            .init
            .into_iter()
            .filter(|(name, _)| !cfg.wrt.contains(name))
            .collect();
        let count = cfg.wrt.len();
        Ok(Self {
            compiled: None,
            resident: Some(trainer),
            optimizer: spec.optimizer,
            optimizer_kind: spec.kind,
            names: cfg.wrt,
            shapes,
            values: Vec::new(),
            frozen: vec![false; count],
            fixed,
            device,
            steps: 0,
            lr: spec.lr,
            clip_norm: None,
            last_grad_norm: f32::NAN,
            package: None,
        })
    }

    /// `inputs` plus `d_output = [1]` unless the caller seeded it (loss scaling).
    fn with_seed<'a>(inputs: &[(&'a str, &'a [f32])]) -> Vec<(&'a str, &'a [f32])> {
        const ONE: &[f32] = &[1.0];
        let mut pairs = inputs.to_vec();
        if !pairs.iter().any(|(n, _)| *n == "d_output") {
            pairs.push(("d_output", ONE));
        }
        pairs
    }

    fn forward_backward(&mut self, inputs: &[(&str, &[f32])]) -> ApiResult<Vec<Vec<f32>>> {
        let pairs = Self::with_seed(inputs);
        let compiled = self
            .compiled
            .as_mut()
            .ok_or_else(|| ApiError::runtime("forward_backward: called on a resident trainer"))?;
        for (name, data) in self.names.iter().zip(&self.values) {
            compiled.set_param(name, data);
        }
        let outputs = catch("Trainer step", || compiled.run(&pairs))?;
        if outputs.len() != self.names.len() + 1 {
            return Err(ApiError::runtime(format!(
                "Trainer: expected a loss and {} gradients, got {} outputs",
                self.names.len(),
                outputs.len()
            )));
        }
        Ok(outputs)
    }

    fn resident_step(&mut self, inputs: &[(&str, &[f32])], evaluate_only: bool) -> ApiResult<f32> {
        let pairs = Self::with_seed(inputs);
        let trainer = self
            .resident
            .as_mut()
            .ok_or_else(|| ApiError::runtime("resident_step: not a resident trainer"))?;
        let outputs = catch("Trainer step (resident)", || {
            if evaluate_only {
                trainer.forward_only(&pairs)
            } else {
                trainer.step(&pairs)
            }
        })?;
        if !evaluate_only {
            self.steps = trainer.steps();
        }
        Ok(outputs
            .first()
            .and_then(|o| o.first().copied())
            .unwrap_or(f32::NAN))
    }

    /// One optimizer update on `inputs` → the loss.
    pub fn step(&mut self, inputs: &[(&str, &[f32])]) -> ApiResult<f32> {
        if self.resident.is_some() {
            return self.resident_step(inputs, false);
        }
        let outputs = self.forward_backward(inputs)?;
        let loss = outputs[0].first().copied().unwrap_or(f32::NAN);
        self.apply_update(&outputs)?;
        Ok(loss)
    }

    /// Loss only, no update — for a validation split.
    pub fn evaluate(&mut self, inputs: &[(&str, &[f32])]) -> ApiResult<f32> {
        if self.resident.is_some() {
            return self.resident_step(inputs, true);
        }
        let outputs = self.forward_backward(inputs)?;
        Ok(outputs[0].first().copied().unwrap_or(f32::NAN))
    }

    /// Clip, then hand the unfrozen parameters to the optimizer.
    ///
    /// The clip norm spans *every* trainable gradient, frozen or not: it is a
    /// property of the step, and leaving frozen tensors out of it would scale
    /// differently from an unfrozen run of the same recipe.
    fn apply_update(&mut self, outputs: &[Vec<f32>]) -> ApiResult<()> {
        let grads: Vec<&[f32]> = outputs[1..].iter().map(Vec::as_slice).collect();
        let norm_sq: f64 = grads
            .iter()
            .map(|g| {
                let n = rlx_optim::l2_norm(g) as f64;
                n * n
            })
            .sum();
        self.last_grad_norm = norm_sq.sqrt() as f32;
        let scale = match self.clip_norm {
            Some(max) => rlx_optim::global_grad_clip_scale(grads.iter().copied(), max),
            None => 1.0,
        };
        // Scaling allocates only when clipping actually fires.
        let clipped: Option<Vec<Vec<f32>>> = (scale < 1.0).then(|| {
            grads
                .iter()
                .map(|g| g.iter().map(|x| x * scale).collect())
                .collect()
        });
        let mut items: Vec<OptItem<'_>> = Vec::with_capacity(self.names.len());
        for (i, ((name, shape), param)) in self
            .names
            .iter()
            .zip(self.shapes.iter())
            .zip(self.values.iter_mut())
            .enumerate()
        {
            if self.frozen[i] {
                continue;
            }
            items.push(OptItem {
                name: name.as_str(),
                shape: shape.as_slice(),
                param: param.as_mut_slice(),
                grad: match &clipped {
                    Some(scaled) => scaled[i].as_slice(),
                    None => outputs[i + 1].as_slice(),
                },
            });
        }
        let optimizer = &mut self.optimizer;
        catch("optimizer step", || {
            optimizer.step_batch(&mut items);
            optimizer.end_iteration();
        })?;
        self.steps += 1;
        Ok(())
    }

    /// Freeze (or unfreeze) `names`, or every trainable param when `None` →
    /// how many are frozen now. The gradient is still computed; leave a param
    /// out of `wrt` to skip it.
    pub fn set_frozen(&mut self, names: Option<&[String]>, freeze: bool) -> ApiResult<usize> {
        let targets: Vec<String> = match names {
            Some(n) => n.to_vec(),
            None => self.names.clone(),
        };
        for name in &targets {
            let Some(i) = self.names.iter().position(|n| n == name) else {
                return Err(ApiError::type_(format!(
                    "freeze('{name}'): not a trainable parameter — this trainer has [{}]",
                    self.names.join(", ")
                )));
            };
            self.frozen[i] = freeze;
            // The resident path gates the update with a scalar input per param,
            // so freezing costs one f32 rather than a recompile.
            if let Some(t) = self.resident.as_mut() {
                t.set_frozen(name, freeze);
            }
        }
        Ok(self.frozen.iter().filter(|f| **f).count())
    }

    pub fn frozen(&self) -> Vec<String> {
        self.names
            .iter()
            .zip(&self.frozen)
            .filter(|(_, f)| **f)
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Every parameter that receives a gradient, frozen or not.
    pub fn trainable(&self) -> &[String] {
        &self.names
    }

    /// A trainable parameter's declared dims (a dynamic axis reads `0`).
    pub fn param_shape(&self, name: &str) -> Option<&[usize]> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.shapes[i].as_slice())
    }

    /// Global gradient norm of the last step, before clipping. `None` on the
    /// resident path: the gradients never leave the device, and a stale or
    /// invented number would be worse than none.
    pub fn grad_norm(&self) -> Option<f32> {
        self.resident.is_none().then_some(self.last_grad_norm)
    }

    /// Whether the update actually runs on-device.
    pub fn is_resident(&self) -> bool {
        self.resident
            .as_ref()
            .is_some_and(ResidentTrainer::is_resident)
    }

    /// Every parameter, trainable and fixed — feed it straight to an inference
    /// graph's `set_params`. (Returning only the trainable set once left a
    /// LoRA run's base weights at zero at evaluation.)
    pub fn params(&mut self) -> Vec<(String, Vec<f32>)> {
        let mut out: Vec<(String, Vec<f32>)> = match self.resident.as_mut() {
            Some(t) => {
                let mut v: Vec<(String, Vec<f32>)> = t.params().into_iter().collect();
                v.sort_by(|a, b| a.0.cmp(&b.0));
                v
            }
            None => self
                .names
                .iter()
                .cloned()
                .zip(self.values.iter().cloned())
                .collect(),
        };
        out.extend(self.fixed.iter().cloned());
        out
    }

    /// Overwrite weights — resuming from a checkpoint → how many were
    /// trainable. Non-trainable names update the fixed set.
    pub fn set_params(&mut self, incoming: Vec<(String, Vec<f32>)>) -> ApiResult<usize> {
        let mut applied = 0usize;
        if self.resident.is_some() {
            let mut map: HashMap<String, Vec<f32>> = HashMap::new();
            for (name, data) in incoming {
                if self.names.contains(&name) {
                    applied += 1;
                } else {
                    self.set_fixed(&name, data.clone());
                }
                map.insert(name, data);
            }
            if let Some(t) = self.resident.as_mut() {
                t.set_params(&map);
            }
            return Ok(applied);
        }
        for (name, data) in incoming {
            match self.names.iter().position(|n| *n == name) {
                Some(i) => {
                    if data.len() != self.values[i].len() {
                        return Err(ApiError::type_(format!(
                            "Trainer.set_params('{name}'): expected {} values, got {}",
                            self.values[i].len(),
                            data.len()
                        )));
                    }
                    self.values[i] = data;
                    applied += 1;
                }
                None => {
                    if let Some(c) = self.compiled.as_mut() {
                        c.set_param(&name, &data);
                    }
                    self.set_fixed(&name, data);
                }
            }
        }
        Ok(applied)
    }

    fn set_fixed(&mut self, name: &str, data: Vec<f32>) {
        match self.fixed.iter_mut().find(|(n, _)| n == name) {
            Some(existing) => existing.1 = data,
            None => self.fixed.push((name.to_string(), data)),
        }
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        match self.resident.as_mut() {
            Some(t) => t.set_lr(lr),
            None => self.optimizer.set_lr(lr),
        }
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// The algorithm, lowercased (`"adamw"`).
    pub fn optimizer_kind(&self) -> &str {
        &self.optimizer_kind
    }

    pub fn steps(&self) -> u64 {
        match self.resident.as_ref() {
            Some(t) => t.steps(),
            None => self.steps,
        }
    }

    pub fn device_label(&self) -> &'static str {
        rlx_runtime::device_label(self.device)
    }

    /// `""`, `" update=on-device"` or `" update=fused-host"` — for `repr`.
    pub fn update_label(&self) -> &'static str {
        match self.resident.as_ref() {
            Some(t) if t.is_resident() => " update=on-device",
            Some(_) => " update=fused-host",
            None => "",
        }
    }
}

/// The running summary of a `run` loop, the same in every binding:
/// `{steps, stopped, first_loss, last_loss, mean_loss, steps_per_second, grad_norm}`.
pub struct RunStats {
    began: std::time::Instant,
    first: Option<f32>,
    last: f32,
    sum: f64,
    pub done: usize,
    pub stopped: bool,
}

impl Default for RunStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunStats {
    pub fn new() -> Self {
        Self {
            began: std::time::Instant::now(),
            first: None,
            last: f32::NAN,
            sum: 0.0,
            done: 0,
            stopped: false,
        }
    }

    pub fn record(&mut self, loss: f32) {
        self.first.get_or_insert(loss);
        self.last = loss;
        self.sum += loss as f64;
        self.done += 1;
    }

    /// Whether `on_step` is due after this step (every `every`, and the last).
    pub fn report_due(&self, every: usize, i: usize, steps: usize) -> bool {
        self.done.is_multiple_of(every.max(1)) || i + 1 == steps
    }

    pub fn summary(&self, grad_norm: Option<f32>) -> Value {
        let seconds = self.began.elapsed().as_secs_f64().max(1e-9);
        json!({
            "steps": self.done,
            "stopped": self.stopped,
            "first_loss": self.first.unwrap_or(f32::NAN),
            "last_loss": self.last,
            "mean_loss": if self.done > 0 { self.sum / self.done as f64 } else { f64::NAN },
            "steps_per_second": self.done as f64 / seconds,
            "grad_norm": grad_norm,
        })
    }
}

// ── checkpoints ─────────────────────────────────────────────

#[cfg(feature = "gguf-io")]
const CKPT_VERSION: i64 = 1;

#[cfg(feature = "gguf-io")]
impl Trainer {
    /// Write a GGUF checkpoint → how many tensors went in. Includes the
    /// optimizer's accumulators when the algorithm exposes them; when it does
    /// not, the header says so and `load` reports it, because resuming Adam with
    /// its moments reset is a different run.
    pub fn save(&mut self, path: &str) -> ApiResult<usize> {
        use rlx_gguf::{GgmlType, GgufWriter, MetaValue};

        let (values, buffers, opt_step, steps) = if let Some(t) = self.resident.as_mut() {
            let params = t.params();
            let mut buffers = Vec::new();
            for (name, m, v) in t.moments() {
                buffers.push((format!("m/{name}"), m));
                buffers.push((format!("v/{name}"), v));
            }
            let values: Vec<Vec<f32>> = self
                .names
                .iter()
                .map(|n| params.get(n).cloned().unwrap_or_default())
                .collect();
            let step = t.steps();
            (values, Some(buffers), step, step)
        } else {
            let state = self.optimizer.state_dict();
            let opt_step = state.as_ref().map(|s| s.step).unwrap_or(0);
            (
                self.values.clone(),
                state.map(|s| s.buffers),
                opt_step,
                self.steps,
            )
        };

        let mut w = GgufWriter::new();
        w.set_arch("rlx-checkpoint");
        w.set_meta("rlx.checkpoint.version", MetaValue::I64(CKPT_VERSION));
        w.set_meta("rlx.checkpoint.step", MetaValue::U64(steps));
        w.set_meta("rlx.checkpoint.lr", MetaValue::F32(self.lr));
        w.set_meta(
            "rlx.checkpoint.optimizer",
            MetaValue::String(self.optimizer_kind.clone()),
        );
        w.set_meta("rlx.checkpoint.optimizer_step", MetaValue::U64(opt_step));
        w.set_meta(
            "rlx.checkpoint.has_optimizer_state",
            MetaValue::Bool(buffers.is_some()),
        );
        if !self.names.is_empty() {
            w.set_meta(
                "rlx.checkpoint.trainable",
                MetaValue::Array(self.names.iter().cloned().map(MetaValue::String).collect()),
            );
        }
        // An empty GGUF array is not representable; absent means "none frozen".
        let frozen: Vec<MetaValue> = self.frozen().into_iter().map(MetaValue::String).collect();
        if !frozen.is_empty() {
            w.set_meta("rlx.checkpoint.frozen", MetaValue::Array(frozen));
        }

        let bytes =
            |data: &[f32]| -> Vec<u8> { data.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let err = |what: String| move |e: anyhow::Error| ApiError::io(&what, e.as_ref());
        let mut written = 0usize;
        for ((name, shape), data) in self.names.iter().zip(&self.shapes).zip(&values) {
            // A dynamic axis records as 0, which GGUF cannot hold; the flat
            // length is all `load` needs.
            let dims = if shape.iter().all(|d| *d > 0) {
                shape.clone()
            } else {
                vec![data.len()]
            };
            w.add_tensor_bytes(format!("param/{name}"), dims, GgmlType::F32, bytes(data))
                .map_err(err(format!("save: param '{name}'")))?;
            written += 1;
        }
        for (name, data) in &self.fixed {
            w.add_tensor_bytes(
                format!("fixed/{name}"),
                vec![data.len()],
                GgmlType::F32,
                bytes(data),
            )
            .map_err(err(format!("save: fixed '{name}'")))?;
            written += 1;
        }
        if let Some(state) = &buffers {
            for (key, data) in state {
                w.add_tensor_bytes(
                    format!("opt/{key}"),
                    vec![data.len()],
                    GgmlType::F32,
                    bytes(data),
                )
                .map_err(err(format!("save: optimizer '{key}'")))?;
                written += 1;
            }
        }
        w.write_to_path(path)
            .map_err(err(format!("save('{path}')")))?;
        Ok(written)
    }

    /// Restore a checkpoint → `{step, restored, optimizer_restored,
    /// had_optimizer_state, unknown}`. Refuses one written by a different
    /// optimizer, or whose shapes do not match this model.
    pub fn load(&mut self, path: &str) -> ApiResult<Value> {
        use rlx_gguf::{GgufFile, MetaValue};

        let file = GgufFile::from_path_mmap(path)
            .map_err(|e| ApiError::io(&format!("load('{path}')"), e.as_ref()))?;
        let meta = &file.metadata;
        let saved_kind = meta
            .get("rlx.checkpoint.optimizer")
            .and_then(MetaValue::as_str)
            .unwrap_or_default();
        if !saved_kind.is_empty() && saved_kind != self.optimizer_kind {
            return Err(ApiError::type_(format!(
                "load('{path}'): checkpoint was written by '{saved_kind}' but this trainer uses \
                 '{}'. The accumulators mean different things, so loading them would train \
                 quietly wrong.",
                self.optimizer_kind
            )));
        }
        let step = meta
            .get("rlx.checkpoint.step")
            .and_then(MetaValue::as_u64)
            .unwrap_or(0);
        let had_state = matches!(
            meta.get("rlx.checkpoint.has_optimizer_state"),
            Some(MetaValue::Bool(true))
        );
        let frozen_names: Vec<String> = meta
            .get("rlx.checkpoint.frozen")
            .and_then(MetaValue::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let (mut params, mut fixed, mut opt_buffers) = (Vec::new(), Vec::new(), Vec::new());
        let mut names: Vec<String> = file.tensors.keys().cloned().collect();
        names.sort();
        for name in names {
            let (data, _) = file
                .dequant_f32(&name)
                .map_err(|e| ApiError::io(&format!("load: tensor '{name}'"), e.as_ref()))?;
            if let Some(rest) = name.strip_prefix("param/") {
                params.push((rest.to_string(), data));
            } else if let Some(rest) = name.strip_prefix("fixed/") {
                fixed.push((rest.to_string(), data));
            } else if let Some(rest) = name.strip_prefix("opt/") {
                opt_buffers.push((rest.to_string(), data));
            }
        }

        let mut restored = 0usize;
        let mut unknown: Vec<String> = Vec::new();
        // On the resident path `values` is empty (the weights live on the
        // device), so the expected size comes from the declared shape.
        let mut restored_params: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, data) in params {
            let Some(i) = self.names.iter().position(|n| *n == name) else {
                unknown.push(name);
                continue;
            };
            let want = match self.values.get(i) {
                Some(v) => v.len(),
                None => self.shapes[i].iter().product(),
            };
            if want != 0 && want != data.len() {
                return Err(ApiError::type_(format!(
                    "load: '{name}' has {} values in the checkpoint but {want} in this model — \
                     the graphs do not match",
                    data.len()
                )));
            }
            if let Some(v) = self.values.get_mut(i) {
                *v = data.clone();
            }
            restored_params.insert(name, data);
            restored += 1;
        }
        for (name, data) in fixed {
            if let Some(c) = self.compiled.as_mut() {
                c.set_param(&name, &data);
            }
            self.set_fixed(&name, data);
            restored += 1;
        }

        let optimizer_restored = if let Some(t) = self.resident.as_mut() {
            // Group `m/<name>` + `v/<name>` back into per-parameter triples.
            let moments: Vec<(String, Vec<f32>, Vec<f32>)> = self
                .names
                .iter()
                .filter_map(|name| {
                    let find = |k: String| {
                        opt_buffers
                            .iter()
                            .find(|(key, _)| *key == k)
                            .map(|(_, v)| v.clone())
                    };
                    Some((
                        name.clone(),
                        find(format!("m/{name}"))?,
                        find(format!("v/{name}"))?,
                    ))
                })
                .collect();
            t.set_params(&restored_params);
            !moments.is_empty() && t.restore_moments(&moments, step)
        } else if opt_buffers.is_empty() {
            false
        } else {
            let opt_step = meta
                .get("rlx.checkpoint.optimizer_step")
                .and_then(MetaValue::as_u64)
                .unwrap_or(step);
            self.optimizer.load_state_dict(&rlx_optim::OptimizerState {
                step: opt_step,
                buffers: opt_buffers,
            })
        };
        self.steps = step;
        if self.resident.is_none()
            && let Some(lr) = meta.get("rlx.checkpoint.lr").and_then(|v| match v {
                MetaValue::F32(x) => Some(*x),
                MetaValue::F64(x) => Some(*x as f32),
                _ => None,
            })
        {
            self.set_lr(lr);
        }
        // The frozen set comes back too, so a resumed run continues the recipe.
        for i in 0..self.names.len() {
            let held = frozen_names.contains(&self.names[i]);
            self.frozen[i] = held;
            if let Some(t) = self.resident.as_mut() {
                t.set_frozen(&self.names[i], held);
            }
        }
        Ok(json!({
            "step": step,
            "restored": restored,
            "optimizer_restored": optimizer_restored,
            "had_optimizer_state": had_state,
            "unknown": unknown,
        }))
    }
}
