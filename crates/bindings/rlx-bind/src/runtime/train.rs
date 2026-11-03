// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Training: `Optimizer` (rlx-optim over caller-owned arrays) and `Trainer`
//! (differentiate once, compile once, keep the params, step).
//!
//! The forward graph's first output is the scalar loss — `grad` seeds
//! `d_output` with `1`, which only means "differentiate the loss" when the
//! loss is what the graph ends on. Any further outputs (an activation, a
//! metric) are computed alongside and handed to [`Trainer::step_with`]'s
//! update by name; they are not differentiated.
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

/// The ranks of a data-parallel run ([`Trainer::set_data_parallel`]).
pub use rlx_driver::{ProcessGroup, ReduceKind, Transport};
use rlx_ir::{Dim, Graph, NodeId, Op};
/// What [`Trainer::step_with`]'s update receives, one per param.
pub use rlx_optim::OptItem;
/// An optimizer's checkpointable state (moments keyed `slot/param`).
pub use rlx_optim::OptimizerState;
use rlx_optim::{
    Adafactor, Adam, AdamW, Lamb, Lion, Mars, Muon, MuonAdamW, NAdamW, Optimizer as OptimizerTrait,
    QHAdamW, RAdam, Sgd, Soap, Sophia,
};
/// Learning-rate schedules ([`TrainingOptions::schedule`]).
pub use rlx_optim::{LrSchedule, ScheduleKind};
use rlx_runtime::train::{AdamSpec, FusedRule, MuonSpec, ResidentTrainer};
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
    "muon_adamw",
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
/// `weight_decay`, `momentum`, `nesterov`, `f32_math`, `gamma`, `rho`,
/// `muon_scale`. `muon_adamw` is Muon on the 2-D matrices and AdamW on every
/// other param, one `lr` for both (Muon's scaled by `muon_scale`, default 0.2).
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
        "muon_adamw" => {
            let mut x = MuonAdamW::new(lr);
            x.muon_scale = num("muon_scale", x.muon_scale)?;
            if !(x.muon_scale.is_finite() && x.muon_scale > 0.0) {
                return Err(ApiError::range(format!(
                    "optimizer 'muon_adamw': muon_scale must be positive, got {}",
                    x.muon_scale
                )));
            }
            x.muon.lr = lr * x.muon_scale;
            x.muon.momentum = num("momentum", x.muon.momentum)?;
            x.muon.nesterov = o.bool("nesterov", x.muon.nesterov)?;
            x.adamw.beta1 = num("beta1", x.adamw.beta1)?;
            x.adamw.beta2 = num("beta2", x.adamw.beta2)?;
            x.adamw.eps = num("eps", x.adamw.eps)?;
            x.adamw.f32_math = o.bool("f32_math", x.adamw.f32_math)?;
            // After muon_scale: Muon's decay is divided by it.
            Box::new(x.with_weight_decay(num("weight_decay", 0.0)?))
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
    /// Schedule, gradient accumulation, loss scaling.
    pub training: TrainingOptions,
}

/// What a step does around the optimizer, as PyTorch training sets it up: a
/// learning-rate schedule (`torch.optim.lr_scheduler`), gradient accumulation
/// over micro-batches, and loss scaling for low-precision gradients
/// (`torch.amp.GradScaler`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainingOptions {
    /// The factor on the base learning rate at each update.
    pub schedule: Option<LrSchedule>,
    /// Micro-batches per update: `step` adds up their gradients and updates
    /// on every `accumulate`-th call. 0 or 1: every call.
    pub accumulate: usize,
    pub loss_scale: Option<LossScale>,
}

/// Loss scaling: the backward pass starts from `scale` instead of 1, the
/// gradients are divided by it, and an update whose gradients are not finite
/// is skipped. Dynamic scaling halves the scale on a skip and doubles it after
/// `interval` good updates in a row, as `GradScaler` does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LossScale {
    pub init: f32,
    pub dynamic: bool,
    pub growth: f32,
    pub backoff: f32,
    pub interval: u64,
}

impl LossScale {
    /// `GradScaler`'s defaults: 65536, ×2 after 2000 good updates, ×0.5 on a skip.
    pub fn dynamic() -> Self {
        Self {
            init: 65536.0,
            dynamic: true,
            growth: 2.0,
            backoff: 0.5,
            interval: 2000,
        }
    }
}

impl TrainingOptions {
    /// `schedule`, `accumulate` and `loss_scale` from an options object (the
    /// other options are left for the caller to read).
    ///
    /// `schedule`: `{kind: "constant" | "linear" | "cosine" | "step" |
    /// "multistep" | "exponential" | "polynomial" | "onecycle", total_steps,
    /// warmup_steps, warmup_start, min_factor, end_factor, step_size, gamma,
    /// milestones, power, pct_start, div_factor, final_div_factor}`.
    /// `loss_scale`: `"dynamic"`, a fixed number, or `{init, dynamic, growth,
    /// backoff, interval}`.
    pub fn read(o: &Opts) -> ApiResult<Self> {
        let schedule = o.get("schedule").map(parse_schedule).transpose()?;
        let accumulate = o.usize("accumulate", 1)?;
        let loss_scale = match o.get("loss_scale") {
            None => None,
            Some(Value::String(s)) if s == "dynamic" => Some(LossScale::dynamic()),
            Some(Value::Number(n)) => Some(LossScale {
                init: n.as_f64().unwrap_or(1.0) as f32,
                dynamic: false,
                ..LossScale::dynamic()
            }),
            Some(v @ Value::Object(_)) => {
                let s = Opts::new("loss_scale", v)?;
                let d = LossScale::dynamic();
                let ls = LossScale {
                    init: s.f32("init", d.init)?,
                    dynamic: s.bool("dynamic", d.dynamic)?,
                    growth: s.f32("growth", d.growth)?,
                    backoff: s.f32("backoff", d.backoff)?,
                    interval: s.u64("interval")?.unwrap_or(d.interval),
                };
                s.finish()?;
                Some(ls)
            }
            Some(other) => {
                return Err(ApiError::type_(format!(
                    "loss_scale is \"dynamic\", a number or an object, got {other}"
                )));
            }
        };
        if let Some(ls) = &loss_scale
            && !(ls.init.is_finite()
                && ls.init > 0.0
                && ls.growth >= 1.0
                && ls.backoff > 0.0
                && ls.backoff <= 1.0)
        {
            return Err(ApiError::range(format!(
                "loss_scale: {ls:?} is not a usable scale"
            )));
        }
        Ok(Self {
            schedule,
            accumulate,
            loss_scale,
        })
    }

    /// [`Self::read`] over an object of just these options.
    pub fn from_json(v: &Value) -> ApiResult<Self> {
        let o = Opts::new("training", v)?;
        let t = Self::read(&o)?;
        o.finish()?;
        Ok(t)
    }

    fn check(&self, resident: bool) -> ApiResult<()> {
        if resident && (self.accumulate > 1 || self.loss_scale.is_some()) {
            return Err(ApiError::type_(
                "Trainer(resident): accumulate and loss_scale need the gradients on the host; \
                 build the trainer without `resident`",
            ));
        }
        Ok(())
    }
}

fn parse_schedule(v: &Value) -> ApiResult<LrSchedule> {
    let o = Opts::new("schedule", v)?;
    let total = || -> ApiResult<u64> {
        o.u64("total_steps")?
            .ok_or_else(|| ApiError::type_("schedule: this kind needs `total_steps`"))
    };
    let gamma = || -> ApiResult<f64> {
        o.f64("gamma")?
            .ok_or_else(|| ApiError::type_("schedule: this kind needs `gamma`"))
    };
    let kind = match o.str("kind")?.unwrap_or("constant") {
        "constant" => ScheduleKind::Constant,
        "linear" => ScheduleKind::Linear {
            total: total()?,
            end_factor: o.f64("end_factor")?.unwrap_or(0.0),
        },
        "cosine" => ScheduleKind::Cosine {
            total: total()?,
            min_factor: o.f64("min_factor")?.unwrap_or(0.0),
        },
        "step" => ScheduleKind::Step {
            step_size: o
                .u64("step_size")?
                .ok_or_else(|| ApiError::type_("schedule step: `step_size` is required"))?,
            gamma: gamma()?,
        },
        "multistep" => ScheduleKind::MultiStep {
            milestones: o
                .get("milestones")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .ok_or_else(|| {
                    ApiError::type_(
                        "schedule multistep: `milestones` (a list of updates) is required",
                    )
                })?,
            gamma: gamma()?,
        },
        "exponential" => ScheduleKind::Exponential { gamma: gamma()? },
        "polynomial" => ScheduleKind::Polynomial {
            total: total()?,
            power: o.f64("power")?.unwrap_or(1.0),
            end_factor: o.f64("end_factor")?.unwrap_or(0.0),
        },
        "onecycle" => ScheduleKind::OneCycle {
            total: total()?,
            pct_start: o.f64("pct_start")?.unwrap_or(0.3),
            div_factor: o.f64("div_factor")?.unwrap_or(25.0),
            final_div_factor: o.f64("final_div_factor")?.unwrap_or(1e4),
        },
        other => {
            return Err(ApiError::type_(format!(
                "schedule kind is constant, linear, cosine, step, multistep, exponential, \
                 polynomial or onecycle, got {other:?}"
            )));
        }
    };
    let s = LrSchedule::new(kind).with_warmup(
        o.u64("warmup_steps")?.unwrap_or(0),
        o.f64("warmup_start")?.unwrap_or(0.0),
    );
    o.finish()?;
    Ok(s)
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
    /// Names of the forward graph's outputs after the loss.
    extra: Vec<String>,
    training: TrainingOptions,
    /// The rate the schedule scales (`set_lr` changes it).
    base_lr: f32,
    /// The rate was set by a schedule or `set_lr`: `step_with` passes it on.
    lr_driven: bool,
    /// Gradient accumulation: the sums so far, how many micro-batches, their loss.
    acc: Vec<Vec<f32>>,
    acc_n: usize,
    acc_loss: f64,
    scale: f32,
    scale_good: u64,
    skipped: u64,
    /// Data-parallel: the ranks training replicas of this model on their own batches.
    group: Option<ProcessGroup>,
}

/// What [`Trainer::step_with`]'s update knows about its step besides the params.
#[derive(Clone, Copy, Debug)]
pub struct StepInfo<'a> {
    /// This step's index, counted from 1.
    pub update: u64,
    /// This step's training loss (computed before the update).
    pub loss: f32,
    /// The forward graph's outputs after the loss, by name (empty when it has
    /// only the loss).
    pub outputs: &'a [(String, Vec<f32>)],
    /// The learning rate for this update, when a schedule or `set_lr` sets
    /// it (`None`: the optimizer config's own rate throughout).
    pub lr: Option<f32>,
}

/// The names of `forward`'s outputs after the loss: each node's name, else
/// `output<i>`.
fn extra_output_names(forward: &Graph) -> Vec<String> {
    forward.outputs[1..]
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            forward
                .node(id)
                .name
                .clone()
                .unwrap_or_else(|| format!("output{}", i + 1))
        })
        .collect()
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
    /// trainer's `{device, policy, clip_norm, resident, schedule, accumulate,
    /// loss_scale}`.
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
        let training: serde_json::Map<String, Value> = ["schedule", "accumulate", "loss_scale"]
            .into_iter()
            .filter_map(|k| Some((k.to_string(), opts.get(k)?.clone())))
            .collect();
        let training = TrainingOptions::from_json(&Value::Object(training))?;
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
                training,
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
        if forward.outputs.is_empty() {
            return Err(ApiError::type_(
                "Trainer: the forward graph needs an output: its first is the scalar loss",
            ));
        }
        // `grad` seeds d_output with 1, which only means "differentiate the
        // loss" when the first output is one number.
        let loss_shape = &forward.node(forward.outputs[0]).shape;
        if loss_shape.num_elements() != Some(1) {
            return Err(ApiError::type_(format!(
                "Trainer: the forward graph's first output must be the scalar loss, got shape {:?} \
                 (further outputs are allowed after it)",
                loss_shape.dims()
            )));
        }
        let extra = extra_output_names(&forward);
        cfg.training.check(cfg.resident != Resident::Off)?;
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
            extra,
            base_lr: spec.lr,
            lr_driven: cfg.training.schedule.is_some(),
            scale: cfg.training.loss_scale.map_or(1.0, |l| l.init),
            training: cfg.training,
            acc: Vec::new(),
            acc_n: 0,
            acc_loss: 0.0,
            scale_good: 0,
            skipped: 0,
            group: None,
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
        if !FUSED_KINDS.contains(&kind.as_str()) {
            return Err(ApiError::type_(format!(
                "Trainer(resident): only {} are fused into the graph; '{kind}' runs on the host. \
                 Drop `resident` to use it.",
                FUSED_KINDS.join(", ")
            )));
        }
        if cfg.clip_norm.is_some() {
            return Err(ApiError::type_(
                "Trainer(resident): `clip_norm` / `clipNorm` needs the gradients on the host, and \
                 the fused path never brings them back. Drop one of the two.",
            ));
        }
        let o = Opts::new("optimizer", &spec.options)?;
        let rule = match kind.as_str() {
            "muon" => {
                let d = Muon::new(spec.lr);
                Some(FusedRule::Muon(MuonSpec {
                    lr: spec.lr,
                    momentum: o.f32("momentum", d.momentum)?,
                    nesterov: o.bool("nesterov", d.nesterov)?,
                    weight_decay: o.f32("weight_decay", d.weight_decay)?,
                    ns_steps: d.ns_steps,
                    ns_coeffs: d.ns_coeffs,
                    adamw: None,
                    muon_scale: 1.0,
                }))
            }
            "muon_adamw" => {
                let d = MuonAdamW::new(spec.lr);
                let scale = o.f32("muon_scale", d.muon_scale)?;
                let adamw = AdamSpec {
                    lr: spec.lr,
                    beta1: o.f32("beta1", d.adamw.beta1)?,
                    beta2: o.f32("beta2", d.adamw.beta2)?,
                    eps: o.f32("eps", d.adamw.eps)?,
                    weight_decay: o.f32("weight_decay", 0.0)?,
                };
                Some(FusedRule::Muon(MuonSpec {
                    lr: spec.lr * scale,
                    momentum: o.f32("momentum", d.muon.momentum)?,
                    nesterov: o.bool("nesterov", d.muon.nesterov)?,
                    weight_decay: 0.0,
                    ns_steps: d.muon.ns_steps,
                    ns_coeffs: d.muon.ns_coeffs,
                    adamw: Some(adamw),
                    muon_scale: scale,
                }))
            }
            _ => None,
        };
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
        let weight_decay = if rule.is_some() {
            0.0
        } else {
            o.f32("weight_decay", d_wd)?
        };
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
        let rule = match rule {
            Some(r) => r,
            None => FusedRule::Adam(AdamSpec {
                lr: spec.lr,
                beta1: o.f32("beta1", d_beta1)?,
                beta2: o.f32("beta2", d_beta2)?,
                eps: o.f32("eps", d_eps)?,
                weight_decay,
            }),
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
            ResidentTrainer::with_rule(&forward, &slots, &seed, &rule, device, &compile_opts)
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
            extra: extra_output_names(&forward),
            base_lr: spec.lr,
            lr_driven: cfg.training.schedule.is_some(),
            scale: 1.0,
            training: cfg.training,
            acc: Vec::new(),
            acc_n: 0,
            acc_loss: 0.0,
            scale_good: 0,
            skipped: 0,
            group: None,
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
        // Loss scaling starts the backward pass from the scale instead of 1.
        let seed = [self.scale];
        let mut pairs = inputs.to_vec();
        if !pairs.iter().any(|(n, _)| *n == "d_output") {
            pairs.push(("d_output", &seed));
        }
        let compiled = self
            .compiled
            .as_mut()
            .ok_or_else(|| ApiError::runtime("forward_backward: called on a resident trainer"))?;
        for (name, data) in self.names.iter().zip(&self.values) {
            compiled.set_param(name, data);
        }
        let outputs = catch("Trainer step", || compiled.run(&pairs))?;
        if outputs.len() != 1 + self.extra.len() + self.names.len() {
            return Err(ApiError::runtime(format!(
                "Trainer: expected a loss, {} further outputs and {} gradients, got {} outputs",
                self.extra.len(),
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
        if !evaluate_only && let Some(s) = &self.training.schedule {
            let lr = (self.base_lr as f64 * s.factor(trainer.steps())) as f32;
            self.lr = lr;
            trainer.set_lr(lr);
        }
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
        self.train_once(inputs, |items, optimizer, _| {
            catch("optimizer step", || {
                optimizer.step_batch(items);
                optimizer.end_iteration();
            })
        })
    }

    /// One step whose update the caller makes, on `inputs` → the loss.
    ///
    /// Forward, backward and clipping run as in [`Trainer::step`]; then
    /// `update` gets every unfrozen trainable param with its gradient, in
    /// [`Trainer::trainable`] order, and the step's [`StepInfo`] (its index,
    /// counted from 1, its loss and the graph's further outputs), and changes
    /// the params in place. The trainer's own optimizer
    /// neither runs nor changes, so a checkpoint holds its state as it was.
    ///
    /// When `update` fails, the params keep whatever it had written and the
    /// step is not counted. Host path only: a resident trainer's update is
    /// part of its compiled graph.
    pub fn step_with<F>(&mut self, inputs: &[(&str, &[f32])], update: F) -> ApiResult<f32>
    where
        F: FnOnce(&mut [OptItem<'_>], &StepInfo<'_>) -> ApiResult<()>,
    {
        if self.resident.is_some() {
            return Err(ApiError::type_(
                "Trainer.step_with: a resident trainer's update is fused into its graph; \
                 build the trainer without `resident` to make the update yourself",
            ));
        }
        self.train_once(inputs, |items, _, info| {
            catch("Trainer.step_with update", || update(items, info))?
        })
    }

    /// Forward and backward on `inputs`; then, when an update is due (every
    /// `accumulate` calls), unscale, skip on non-finite gradients under loss
    /// scaling, move the schedule, clip and hand the update to `f` → the loss
    /// of `inputs`.
    fn train_once<F>(&mut self, inputs: &[(&str, &[f32])], f: F) -> ApiResult<f32>
    where
        F: FnOnce(&mut [OptItem<'_>], &mut dyn OptimizerTrait, &StepInfo<'_>) -> ApiResult<()>,
    {
        let mut outputs = self.forward_backward(inputs)?;
        let loss = outputs[0].first().copied().unwrap_or(f32::NAN);
        let first = 1 + self.extra.len();
        let extra: Vec<(String, Vec<f32>)> = self
            .extra
            .iter()
            .cloned()
            .zip(outputs[1..first].iter().cloned())
            .collect();
        let mut grads = outputs.split_off(first);
        if self.scale != 1.0 {
            let inv = 1.0 / self.scale;
            for g in &mut grads {
                g.iter_mut().for_each(|x| *x *= inv);
            }
        }
        let k = self.training.accumulate.max(1);
        let mut step_loss = loss;
        if k > 1 {
            if self.acc.is_empty() {
                self.acc = grads.iter().map(|g| vec![0.0; g.len()]).collect();
            }
            let w = 1.0 / k as f32;
            for (a, g) in self.acc.iter_mut().zip(&grads) {
                a.iter_mut().zip(g).for_each(|(a, g)| *a += g * w);
            }
            self.acc_n += 1;
            self.acc_loss += loss as f64;
            if self.acc_n < k {
                return Ok(loss);
            }
            grads = std::mem::take(&mut self.acc);
            step_loss = (self.acc_loss / k as f64) as f32;
            self.acc_n = 0;
            self.acc_loss = 0.0;
        }
        if let Some(g) = &self.group {
            step_loss = all_reduce_mean(g, &mut grads, step_loss)?;
        }
        if let Some(ls) = self.training.loss_scale {
            let finite = grads.iter().all(|g| g.iter().all(|x| x.is_finite()));
            if ls.dynamic {
                if finite {
                    self.scale_good += 1;
                    if self.scale_good >= ls.interval {
                        self.scale *= ls.growth;
                        self.scale_good = 0;
                    }
                } else {
                    self.scale *= ls.backoff;
                    self.scale_good = 0;
                }
            }
            if !finite {
                // Skipped as GradScaler skips it: the update's index (and the
                // schedule) move on, the optimizer's own step count does not.
                self.skipped += 1;
                self.steps += 1;
                return Ok(loss);
            }
        }
        if let Some(s) = &self.training.schedule {
            let lr = (self.base_lr as f64 * s.factor(self.steps)) as f32;
            if lr != self.lr {
                self.lr = lr;
                self.optimizer.set_lr(lr);
            }
        }
        let lr = self.lr_driven.then_some(self.lr);
        self.update_params(&grads, |items, optimizer, step| {
            let info = StepInfo {
                update: step,
                loss: step_loss,
                outputs: &extra,
                lr,
            };
            f(items, optimizer, &info)
        })?;
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

    /// Clip, then hand the unfrozen parameters to `update` with the optimizer
    /// and the index of this step; count the step once `update` succeeds.
    ///
    /// The clip norm spans *every* trainable gradient, frozen or not: it is a
    /// property of the step, and leaving frozen tensors out of it would scale
    /// differently from an unfrozen run of the same recipe.
    fn update_params<F>(&mut self, grads: &[Vec<f32>], update: F) -> ApiResult<()>
    where
        F: FnOnce(&mut [OptItem<'_>], &mut dyn OptimizerTrait, u64) -> ApiResult<()>,
    {
        let grads: Vec<&[f32]> = grads.iter().map(Vec::as_slice).collect();
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
                    None => grads[i],
                },
            });
        }
        update(&mut items, self.optimizer.as_mut(), self.steps + 1)?;
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

    /// Set the base learning rate; under a schedule, the rate the schedule
    /// scales from the next update on.
    pub fn set_lr(&mut self, lr: f32) {
        self.base_lr = lr;
        self.lr_driven = true;
        if self.training.schedule.is_some() {
            return;
        }
        self.lr = lr;
        match self.resident.as_mut() {
            Some(t) => t.set_lr(lr),
            None => self.optimizer.set_lr(lr),
        }
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// The loss scale now, under loss scaling.
    pub fn loss_scale(&self) -> Option<f32> {
        self.training.loss_scale.map(|_| self.scale)
    }

    /// Updates skipped for non-finite gradients under loss scaling.
    pub fn skipped_updates(&self) -> u64 {
        self.skipped
    }

    /// Micro-batches waiting for the next update under gradient accumulation.
    pub fn pending_micro_batches(&self) -> usize {
        self.acc_n
    }

    /// Train data-parallel over `group`, as PyTorch's DDP does: every rank
    /// holds a replica, steps on its own batches, and each update uses the
    /// mean gradient (and reports the mean loss) over the ranks. Rank 0's
    /// params are broadcast now, so the replicas start equal; clipping, loss
    /// scaling and the schedule then act on the same numbers on every rank.
    /// Every rank must call this, and then step, the same number of times.
    pub fn set_data_parallel(&mut self, group: ProcessGroup) -> ApiResult<()> {
        if self.resident.is_some() {
            return Err(ApiError::type_(
                "Trainer.set_data_parallel: a resident trainer's gradients stay on the device; \
                 build the trainer without `resident`",
            ));
        }
        if self.acc_n > 0 {
            return Err(ApiError::type_(
                "Trainer.set_data_parallel: micro-batches are waiting; attach the group between updates",
            ));
        }
        for v in &mut self.values {
            group.broadcast(0, v).map_err(collective)?;
        }
        self.group = Some(group);
        Ok(())
    }

    pub fn process_group(&self) -> Option<&ProcessGroup> {
        self.group.as_ref()
    }

    /// Whether every rank holds the same params (a strided sample of each,
    /// compared by the min and max over the ranks). True without a group.
    pub fn replicas_agree(&self) -> ApiResult<bool> {
        let Some(g) = &self.group else {
            return Ok(true);
        };
        let mut lo: Vec<f32> = self
            .values
            .iter()
            .flat_map(|v| v.iter().step_by(v.len().div_ceil(256).max(1)).copied())
            .collect();
        let mut hi = lo.clone();
        g.all_reduce(&mut lo, ReduceKind::Min).map_err(collective)?;
        g.all_reduce(&mut hi, ReduceKind::Max).map_err(collective)?;
        Ok(lo.iter().zip(&hi).all(|(a, b)| a.to_bits() == b.to_bits()))
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

/// The optimizer kinds the resident path fuses into the graph.
pub const FUSED_KINDS: [&str; 4] = ["adam", "adamw", "muon", "muon_adamw"];

impl Trainer {
    /// Where param `name`'s state goes in a checkpoint, as the host optimizer
    /// of this kind names it: the first moment's (or momentum's) slot, and
    /// whether a second moment `v/` exists.
    fn state_slots(&self) -> HashMap<String, (&'static str, bool)> {
        self.names
            .iter()
            .zip(&self.shapes)
            .map(|(name, shape)| {
                let slots = match self.optimizer_kind.as_str() {
                    "muon" => ("m", false),
                    "muon_adamw" if MuonAdamW::is_matrix(shape) => ("muon_m", false),
                    _ => ("m", true),
                };
                (name.clone(), slots)
            })
            .collect()
    }
}

fn collective(e: rlx_driver::CollectiveError) -> ApiError {
    ApiError::runtime(format!("data-parallel collective: {e:?}"))
}

/// The mean of `grads` and `loss` over the ranks, in one all-reduce.
fn all_reduce_mean(g: &ProcessGroup, grads: &mut [Vec<f32>], loss: f32) -> ApiResult<f32> {
    if g.world_size() <= 1 {
        return Ok(loss);
    }
    let mut flat: Vec<f32> = Vec::with_capacity(grads.iter().map(Vec::len).sum::<usize>() + 1);
    for v in grads.iter() {
        flat.extend_from_slice(v);
    }
    flat.push(loss);
    g.all_reduce(&mut flat, ReduceKind::Mean)
        .map_err(collective)?;
    let mut at = 0;
    for v in grads.iter_mut() {
        let n = v.len();
        v.copy_from_slice(&flat[at..at + n]);
        at += n;
    }
    Ok(flat[at])
}

/// Evaluations during a run (`run`'s `eval_every`): the history, the best so
/// far, and early stopping after `patience` evaluations without improvement —
/// what a training loop needs to keep the best checkpoint and stop on time.
#[derive(Clone, Debug, Default)]
pub struct EvalTracker {
    pub evals: Vec<(u64, f32)>,
    pub best: Option<(u64, f32)>,
    /// Stop after this many evaluations in a row without a new best.
    pub patience: Option<usize>,
    /// A new best must beat the old by more than this.
    pub min_delta: f32,
    since_best: usize,
}

impl EvalTracker {
    pub fn new(patience: Option<usize>, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            ..Self::default()
        }
    }

    /// Record the evaluation loss after `step` → (a new best?, stop now?). A
    /// non-finite loss is never a best.
    pub fn record(&mut self, step: u64, loss: f32) -> (bool, bool) {
        self.evals.push((step, loss));
        let improved = loss.is_finite() && self.best.is_none_or(|(_, b)| loss < b - self.min_delta);
        if improved {
            self.best = Some((step, loss));
            self.since_best = 0;
        } else {
            self.since_best += 1;
        }
        let stop = self.patience.is_some_and(|p| self.since_best >= p);
        (improved, stop)
    }

    /// `{evals: [[step, loss], …], best_step, best_loss}` for a run summary.
    pub fn summary(&self) -> Value {
        json!({
            "evals": self.evals,
            "best_step": self.best.map(|b| b.0),
            "best_loss": self.best.map(|b| b.1),
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

        let slots = self.state_slots();
        let (values, buffers, opt_step, steps) = if let Some(t) = self.resident.as_mut() {
            let params = t.params();
            let mut buffers = Vec::new();
            for (name, m, v) in t.moments() {
                // The host optimizer's slots, so a checkpoint resumes on either path.
                let (m_slot, v_used) = slots.get(&name).copied().unwrap_or(("m", true));
                buffers.push((format!("{m_slot}/{name}"), m));
                if v_used {
                    buffers.push((format!("v/{name}"), v));
                }
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

        let slots = self.state_slots();
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
                    let (m_slot, v_used) = slots.get(name).copied().unwrap_or(("m", true));
                    let m = find(format!("{m_slot}/{name}"))?;
                    let v = if v_used {
                        find(format!("v/{name}"))?
                    } else {
                        vec![0.0; m.len()]
                    };
                    Some((name.clone(), m, v))
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
