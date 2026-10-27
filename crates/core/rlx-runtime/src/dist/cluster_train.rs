// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cluster training over a [`ProcessGroup`]: the [`TrainSpec`] loop with the
//! communication options a real cluster needs — gradient accumulation, a
//! bf16/f16 wire, compute/communication overlap, throughput-balanced work,
//! local SGD, ZeRO-1 optimizer-state sharding and checkpoint/resume.
//!
//! [`run_train_group`] is what a generic worker calls when the shipped spec
//! carries [`TrainOptions`]; [`super::run_train`] stays the options-free path.
//!
//! ## Protocol (every rank, in this order)
//!
//! 1. agree on the job — a fingerprint of the spec is all-gathered and must
//!    match, so a mismatched rank fails everywhere instead of deadlocking;
//! 2. calibrate — time a few micro-steps on zero data (when `balance`);
//! 3. plan — all-gather `(step time, lanes)` and compute the same
//!    [`TrainPlan`] on every rank: micro-steps per sync and a data slice per
//!    rank, both proportional to throughput;
//! 4. resume — rank 0 loads a checkpoint, if any, and broadcasts it;
//! 5. distribute — with `push_data`, rank 0 broadcasts init params and sends
//!    each rank its planned data slice;
//! 6. train — `syncs_per_epoch` lock-step syncs per epoch.
//!
//! The gradient of a micro-batch is its batch mean; each rank sends
//! `Σ batch·grad / S` (`S` = samples per sync, cluster-wide), so a Sum
//! all-reduce is the exact global per-sample mean whatever each rank's share.

use super::WeightCache;
use super::training::{Feed, TrainMetrics, TrainSpec, record_epoch, resolve_shard, training_lanes};
use crate::{Device, Session};
use rlx_driver::{ProcessGroup, ReduceKind};
use rlx_ir::{DType, Graph, Op};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

/// Wire tag for a pushed data slice (`TAG_SLICE_BASE + input_index`). Clear of
/// the legacy push tags (45.., 60..) and the spec/activation tags (40-42).
const TAG_SLICE_BASE: u32 = 80;

/// Element type a gradient or activation travels as. The discriminant is the
/// activation wire header byte — do not reorder.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum WireDtype {
    /// Full precision (4 B/element).
    #[default]
    F32 = 0,
    /// bfloat16 (2 B/element): f32's range, 8-bit mantissa. The safe 2×.
    Bf16 = 1,
    /// IEEE half (2 B/element): more mantissa, range to ±65504.
    F16 = 2,
    /// Choose per use (configuration only; a message always carries a concrete
    /// width). Gradients: bf16 with error feedback when the bucket is
    /// bandwidth-bound on the measured link (`bytes/β > α`), else f32 —
    /// decided by rank 0 and broadcast. Activations: f16 per message when every
    /// value is finite and within ±65504 (3 more mantissa bits than bf16), else
    /// bf16.
    Auto = 3,
}

impl std::str::FromStr for WireDtype {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Ok(Self::F32),
            "bf16" => Ok(Self::Bf16),
            "f16" | "fp16" | "half" => Ok(Self::F16),
            "auto" => Ok(Self::Auto),
            o => Err(format!(
                "unknown wire dtype {o:?} (f32 | bf16 | f16 | auto)"
            )),
        }
    }
}

/// Activation messages up to this size go f32 under [`WireDtype::Auto`].
pub const EXACT_ACTIVATION_BYTES: usize = 1 << 20;

impl WireDtype {
    fn dtype(self) -> DType {
        match self {
            Self::F32 | Self::Auto => DType::F32,
            Self::Bf16 => DType::BF16,
            Self::F16 => DType::F16,
        }
    }

    /// Bytes per element on the wire.
    pub fn bytes(self) -> usize {
        match self {
            Self::F32 | Self::Auto => 4,
            Self::Bf16 | Self::F16 => 2,
        }
    }

    /// `x` rounded to this width (nearest-even), back in f32.
    pub fn round(self, x: f32) -> f32 {
        match self {
            Self::F32 | Self::Auto => x,
            Self::Bf16 => half::bf16::from_f32(x).to_f32(),
            Self::F16 => half::f16::from_f32(x).to_f32(),
        }
    }

    /// The width an `Auto` activation message takes: f32 up to
    /// [`EXACT_ACTIVATION_BYTES`] (latency-bound, so halving it saves almost
    /// nothing, while rounding the hidden state can flip a near-tied greedy
    /// token — a Qwen3 2-stage split decoded differently from one node at
    /// token 16 with f16, identically with f32); above that, f16 when every
    /// value is finite and inside f16's range, else bf16.
    pub fn for_activation(self, data: &[f32]) -> Self {
        match self {
            Self::Auto if data.len() * 4 <= EXACT_ACTIVATION_BYTES => Self::F32,
            Self::Auto => {
                if data.iter().all(|v| v.is_finite() && v.abs() <= 65504.0) {
                    Self::F16
                } else {
                    Self::Bf16
                }
            }
            w => w,
        }
    }
}

/// Communication and optimizer options for [`run_train_group`]. Shipped inside
/// the [`TrainSpec`], so every rank applies the same ones.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct TrainOptions {
    /// Micro-batches the **slowest** rank runs per gradient sync (≥ 1). Larger
    /// = fewer syncs per epoch; the effective batch grows by the same factor.
    pub accum: usize,
    /// Gradient wire dtype. bf16/f16 halve the bytes; the reduce accumulates
    /// in f64 either way. `Auto` (the default) picks per measured link — bf16
    /// with error feedback when the bucket is bandwidth-bound, else f32 (see
    /// [`WireDtype`]); on the mm4/msi/amd LAN it matched f32's accuracy
    /// 25–40% faster.
    pub wire: WireDtype,
    /// With a 16-bit wire, carry each sync's rounding error into the next
    /// (error feedback), so the rounding averages out instead of biasing the
    /// trajectory. Costs one f32 buffer the size of the gradients.
    pub wire_feedback: bool,
    /// Overlap each sync's reduce with the next sync's compute. Gradients are
    /// then applied one sync late (1-step-stale SGD).
    pub overlap: bool,
    /// Give each rank work in proportion to its measured throughput (more
    /// micro-batches per sync and a larger data slice for faster ranks), so no
    /// rank idles at the sync. Off = every rank runs `accum` micro-batches.
    pub balance: bool,
    /// Local SGD: when > 0, each rank steps on its own gradients and the
    /// parameters are averaged every `local_steps` syncs (and at each epoch's
    /// end) instead of reducing gradients every sync. Cuts traffic ~that
    /// factor; trades some statistical efficiency.
    pub local_steps: usize,
    /// ZeRO-1: shard the momentum buffer across ranks — each rank keeps and
    /// updates only its slice (reduce-scatter → local update → all-gather).
    /// Same bytes on the wire; optimizer state memory ÷ world.
    pub zero: bool,
    /// Micro-batches timed per lane (after one warm-up; median) to calibrate
    /// `balance`.
    pub probe_steps: usize,
    /// Rank 0 writes params + momentum here after every epoch, and every rank
    /// resumes from it at start when it exists and matches the job.
    pub checkpoint_dir: Option<String>,
    /// Number format every lane computes in (`f32` when unset): `f64`,
    /// `f64x2`, `tf32`, `bf16`, `f16`, `f8e4m3`, `f8e5m2`, `f6`, `f4`, any
    /// `fNeXmY`, or `weights/activations` like `f4/f8` — see
    /// [`rlx_ir::NumberFormat`]. Weights are rounded as they are uploaded and
    /// every intermediate (forward and backward) after the op that makes it;
    /// the master weights, momentum and the gradient sum stay `f32`.
    pub number_format: Option<String>,
    /// Loss scale for narrow formats: the backward seed is multiplied by it
    /// and the gradients divided, so small gradients stay above the format's
    /// underflow. `None` = 1024 for formats with ≤ 5 exponent bits (f16, fp8,
    /// fp6, fp4), else 1. Halved automatically when a sync's gradient is not
    /// finite (that update is skipped).
    pub loss_scale: Option<f32>,
    /// Round explicit `Cast`s to f16 / bf16 in the graph instead of keeping
    /// the f32 value (see [`crate::HalfCasts`]). Off by default.
    pub half_casts: bool,
    /// The update rule (host side, f32; ZeRO shards its state too).
    pub optimizer: Optimizer,
    /// Scale the learning rate with the batch each update sees. The spec's
    /// rate is taken as tuned for one `batch`; an update here covers the
    /// cluster's samples per sync (per rank, under local SGD), which grows
    /// with lanes, balance and accumulation — fewer, larger steps that, at an
    /// unscaled rate, learn less per epoch.
    pub lr_scaling: LrScaling,
    /// Updates over which the rate ramps up linearly. `None` = 5% of the run
    /// when `lr_scaling` is on, none otherwise.
    pub warmup: Option<usize>,
}

/// How [`TrainOptions::lr_scaling`] grows the learning rate with the update's
/// batch `B` relative to the spec's `batch` `b`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LrScaling {
    /// The spec's rate as is.
    #[default]
    None,
    /// × √(B/b) — the safer rule for SGD with momentum and for Adam.
    Sqrt,
    /// × B/b — Goyal et al.'s rule; wants a warmup.
    Linear,
}

impl std::str::FromStr for LrScaling {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "off" => Ok(Self::None),
            "sqrt" => Ok(Self::Sqrt),
            "linear" => Ok(Self::Linear),
            o => Err(format!("lr scaling {o:?}: expected none, sqrt or linear")),
        }
    }
}

impl LrScaling {
    /// The factor for an update over `update` samples when the rate was tuned
    /// for `batch`.
    pub fn factor(self, update: f64, batch: f64) -> f32 {
        let r = (update / batch.max(1.0)).max(1.0);
        match self {
            Self::None => 1.0,
            Self::Sqrt => r.sqrt() as f32,
            Self::Linear => r as f32,
        }
    }
}

/// How [`run_train_group`] turns the reduced gradient into an update.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Optimizer {
    /// Momentum SGD (`TrainSpec::momentum`).
    #[default]
    Sgd,
    /// AdamW: Adam with bias correction and decoupled weight decay.
    #[serde(rename = "adamw")]
    AdamW {
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    },
}

impl Optimizer {
    /// AdamW with the usual defaults (0.9, 0.999, 1e-8, no decay).
    pub fn adamw() -> Self {
        Self::AdamW {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// State buffers per parameter element (SGD: velocity; AdamW: m, v).
    fn buffers(&self) -> usize {
        match self {
            Self::Sgd => 1,
            Self::AdamW { .. } => 2,
        }
    }
}

/// An optimizer's state over the elements a rank updates (all of them, or a
/// ZeRO slice): one buffer per moment, and the AdamW step count.
struct OptState {
    bufs: Vec<Vec<f32>>,
    step: u64,
}

impl OptState {
    /// Update `params[range]` from `grads[range]`; the state covers `range`.
    fn apply(
        &mut self,
        opt: Optimizer,
        momentum: f32,
        params: &mut [f32],
        grads: &[f32],
        range: std::ops::Range<usize>,
        lr: f32,
    ) {
        match opt {
            Optimizer::Sgd => sgd_range(params, &mut self.bufs[0], grads, range, momentum, lr),
            Optimizer::AdamW {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                self.step += 1;
                let t = self.step as i32;
                let (c1, c2) = (1.0 - beta1.powi(t), 1.0 - beta2.powi(t));
                let (m, v) = self.bufs.split_at_mut(1);
                let (m, v) = (&mut m[0], &mut v[0]);
                for (k, i) in range.enumerate() {
                    let g = grads[i];
                    m[k] = beta1 * m[k] + (1.0 - beta1) * g;
                    v[k] = beta2 * v[k] + (1.0 - beta2) * g * g;
                    let update = (m[k] / c1) / ((v[k] / c2).sqrt() + eps);
                    params[i] -= lr * (update + weight_decay * params[i]);
                }
            }
        }
    }
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            accum: 1,
            wire: WireDtype::Auto,
            wire_feedback: true,
            overlap: false,
            balance: false,
            local_steps: 0,
            zero: false,
            probe_steps: 10,
            checkpoint_dir: None,
            number_format: None,
            loss_scale: None,
            half_casts: false,
            optimizer: Optimizer::Sgd,
            lr_scaling: LrScaling::None,
            warmup: None,
        }
    }
}

impl TrainOptions {
    /// Reject combinations that are not defined. Depends only on the options,
    /// so every rank returns the same verdict.
    pub fn validate(&self) -> Result<(), String> {
        self.parsed_number_format()?;
        if self.accum == 0 {
            return Err("TrainOptions: accum must be ≥ 1".into());
        }
        if self.local_steps > 0 && (self.zero || self.overlap) {
            return Err(
                "TrainOptions: local_steps averages parameters; it does not combine with \
                 zero or overlap"
                    .into(),
            );
        }
        if self.zero && self.overlap {
            return Err("TrainOptions: zero and overlap do not combine".into());
        }
        if self.zero && matches!(self.wire, WireDtype::Bf16 | WireDtype::F16) {
            return Err("TrainOptions: zero's reduce-scatter carries f32 only".into());
        }
        Ok(())
    }
}

/// Most lanes one rank reports in the plan exchange.
pub const MAX_LANES: usize = 16;

impl TrainOptions {
    /// The parsed [`Self::number_format`].
    pub fn parsed_number_format(&self) -> Result<Option<rlx_ir::NumberFormat>, String> {
        self.number_format
            .as_deref()
            .map(|s| s.parse::<rlx_ir::NumberFormat>())
            .transpose()
            .map_err(|e| format!("TrainOptions.number_format: {e}"))
    }

    /// The loss scale to start from (see [`Self::loss_scale`]).
    pub fn initial_loss_scale(&self) -> f32 {
        if let Some(s) = self.loss_scale {
            return s;
        }
        let narrow = |f: Option<(rlx_ir::FloatFormat, rlx_ir::FormatScaling)>| {
            f.is_some_and(|(f, _)| f.exp_bits <= 5)
        };
        match self.parsed_number_format().ok().flatten() {
            Some(nf) if narrow(nf.activations()) || narrow(nf.weights()) => 1024.0,
            _ => 1.0,
        }
    }
}

/// One rank's share of the work.
#[derive(Clone, Debug, PartialEq)]
pub struct RankPlan {
    /// Micro-batches per sync for each of the rank's device lanes. Lanes run
    /// concurrently, each through its own count.
    pub lane_micro: Vec<usize>,
    /// Measured seconds per micro-batch per lane (1.0 when not calibrated).
    pub lane_s: Vec<f64>,
    /// First sample of the rank's slice, relative to the dataset start.
    pub data_start: usize,
    /// Samples in the rank's slice (`syncs_per_epoch · samples_per_sync`).
    pub data_len: usize,
}

impl RankPlan {
    /// Micro-batches per sync across the rank's lanes.
    pub fn micro_steps(&self) -> usize {
        self.lane_micro.iter().sum()
    }
    /// Samples this rank contributes per sync.
    pub fn samples_per_sync(&self, batch: usize) -> usize {
        self.micro_steps() * batch
    }
    /// Predicted compute seconds per sync (the busiest lane).
    pub fn sync_s(&self) -> f64 {
        self.lane_micro
            .iter()
            .zip(&self.lane_s)
            .map(|(&k, &t)| k as f64 * t)
            .fold(0.0, f64::max)
    }
}

/// The cluster's work split, identical on every rank.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainPlan {
    pub ranks: Vec<RankPlan>,
    /// Samples per sync across the cluster.
    pub samples_per_sync: usize,
    pub syncs_per_epoch: usize,
    /// Lane times were measured (`balance`), not placeholders.
    pub calibrated: bool,
}

/// Split `n_samples` across ranks. `lane_s[r]` holds rank r's seconds per
/// micro-batch for each of its lanes. With `balance`, the slowest lane in the
/// cluster runs `accum` micro-batches per sync and every other lane as many as
/// fit in the same time — so a GPU beside a CPU, or a fast node beside a slow
/// one, never waits; otherwise every lane runs `accum`. Each rank's data slice
/// is contiguous, in rank order, sized to its per-sync share.
/// Micro-batch gradients a rank buffers to sum them in batch order.
const ORDERED_BYTES: usize = 256 << 20;

/// A balanced lane slower than this × its rank's fastest lane is left out.
pub const SLOW_LANE: f64 = 8.0;

pub fn plan_training(
    lane_s: &[Vec<f64>],
    batch: usize,
    accum: usize,
    balance: bool,
    n_samples: usize,
) -> Result<TrainPlan, String> {
    if lane_s.is_empty() || lane_s.iter().any(Vec::is_empty) {
        return Err("plan_training: every rank needs at least one lane".into());
    }
    let accum = accum.max(1);
    // Balanced, a lane more than SLOW_LANE× slower than its own rank's fastest
    // lane sits the job out (no micro-batches) instead of setting the pace for
    // the whole cluster — a Core ML lane that recompiles on every parameter
    // upload measured ~10× its probe. A rank always keeps its fastest lane.
    let keep: Vec<Vec<bool>> = lane_s
        .iter()
        .map(|ts| {
            let fastest = ts.iter().copied().fold(f64::INFINITY, f64::min);
            ts.iter()
                .map(|&t| !balance || ts.len() == 1 || t <= SLOW_LANE * fastest)
                .collect()
        })
        .collect();
    let kept = || {
        lane_s
            .iter()
            .zip(&keep)
            .flat_map(|(ts, k)| ts.iter().zip(k).filter(|(_, k)| **k).map(|(t, _)| *t))
    };
    let slowest = kept().fold(0.0f64, f64::max);
    let share = |t: f64| {
        if balance && t > 0.0 && slowest > 0.0 {
            accum as f64 * slowest / t
        } else {
            accum as f64
        }
    };
    // A fast lane beside a slow one can ask for more than the dataset holds:
    // shrink every share by the same factor (keeping the balance) to fit.
    let wanted: f64 = kept().map(share).sum::<f64>() * batch as f64;
    let fit = if balance && wanted > n_samples as f64 {
        n_samples as f64 / wanted
    } else {
        1.0
    };
    let micro = |t: f64| {
        if balance {
            let k = share(t) * fit;
            // Round normally; floor when shrinking so the total still fits.
            (if fit < 1.0 { k.floor() } else { k.round() } as usize).max(1)
        } else {
            accum
        }
    };
    let lane_micro: Vec<Vec<usize>> = lane_s
        .iter()
        .zip(&keep)
        .map(|(ts, k)| {
            ts.iter()
                .zip(k)
                .map(|(&t, &k)| if k { micro(t) } else { 0 })
                .collect()
        })
        .collect();
    let per: Vec<usize> = lane_micro
        .iter()
        .map(|k| k.iter().sum::<usize>() * batch)
        .collect();
    let samples_per_sync: usize = per.iter().sum();
    let syncs_per_epoch = n_samples.checked_div(samples_per_sync).unwrap_or(0);
    if syncs_per_epoch == 0 {
        return Err(format!(
            "plan_training: {n_samples} samples cannot fill one sync of {samples_per_sync} \
             (lower accum or batch)"
        ));
    }
    let mut start = 0usize;
    let ranks = lane_micro
        .into_iter()
        .zip(lane_s)
        .zip(&per)
        .map(|((k, ts), &p)| {
            let len = syncs_per_epoch * p;
            let r = RankPlan {
                lane_micro: k,
                lane_s: ts.clone(),
                data_start: start,
                data_len: len,
            };
            start += len;
            r
        })
        .collect();
    Ok(TrainPlan {
        ranks,
        samples_per_sync,
        syncs_per_epoch,
        calibrated: balance,
    })
}

impl TrainPlan {
    /// One line per rank, for logs. `lane_names[r]` labels rank r's lanes.
    pub fn table(&self, batch: usize, lane_names: &[Vec<String>]) -> String {
        let mut s = format!(
            "  plan: {} samples/sync, {} syncs/epoch\n",
            self.samples_per_sync, self.syncs_per_epoch
        );
        for (r, p) in self.ranks.iter().enumerate() {
            let lanes: Vec<String> = p
                .lane_micro
                .iter()
                .zip(&p.lane_s)
                .enumerate()
                .map(|(i, (k, t))| {
                    let name = lane_names
                        .get(r)
                        .and_then(|n| n.get(i))
                        .map(String::as_str)
                        .unwrap_or("?");
                    if self.calibrated {
                        format!("{name}×{k} ({:.2} ms)", t * 1e3)
                    } else {
                        format!("{name}×{k}")
                    }
                })
                .collect();
            let est = if self.calibrated {
                format!(", ~{:.1} ms/sync", p.sync_s() * 1e3)
            } else {
                String::new()
            };
            s += &format!(
                "    rank {r}: {} = {} samples/sync ({:.1}%){est}, data [{}, {})\n",
                lanes.join(" + "),
                p.samples_per_sync(batch),
                100.0 * p.samples_per_sync(batch) as f64 / self.samples_per_sync as f64,
                p.data_start,
                p.data_start + p.data_len,
            );
        }
        s
    }
}

fn cerr(what: &str) -> impl Fn(rlx_driver::CollectiveError) -> String + '_ {
    move |e| format!("{what}: {e}")
}

/// Element count of trainable param `name`, from its graph node.
fn param_len(graph: &Graph, name: &str) -> Option<usize> {
    graph.nodes().iter().find_map(|n| match &n.op {
        Op::Param { name: p } if p == name => n.shape.num_elements(),
        _ => None,
    })
}

// ── device lanes ────────────────────────────────────────────────────────────

enum LaneCmd {
    /// New parameter values (shared, uploaded once per change).
    Params(Arc<Vec<Vec<f32>>>),
    /// Params bound once and never updated (`TrainSpec::frozen`).
    Frozen(Arc<Vec<(String, Vec<f32>)>>),
    /// Run batch `b` of the shared feeds; reply loss + flat gradients.
    Run(usize),
    /// Time `n` micro-steps on zero data after one warm-up; reply seconds/step.
    Probe(usize),
    /// Backward seed value (the loss scale).
    Seed(f32),
    Stop,
}

enum LaneReply {
    /// Compiled and ready for commands.
    Ready,
    Step {
        loss: f32,
        grads: Vec<f32>,
    },
    Probe(f64),
    /// The backend panicked (compile, upload or run); the lane has exited.
    Failed(String),
}

fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "backend panic".into())
}

struct Lane {
    dev: Device,
    cmd: Sender<LaneCmd>,
    reply: Receiver<LaneReply>,
    handle: std::thread::JoinHandle<f64>,
}

/// What every lane needs to run a step.
struct LaneShared {
    graph: Graph,
    names: Vec<String>,
    /// Floats in one micro-batch's flattened gradient.
    grad_len: usize,
    inputs: Vec<(String, usize)>,
    seed: Option<String>,
    batch: usize,
    loss_index: usize,
    grad_start: usize,
    /// Compile options (the number format) every lane compiles with.
    compile: crate::CompileOptions,
}

/// Spawn one lane thread per device. Each compiles the graph on its backend
/// and keeps it (`CompiledGraph` is not `Send`); returns upload seconds when it
/// stops. With `faults`, `RLX_TRAIN_LANE_FAULT=<lane>@<run>` makes lane
/// `<lane>` panic on its `<run>`-th step (0: while compiling) — a test aid for
/// the fallback path.
fn spawn_lanes(
    shared: &Arc<LaneShared>,
    feeds: &Arc<Vec<Feed>>,
    devices: &[Device],
    faults: bool,
) -> Vec<Lane> {
    let fault: Option<(usize, usize)> = faults
        .then(|| rlx_ir::env::read("RLX_TRAIN_LANE_FAULT").ok())
        .flatten()
        .and_then(|v| {
            let (l, r) = v.split_once('@')?;
            Some((l.trim().parse().ok()?, r.trim().parse().ok()?))
        });
    devices
        .iter()
        .enumerate()
        .map(|(i, &dev)| {
            let (ctx, crx) = channel::<LaneCmd>();
            let (rtx, rrx) = channel::<LaneReply>();
            let (sh, fd) = (shared.clone(), feeds.clone());
            let fault = fault.filter(|f| f.0 == i).map(|f| f.1);
            let handle = std::thread::spawn(move || lane_main(dev, &sh, &fd, crx, rtx, fault));
            Lane {
                dev,
                cmd: ctx,
                reply: rrx,
                handle,
            }
        })
        .collect()
}

fn lane_main(
    dev: Device,
    sh: &LaneShared,
    feeds: &[Feed],
    cmd: Receiver<LaneCmd>,
    reply: Sender<LaneReply>,
    fault: Option<usize>,
) -> f64 {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut runs = 0usize;
    // A device that cannot hold the graph (out of memory, busy with another
    // job) fails here or on its first run; report it instead of dying, so the
    // rank can carry on with its other lanes.
    let mut compiled = match catch_unwind(AssertUnwindSafe(|| {
        assert!(
            fault != Some(0),
            "injected fault (RLX_TRAIN_LANE_FAULT) at compile"
        );
        Session::new(dev).compile_with(sh.graph.clone(), &sh.compile)
    })) {
        Ok(c) => c,
        Err(p) => {
            let _ = reply.send(LaneReply::Failed(panic_text(p)));
            return 0.0;
        }
    };
    if reply.send(LaneReply::Ready).is_err() {
        return 0.0;
    }
    let np = sh.names.len();
    // Read only the loss and the gradients (skip logits and other outputs).
    let mut read: Vec<usize> = std::iter::once(sh.loss_index)
        .chain(sh.grad_start..sh.grad_start + np)
        .collect();
    read.sort_unstable();
    read.dedup();
    let n_out = sh.graph.outputs.len();
    let mut seed = [1.0f32];
    let zeros: Vec<Vec<f32>> = sh
        .inputs
        .iter()
        .map(|(_, e)| vec![0.0f32; sh.batch * e])
        .collect();
    let mut upload_s = 0.0f64;
    // Whether any parameter values have been uploaded. The calibration probe
    // runs before they have; CPU and GPU backends read unset params as zeros,
    // but Core ML refuses to run without every param ("missing param").
    let mut have_params = false;
    let run = |compiled: &mut crate::CompiledGraph,
               data: Vec<&[f32]>,
               one: &[f32; 1]|
     -> (f32, Vec<f32>) {
        let mut feed: Vec<(&str, &[f32])> = sh
            .inputs
            .iter()
            .zip(data)
            .map(|((n, _), d)| (n.as_str(), d))
            .collect();
        if let Some(s) = &sh.seed {
            feed.push((s.as_str(), one));
        }
        let outs = compiled.run_read_outputs(&feed, Some(&read));
        // Backends return either the selected outputs (sorted) or all of them.
        let pick = |i: usize| -> &[f32] {
            if outs.len() == n_out && n_out != read.len() {
                &outs[i]
            } else {
                &outs[read.binary_search(&i).unwrap()]
            }
        };
        let loss = pick(sh.loss_index).first().copied().unwrap_or(f32::NAN);
        let mut grads = Vec::new();
        for i in 0..np {
            grads.extend_from_slice(pick(sh.grad_start + i));
        }
        (loss, grads)
    };
    while let Ok(c) = cmd.recv() {
        let r = catch_unwind(AssertUnwindSafe(|| -> Option<LaneReply> {
            match c {
                LaneCmd::Params(p) => {
                    have_params = true;
                    let t = Instant::now();
                    for (n, v) in sh.names.iter().zip(p.iter()) {
                        compiled.set_param(n, v);
                    }
                    upload_s += t.elapsed().as_secs_f64();
                }
                LaneCmd::Run(b) => {
                    runs += 1;
                    assert!(
                        fault != Some(runs),
                        "injected fault (RLX_TRAIN_LANE_FAULT) at run {runs}"
                    );
                    let data: Vec<&[f32]> = feeds
                        .iter()
                        .map(|f| {
                            let off = b * sh.batch * f.elem;
                            &f.data[off..off + sh.batch * f.elem]
                        })
                        .collect();
                    let (loss, grads) = run(&mut compiled, data, &seed);
                    return Some(LaneReply::Step { loss, grads });
                }
                LaneCmd::Probe(n) => {
                    if !have_params {
                        for node in sh.graph.nodes() {
                            if let rlx_ir::Op::Param { name } = &node.op {
                                let len = node.shape.num_elements().unwrap_or(0);
                                compiled.set_param(name, &vec![0.0f32; len]);
                            }
                        }
                        have_params = true;
                    }
                    let data = || zeros.iter().map(Vec::as_slice).collect::<Vec<_>>();
                    run(&mut compiled, data(), &seed); // warm-up (first-run compile/upload)
                    // Every sync uploads the trained params before its steps, so
                    // each timed step does too. A backend that bakes weights
                    // into its compiled model (Core ML) recompiles on every
                    // upload; a probe that skips it times the fast path and
                    // plans that lane a full share.
                    let trained: Vec<(String, usize)> = sh
                        .graph
                        .nodes()
                        .iter()
                        .filter_map(|n| match &n.op {
                            rlx_ir::Op::Param { name } if sh.names.contains(name) => {
                                Some((name.clone(), n.shape.num_elements().unwrap_or(0)))
                            }
                            _ => None,
                        })
                        .collect();
                    let fresh: Vec<Vec<f32>> =
                        trained.iter().map(|(_, len)| vec![0.0f32; *len]).collect();
                    // Median, not mean: ms-scale steps jitter with the OS and the
                    // other lanes; one slow step must not move the whole plan.
                    let mut ts: Vec<f64> = (0..n.max(1))
                        .map(|_| {
                            let t = Instant::now();
                            for ((name, _), v) in trained.iter().zip(&fresh) {
                                compiled.set_param(name, v);
                            }
                            run(&mut compiled, data(), &seed);
                            t.elapsed().as_secs_f64()
                        })
                        .collect();
                    ts.sort_by(f64::total_cmp);
                    let per = ts[ts.len() / 2];
                    return Some(LaneReply::Probe(per));
                }
                LaneCmd::Seed(v) => seed[0] = v,
                LaneCmd::Frozen(f) => {
                    have_params = true;
                    let t = Instant::now();
                    for (n, v) in f.iter() {
                        compiled.set_param(n, v);
                    }
                    upload_s += t.elapsed().as_secs_f64();
                }
                LaneCmd::Stop => return Some(LaneReply::Failed(String::new())),
            }
            None
        }));
        match r {
            Ok(None) => {}
            // Stop: leave quietly.
            Ok(Some(LaneReply::Failed(m))) if m.is_empty() => break,
            Ok(Some(rep)) => {
                if reply.send(rep).is_err() {
                    break;
                }
            }
            Err(p) => {
                let _ = reply.send(LaneReply::Failed(panic_text(p)));
                break;
            }
        }
    }
    upload_s
}

/// A rank's device lanes, with what replaces a lane that fails: every lane's
/// micro-batches are routed to a live lane — its own, or after a failure the
/// first survivor (a CPU lane spawned if none survives) — so the cluster's
/// plan and sync cadence never change.
struct LanePool {
    shared: Arc<LaneShared>,
    feeds: Arc<Vec<Feed>>,
    lanes: Vec<Option<Lane>>,
    /// Logical lane (the plan's) → physical lane.
    route: Vec<usize>,
    params: Option<Arc<Vec<Vec<f32>>>>,
    frozen: Option<Arc<Vec<(String, Vec<f32>)>>>,
    seed: f32,
    upload_s: f64,
    rank: usize,
    log: bool,
}

impl LanePool {
    /// Spawn a lane per device and keep those that compile; with none, a CPU
    /// lane. There is one logical lane per device either way (a dropped one
    /// routes to a survivor), so a plan made for `devices` still holds.
    /// Returns the pool and the devices that actually run.
    fn start(
        shared: &Arc<LaneShared>,
        feeds: &Arc<Vec<Feed>>,
        devices: &[Device],
        rank: usize,
        log: bool,
    ) -> Result<(Self, Vec<Device>), String> {
        let mut alive = Vec::new();
        let mut route = Vec::with_capacity(devices.len());
        for lane in spawn_lanes(shared, feeds, devices, true) {
            match lane.reply.recv() {
                Ok(LaneReply::Ready) => {
                    route.push(Some(alive.len()));
                    alive.push(lane);
                }
                other => {
                    let why = match other {
                        Ok(LaneReply::Failed(e)) => e,
                        _ => "exited".into(),
                    };
                    if log {
                        eprintln!(
                            "  rank {rank}: lane {} cannot run this graph ({}) — dropped",
                            crate::device_label(lane.dev),
                            why.lines().next().unwrap_or("")
                        );
                    }
                    let _ = lane.handle.join();
                    route.push(None);
                }
            }
        }
        if alive.is_empty() {
            if log {
                eprintln!(
                    "  rank {rank}: no device lane runs this graph — falling back to the CPU"
                );
            }
            let cpu = spawn_lanes(shared, feeds, &[Device::Cpu], false).remove(0);
            match cpu.reply.recv() {
                Ok(LaneReply::Ready) => alive.push(cpu),
                Ok(LaneReply::Failed(e)) => {
                    return Err(format!("rank {rank}: even the CPU lane failed: {e}"));
                }
                _ => return Err(format!("rank {rank}: the CPU lane exited")),
            }
        }
        let devs = alive.iter().map(|l| l.dev).collect();
        Ok((
            Self {
                shared: shared.clone(),
                feeds: feeds.clone(),
                lanes: alive.into_iter().map(Some).collect(),
                route: route.into_iter().map(|r| r.unwrap_or(0)).collect(),
                params: None,
                frozen: None,
                seed: 1.0,
                upload_s: 0.0,
                rank,
                log,
            },
            devs,
        ))
    }

    fn live(&self) -> impl Iterator<Item = &Lane> {
        self.lanes.iter().flatten()
    }

    fn devices(&self) -> Vec<Device> {
        self.live().map(|l| l.dev).collect()
    }

    fn broadcast(&self, mk: impl Fn() -> LaneCmd) {
        for l in self.live() {
            let _ = l.cmd.send(mk());
        }
    }

    fn set_params(&mut self, p: Arc<Vec<Vec<f32>>>) {
        self.broadcast(|| LaneCmd::Params(p.clone()));
        self.params = Some(p);
    }

    fn set_frozen(&mut self, f: Arc<Vec<(String, Vec<f32>)>>) {
        self.broadcast(|| LaneCmd::Frozen(f.clone()));
        self.frozen = Some(f);
    }

    fn set_seed(&mut self, v: f32) {
        self.broadcast(|| LaneCmd::Seed(v));
        self.seed = v;
    }

    /// Lane `p` failed: drop it, and route everything it served to a live lane
    /// (spawning a CPU lane, brought up to date, if none is left).
    fn replace(&mut self, p: usize, why: &str) -> Result<usize, String> {
        if let Some(l) = self.lanes[p].take() {
            if self.log {
                eprintln!(
                    "  rank {}: lane {} failed ({}) — its micro-batches move to another lane",
                    self.rank,
                    crate::device_label(l.dev),
                    why.lines().next().unwrap_or(why)
                );
            }
            let _ = l.handle.join();
        }
        let to = match self.lanes.iter().position(Option::is_some) {
            Some(i) => i,
            None => {
                let cpu = spawn_lanes(&self.shared, &self.feeds, &[Device::Cpu], false).remove(0);
                match cpu.reply.recv() {
                    Ok(LaneReply::Ready) => {}
                    other => {
                        let why = match other {
                            Ok(LaneReply::Failed(e)) => e,
                            _ => "exited".into(),
                        };
                        return Err(format!(
                            "rank {}: no lane left and the CPU fallback failed: {why}",
                            self.rank
                        ));
                    }
                }
                if let Some(f) = &self.frozen {
                    let _ = cpu.cmd.send(LaneCmd::Frozen(f.clone()));
                }
                if let Some(pr) = &self.params {
                    let _ = cpu.cmd.send(LaneCmd::Params(pr.clone()));
                }
                let _ = cpu.cmd.send(LaneCmd::Seed(self.seed));
                self.lanes.push(Some(cpu));
                self.lanes.len() - 1
            }
        };
        for r in &mut self.route {
            if *r == p {
                *r = to;
            }
        }
        Ok(to)
    }

    /// One sync's micro-batches. `batches[li]` is logical lane `li`'s planned
    /// share; the lanes with a share then *pull* from one queue of all of
    /// them — each keeps two in flight and gets the next as one finishes — so
    /// a lane slowed mid-run (a CPU lane on a machine that started compiling)
    /// takes fewer and the others more, while the rank's total, and so the
    /// cluster's plan and sync cadence, stay fixed. A lane the plan sat out
    /// (no share) pulls nothing. `each(loss, grads)` sees every micro-batch in
    /// batch order when the buffered gradients fit `ORDERED_BYTES` (the sum is
    /// then the same whichever lane ran what), else in arrival order. A lane
    /// that fails has its unanswered batches put back in the queue.
    fn sync(
        &mut self,
        batches: &[Vec<usize>],
        mut each: impl FnMut(f32, &[f32]),
    ) -> Result<(), String> {
        use std::collections::{BTreeMap, VecDeque};
        let mut queue: VecDeque<usize> = {
            let mut all: Vec<usize> = batches.iter().flatten().copied().collect();
            all.sort_unstable();
            all.into()
        };
        let n = queue.len();
        let mut workers: Vec<usize> = batches
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.is_empty())
            .map(|(li, _)| self.route[li])
            .collect();
        workers.sort_unstable();
        workers.dedup();
        let ordered = n * self.shared.grad_len * 4 <= ORDERED_BYTES;
        let mut held: BTreeMap<usize, (f32, Vec<f32>)> = BTreeMap::new();
        let mut out: Vec<VecDeque<usize>> = vec![VecDeque::new(); self.lanes.len()];
        let mut done = 0usize;
        let feed = |lanes: &[Option<Lane>],
                    out: &mut Vec<VecDeque<usize>>,
                    queue: &mut VecDeque<usize>,
                    p: usize| {
            while out[p].len() < 2 {
                let Some(b) = queue.pop_front() else { break };
                if let Some(l) = &lanes[p] {
                    let _ = l.cmd.send(LaneCmd::Run(b));
                }
                out[p].push_back(b);
            }
        };
        for &p in &workers {
            feed(&self.lanes, &mut out, &mut queue, p);
        }
        while done < n {
            let mut progressed = false;
            for i in 0..workers.len() {
                let p = workers[i];
                if out[p].is_empty() {
                    continue;
                }
                let r = match self.lanes[p].as_ref().map(|l| l.reply.try_recv()) {
                    Some(Ok(r)) => Ok(r),
                    Some(Err(std::sync::mpsc::TryRecvError::Empty)) => continue,
                    _ => Err("exited".to_string()),
                };
                progressed = true;
                match r {
                    Ok(LaneReply::Step { loss, grads }) => {
                        let b = out[p].pop_front().expect("an answered batch was sent");
                        if ordered {
                            held.insert(b, (loss, grads));
                        } else {
                            each(loss, &grads);
                        }
                        done += 1;
                        feed(&self.lanes, &mut out, &mut queue, p);
                    }
                    Ok(LaneReply::Failed(e)) | Err(e) => {
                        // Its unanswered work goes back; the replacement (or a
                        // survivor already working) takes it from the queue.
                        for b in std::mem::take(&mut out[p]).into_iter().rev() {
                            queue.push_front(b);
                        }
                        let to = self.replace(p, &e)?;
                        if out.len() < self.lanes.len() {
                            out.resize(self.lanes.len(), VecDeque::new());
                        }
                        workers.retain(|&w| w != p);
                        if !workers.contains(&to) {
                            workers.push(to);
                        }
                        for &w in &workers {
                            feed(&self.lanes, &mut out, &mut queue, w);
                        }
                        break;
                    }
                    Ok(_) => return Err("lane replied out of order".into()),
                }
            }
            if !progressed {
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }
        for (_, (loss, grads)) in held {
            each(loss, &grads);
        }
        Ok(())
    }

    fn stop(self) -> f64 {
        let mut up = self.upload_s;
        for l in self.lanes.into_iter().flatten() {
            let _ = l.cmd.send(LaneCmd::Stop);
            up = up.max(l.handle.join().unwrap_or(0.0));
        }
        up
    }
}

// ── wire ────────────────────────────────────────────────────────────────────

/// Sum-all-reduce `v` in place over `wire`.
fn all_reduce_sum(group: &ProcessGroup, v: &mut [f32], wire: WireDtype) -> Result<(), String> {
    match wire {
        WireDtype::F32 | WireDtype::Auto => group
            .all_reduce(v, ReduceKind::Sum)
            .map_err(cerr("gradient all-reduce")),
        WireDtype::Bf16 | WireDtype::F16 => {
            let mut bytes: Vec<u8> = Vec::with_capacity(v.len() * 2);
            for &x in v.iter() {
                let b = if wire == WireDtype::Bf16 {
                    half::bf16::from_f32(x).to_bits()
                } else {
                    half::f16::from_f32(x).to_bits()
                };
                bytes.extend_from_slice(&b.to_le_bytes());
            }
            group
                .all_reduce_typed(&mut bytes, wire.dtype(), ReduceKind::Sum)
                .map_err(cerr("gradient all-reduce"))?;
            for (x, c) in v.iter_mut().zip(bytes.as_chunks::<2>().0) {
                let bits = u16::from_le_bytes(*c);
                *x = if wire == WireDtype::Bf16 {
                    half::bf16::from_bits(bits).to_f32()
                } else {
                    half::f16::from_bits(bits).to_f32()
                };
            }
            Ok(())
        }
    }
}

/// `(α seconds, β bytes/s)` of the group's link: median of 10 barriers, and a
/// 256 KiB f32 ring all-reduce (3 runs, best). Every rank must call it.
fn measure_link(group: &ProcessGroup) -> Result<(f64, f64), String> {
    let mut bar = Vec::with_capacity(10);
    for _ in 0..10 {
        let t = Instant::now();
        group.barrier().map_err(cerr("link probe"))?;
        bar.push(t.elapsed().as_secs_f64());
    }
    bar.sort_by(f64::total_cmp);
    let alpha = bar[bar.len() / 2];
    let elems = 65_536usize;
    let mut buf = vec![1.0f32; elems];
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        group
            .all_reduce(&mut buf, ReduceKind::Sum)
            .map_err(cerr("link probe"))?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    let n = group.world_size().max(2) as f64;
    let moved = 2.0 * (n - 1.0) / n * (elems * 4) as f64;
    Ok((alpha, moved / (best - alpha).max(1e-6)))
}

// ── optimizer state + checkpoint ────────────────────────────────────────────

/// Momentum-SGD on a flat parameter vector; `vel` covers `range` only (the
/// whole vector, or a ZeRO slice).
fn sgd_range(
    params: &mut [f32],
    vel: &mut [f32],
    grads: &[f32],
    range: std::ops::Range<usize>,
    momentum: f32,
    lr: f32,
) {
    for (k, i) in range.enumerate() {
        vel[k] = momentum * vel[k] + grads[i];
        params[i] -= lr * vel[k];
    }
}

fn unflatten(flat: &[f32], lens: &[usize]) -> Vec<Vec<f32>> {
    let mut off = 0;
    lens.iter()
        .map(|&l| {
            let v = flat[off..off + l].to_vec();
            off += l;
            v
        })
        .collect()
}

#[derive(Serialize, Deserialize)]
struct CkptMeta {
    /// Epochs completed.
    epochs_done: usize,
    names: Vec<String>,
    lens: Vec<usize>,
    /// Optimizer state buffers saved (1: SGD velocity; 2: AdamW m, v).
    #[serde(default = "one")]
    buffers: usize,
    /// AdamW's step count.
    #[serde(default)]
    step: u64,
}

fn one() -> usize {
    1
}

const CKPT_META: &str = "rlx_train_ckpt.json";
const CKPT_PARAMS: &str = "rlx_train_params.f32";
const CKPT_VEL: &str = "rlx_train_vel.f32";

fn write_f32(path: &std::path::Path, v: &[f32]) -> std::io::Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes)
}

fn read_f32(path: &std::path::Path) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    Some(
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    )
}

/// Write atomically enough for a crash mid-write to leave the previous
/// checkpoint: data files first, the meta (the commit point) last via rename.
fn save_checkpoint(
    dir: &str,
    epochs_done: usize,
    names: &[String],
    lens: &[usize],
    params: &[f32],
    vel: &[f32],
    step: u64,
) -> Result<(), String> {
    let d = std::path::Path::new(dir);
    std::fs::create_dir_all(d).map_err(|e| format!("checkpoint dir {dir}: {e}"))?;
    write_f32(&d.join(CKPT_PARAMS), params).map_err(|e| format!("checkpoint: {e}"))?;
    write_f32(&d.join(CKPT_VEL), vel).map_err(|e| format!("checkpoint: {e}"))?;
    let total: usize = lens.iter().sum();
    let meta = CkptMeta {
        epochs_done,
        names: names.to_vec(),
        lens: lens.to_vec(),
        buffers: vel.len().checked_div(total).unwrap_or(1),
        step,
    };
    let tmp = d.join(format!("{CKPT_META}.tmp"));
    std::fs::write(&tmp, serde_json::to_vec(&meta).unwrap()).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, d.join(CKPT_META)).map_err(|e| e.to_string())
}

/// `(epochs_done, params, state, step)` if `dir` holds a checkpoint of this
/// job; `state` is the optimizer's buffers back to back.
fn load_checkpoint(
    dir: &str,
    names: &[String],
    lens: &[usize],
) -> Option<(usize, Vec<f32>, Vec<f32>, u64)> {
    let d = std::path::Path::new(dir);
    let meta: CkptMeta = serde_json::from_slice(&std::fs::read(d.join(CKPT_META)).ok()?).ok()?;
    if meta.names != names || meta.lens != lens {
        return None;
    }
    let total: usize = lens.iter().sum();
    let p = read_f32(&d.join(CKPT_PARAMS))?;
    let v = read_f32(&d.join(CKPT_VEL))?;
    (p.len() == total && v.len() == total * meta.buffers).then_some((
        meta.epochs_done,
        p,
        v,
        meta.step,
    ))
}

// ── a worker rank ───────────────────────────────────────────────────────────

/// A whole `dist_node MODE=trainserve` worker for a cluster-options job
/// (`TrainSpec::options` set — what `rlx-vision-bench --coordinate --cluster`
/// and `model::distributed::TrainJob` ship), in the order every rank runs it:
/// a barrier once connected, [`recv_train`], [`run_train_group`], then the
/// post-job and final barriers. A rank that is not `dist_node` (the iOS /
/// Android node) uses this; any collective out of that order stalls the mesh.
pub fn trainserve_worker(group: &ProcessGroup, log: bool) -> Result<TrainMetrics, String> {
    group
        .barrier()
        .map_err(|e| format!("initial barrier: {e}"))?;
    let spec = super::recv_train(group)?;
    if spec.options.is_none() {
        return Err(
            "this job uses the in-graph collective path (no cluster options), \
                    which needs dist_node; run the master with cluster options"
                .into(),
        );
    }
    if log {
        eprintln!(
            "rlx-node: training job — {} params, {} inputs, {} epoch(s), batch {}",
            spec.params.len(),
            spec.data.len(),
            spec.lr_per_epoch.len(),
            spec.batch
        );
    }
    let (metrics, _) = run_train_group(&spec, group, super::uri_resolver, log)?;
    for what in ["post-job", "final"] {
        group
            .barrier()
            .map_err(|e| format!("{what} barrier: {e}"))?;
    }
    Ok(metrics)
}

// ── the loop ────────────────────────────────────────────────────────────────

/// Train `spec` across `group` with `spec.options` (defaults when `None`).
/// Every rank calls this with the same spec; rank 0 is the master that owns
/// the data and init params when `spec.push_data` is set. `resolve` is the
/// fallback for weight/data URI schemes the built-in cache does not handle.
///
/// Returns this rank's metrics and the final parameters (identical on every
/// rank). Communication failures (a peer died, `RLX_COLLECTIVE_TIMEOUT`) come
/// back as `Err` naming the rank; with a `checkpoint_dir`, relaunching resumes
/// from the last completed epoch.
pub fn run_train_group<W>(
    spec: &TrainSpec,
    group: &ProcessGroup,
    mut resolve: W,
    log: bool,
) -> Result<(TrainMetrics, Vec<(String, Vec<f32>)>), String>
where
    W: FnMut(&str) -> Vec<f32>,
{
    let opts = spec.options.clone().unwrap_or_default();
    opts.validate()?;
    let rank = group.rank() as usize;
    let names: Vec<String> = spec.params.iter().map(|w| w.name.clone()).collect();
    let lens: Vec<usize> = names
        .iter()
        .map(|n| param_len(&spec.graph, n).ok_or_else(|| format!("param {n} is not in the graph")))
        .collect::<Result<_, _>>()?;
    let total: usize = lens.iter().sum();
    let n_samples = spec.data.first().map(|d| d.shard_len).unwrap_or(0);
    let batch = spec.batch.max(1);

    // 1. Agree on the job.
    let fp = [
        n_samples as f32,
        batch as f32,
        spec.lr_per_epoch.len() as f32,
        total as f32,
        opts.accum as f32,
        opts.local_steps as f32,
        opts.zero as u8 as f32,
        opts.wire as u8 as f32,
        opts.balance as u8 as f32,
        opts.overlap as u8 as f32,
        opts.half_casts as u8 as f32,
        opts.optimizer.buffers() as f32,
        // Every rank must compute in the same format: compare a 23-bit hash.
        opts.number_format
            .as_deref()
            .unwrap_or("f32")
            .bytes()
            .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32))
            .rem_euclid(1 << 23) as f32,
    ];
    let all = group.all_gather(&fp).map_err(cerr("job check"))?;
    if all.chunks(fp.len()).any(|c| c != fp) {
        return Err(format!(
            "run_train_group: ranks disagree on the job (rank {rank} has {fp:?}) — every \
             rank must receive the same TrainSpec"
        ));
    }

    // Lanes (compile lazily inside the threads; feeds come after planning).
    let inventory = crate::available_devices();
    // `RLX_TRAIN_LANES` lets one node pick its own lanes (e.g. add an iGPU only
    // Vulkan reaches). Safe per node: lane counts are exchanged when planning.
    let lane_spec = rlx_ir::env::read("RLX_TRAIN_LANES").unwrap_or_else(|_| spec.device.clone());
    let mut devices = training_lanes(&lane_spec, &inventory, &spec.graph);
    let shared = Arc::new(LaneShared {
        graph: spec.graph.clone(),
        names: names.clone(),
        inputs: spec
            .data
            .iter()
            .map(|d| (d.input.clone(), d.elem))
            .collect(),
        seed: spec.seed_input.clone(),
        batch,
        loss_index: spec.loss_index,
        grad_start: spec.grad_start,
        grad_len: total,
        compile: crate::CompileOptions {
            number_format: opts.parsed_number_format()?,
            half_casts: if opts.half_casts {
                crate::HalfCasts::Round
            } else {
                crate::CompileOptions::default().half_casts
            },
            ..Default::default()
        },
    });

    // 2. Calibrate each lane on zero data (real compute), all lanes at once
    //    so CPU and GPU lanes contend the way they will when training.
    //    A lane that cannot run the graph drops out of the plan here.
    let lane_times: Vec<f64> = if opts.balance {
        let (pool, alive) = LanePool::start(&shared, &Arc::new(Vec::new()), &devices, rank, log)?;
        devices = alive;
        pool.broadcast(|| LaneCmd::Probe(opts.probe_steps));
        let mut ts = Vec::with_capacity(devices.len());
        for l in pool.live() {
            match l.reply.recv() {
                Ok(LaneReply::Probe(s)) => ts.push(s),
                Ok(LaneReply::Failed(e)) => {
                    return Err(format!(
                        "rank {rank}: lane {} failed while calibrating: {e}",
                        crate::device_label(l.dev)
                    ));
                }
                _ => return Err("lane replied out of order".into()),
            }
        }
        pool.stop();
        ts
    } else {
        vec![1.0; devices.len()]
    };
    if devices.len() > MAX_LANES {
        return Err(format!(
            "rank {rank}: {} lanes > {MAX_LANES}",
            devices.len()
        ));
    }

    // 3. Plan: all-gather [lanes, t_0..t_15, device codes…] fixed-width rows.
    let mut mine = vec![0.0f32; 1 + 2 * MAX_LANES];
    mine[0] = devices.len() as f32;
    for (i, (t, d)) in lane_times.iter().zip(&devices).enumerate() {
        mine[1 + i] = *t as f32;
        mine[1 + MAX_LANES + i] = Device::all().iter().position(|x| x == d).unwrap_or(0) as f32;
    }
    let gathered = group.all_gather(&mine).map_err(cerr("plan exchange"))?;
    let rows: Vec<&[f32]> = gathered.chunks(mine.len()).collect();
    let lane_s: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| (0..row[0] as usize).map(|i| row[1 + i] as f64).collect())
        .collect();
    let lane_names: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            (0..row[0] as usize)
                .map(|i| {
                    Device::all()
                        .get(row[1 + MAX_LANES + i] as usize)
                        .map(|d| crate::device_label(*d).to_string())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();
    let plan = plan_training(&lane_s, batch, opts.accum, opts.balance, n_samples)?;
    if log && rank == 0 {
        eprint!("{}", plan.table(batch, &lane_names));
    }
    let me = plan.ranks[rank].clone();
    let s_total = plan.samples_per_sync as f32;
    let my_samples = me.samples_per_sync(batch) as f32;

    // Gradient wire: resolve `Auto` once, on rank 0's measurement of the link,
    // and broadcast it so every rank sends the same width.
    let world_n = group.world_size() as usize;
    let wire = if opts.wire == WireDtype::Auto {
        if world_n <= 1 || opts.zero || opts.local_steps > 0 {
            WireDtype::F32
        } else {
            let (alpha, beta) = measure_link(group)?;
            let bytes = (total * 4) as f64;
            let mut pick = [if bytes / beta > alpha { 1.0f32 } else { 0.0 }];
            group
                .broadcast(0, &mut pick)
                .map_err(cerr("wire decision"))?;
            if log && rank == 0 {
                eprintln!(
                    "  wire auto: link α={:.2} ms β={:.1} MB/s, gradients {:.1} KiB → {}",
                    alpha * 1e3,
                    beta / 1e6,
                    bytes / 1024.0,
                    if pick[0] > 0.5 {
                        "bf16 + error feedback (bandwidth-bound)"
                    } else {
                        "f32 (latency-bound)"
                    }
                );
            }
            if pick[0] > 0.5 {
                WireDtype::Bf16
            } else {
                WireDtype::F32
            }
        }
    } else {
        opts.wire
    };
    let half_wire = matches!(wire, WireDtype::Bf16 | WireDtype::F16);
    let mut residual = if half_wire && opts.wire_feedback {
        vec![0.0f32; total]
    } else {
        Vec::new()
    };
    let (mut rel_err_sum, mut rel_err_n) = (0.0f64, 0usize);
    let mut wire_bytes = 0u64;
    // Ring all-reduce: each rank sends 2·(n−1)/n of the buffer.
    let ring = |elems: usize, esize: usize| -> u64 {
        if world_n <= 1 {
            0
        } else {
            (2 * (world_n - 1) * elems * esize / world_n) as u64
        }
    };

    // 4. Params: resume > pushed > local.
    let mut cache = WeightCache::new();
    let mut flat: Vec<f32> = if rank == 0 || !spec.push_data {
        spec.params
            .iter()
            .flat_map(|w| cache.f32(&w.uri).unwrap_or_else(|_| resolve(&w.uri)))
            .collect()
    } else {
        vec![0.0; total]
    };
    if flat.len() != total {
        return Err(format!(
            "rank {rank}: resolved {} param values, the graph needs {total}",
            flat.len()
        ));
    }
    let nbuf = opts.optimizer.buffers();
    let mut vel_full = vec![0.0f32; total * nbuf];
    let mut start_epoch = 0usize;
    // [epochs done, optimizer step]
    let mut resume = [0.0f32; 2];
    if rank == 0
        && let Some(dir) = &opts.checkpoint_dir
        && let Some((done, p, v, step)) = load_checkpoint(dir, &names, &lens)
        && done < spec.lr_per_epoch.len()
        && v.len() == total * nbuf
    {
        resume = [done as f32, step as f32];
        flat = p;
        vel_full = v;
    }
    group
        .broadcast(0, &mut resume)
        .map_err(cerr("resume broadcast"))?;
    start_epoch = start_epoch.max(resume[0] as usize);
    if spec.push_data || start_epoch > 0 {
        group
            .broadcast(0, &mut flat)
            .map_err(cerr("param broadcast"))?;
    }
    if start_epoch > 0 {
        group
            .broadcast(0, &mut vel_full)
            .map_err(cerr("momentum broadcast"))?;
        if log && rank == 0 {
            eprintln!("  resuming after epoch {start_epoch} from checkpoint");
        }
    }
    // ZeRO keeps only this rank's slice of the optimizer state.
    let own = if opts.zero {
        group.owned_range(total)
    } else {
        0..total
    };
    let mut vel = OptState {
        bufs: (0..nbuf)
            .map(|b| vel_full[b * total + own.start..b * total + own.end].to_vec())
            .collect(),
        step: resume[1] as u64,
    };
    drop(vel_full);
    let optimizer = opts.optimizer;

    // Frozen params: resolved (or, with push_data, sent by rank 0) once.
    let frozen: Vec<(String, Vec<f32>)> = spec
        .frozen
        .iter()
        .map(|w| {
            let len = param_len(&spec.graph, &w.name)
                .ok_or_else(|| format!("frozen param {} is not in the graph", w.name))?;
            let mut v = if rank == 0 || !spec.push_data {
                cache.f32(&w.uri).unwrap_or_else(|_| resolve(&w.uri))
            } else {
                vec![0.0; len]
            };
            if spec.push_data {
                v.resize(len, 0.0);
                group
                    .broadcast(0, &mut v)
                    .map_err(cerr("frozen param broadcast"))?;
            }
            if v.len() != len {
                return Err(format!(
                    "rank {rank}: frozen param {} has {} values, the graph needs {len}",
                    w.name,
                    v.len()
                ));
            }
            Ok((w.name.clone(), v))
        })
        .collect::<Result<_, String>>()?;

    // 5. Data slices.
    let feeds: Vec<Feed> = if spec.push_data {
        if rank == 0 {
            let mut mine_feeds = Vec::new();
            for (i, d) in spec.data.iter().enumerate() {
                let full = resolve_shard(
                    &d.uri,
                    d.elem,
                    d.shard_start,
                    d.shard_len,
                    &mut cache,
                    &mut resolve,
                );
                for (r, p) in plan.ranks.iter().enumerate() {
                    let sl = &full[p.data_start * d.elem..(p.data_start + p.data_len) * d.elem];
                    if r == 0 {
                        mine_feeds.push(Feed {
                            input: d.input.clone(),
                            elem: d.elem,
                            data: sl.to_vec(),
                        });
                    } else {
                        group
                            .send_f32(r as u32, TAG_SLICE_BASE + i as u32, sl)
                            .map_err(cerr("push data slice"))?;
                    }
                }
            }
            mine_feeds
        } else {
            spec.data
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    group
                        .recv_f32(0, TAG_SLICE_BASE + i as u32)
                        .map(|data| Feed {
                            input: d.input.clone(),
                            elem: d.elem,
                            data,
                        })
                        .map_err(cerr("receive data slice"))
                })
                .collect::<Result<_, _>>()?
        }
    } else {
        spec.data
            .iter()
            .map(|d| Feed {
                input: d.input.clone(),
                elem: d.elem,
                data: resolve_shard(
                    &d.uri,
                    d.elem,
                    d.shard_start + me.data_start,
                    me.data_len,
                    &mut cache,
                    &mut resolve,
                ),
            })
            .collect()
    };
    if feeds.iter().any(|f| f.data.len() < me.data_len * f.elem) {
        return Err(format!("rank {rank}: data slice shorter than planned"));
    }

    // 6. Train.
    let feeds = Arc::new(feeds);
    let (mut pool, running) = LanePool::start(&shared, &feeds, &devices, rank, log)?;
    if log {
        let ls: Vec<&str> = running.iter().map(|d| crate::device_label(*d)).collect();
        eprintln!(
            "  rank {rank}: lanes {ls:?} × {:?} micro-batches/sync, {} samples/sync of {}",
            me.lane_micro, my_samples, s_total
        );
    }
    if !frozen.is_empty() {
        pool.set_frozen(Arc::new(frozen));
    }
    pool.set_params(Arc::new(unflatten(&flat, &lens)));

    let wall = Instant::now();
    let (mut compute_s, mut comm_s) = (0.0f64, 0.0f64);
    let (mut first_loss, mut last_loss) = (f32::NAN, f32::NAN);
    let mut samples = 0usize;
    let mut syncs_done = 0usize;
    // Lane li owns micro-batches [offs[li], offs[li] + k[li]) of each sync.
    let per_sync = me.micro_steps();
    let offs: Vec<usize> = me
        .lane_micro
        .iter()
        .scan(0, |a, &k| {
            let o = *a;
            *a += k;
            Some(o)
        })
        .collect();
    // Each micro-batch's mean gradient enters the bucket weighted by its
    // share of the sync: B/S cluster-wide, or B/(own samples) for local SGD.
    // Loss scaling: lanes seed the backward with `scale`; dividing here
    // brings the gradients back to true units before they leave the rank.
    let mut scale = opts.initial_loss_scale();
    let mut skipped = 0usize;
    pool.set_seed(scale);
    if log && rank == 0 && (scale != 1.0 || opts.number_format.is_some()) {
        eprintln!(
            "  number format {} · loss scale {scale}",
            opts.number_format.as_deref().unwrap_or("f32")
        );
    }
    let base_weight = if opts.local_steps > 0 {
        batch as f32 / my_samples
    } else {
        batch as f32 / s_total
    };
    let epochs = spec.lr_per_epoch.len();
    // Learning-rate scaling and warmup, from plan-wide numbers so every rank
    // applies the same rate.
    let update_batch = if opts.local_steps > 0 {
        s_total as f64 / plan.ranks.len().max(1) as f64
    } else {
        s_total as f64
    };
    let lr_factor = opts.lr_scaling.factor(update_batch, batch as f64);
    let total_updates = epochs * plan.syncs_per_epoch;
    let warmup = opts
        .warmup
        .unwrap_or(if opts.lr_scaling == LrScaling::None {
            0
        } else {
            (total_updates / 20).max(1)
        });
    if log && rank == 0 && (lr_factor != 1.0 || warmup > 0) {
        eprintln!(
            "  lr × {lr_factor:.2} ({:?}: {update_batch:.0} samples per update vs batch {batch}), \
             warmup {warmup} update(s)",
            opts.lr_scaling
        );
    }
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
    // A sync whose gradient overflowed: skip its update, halve the scale.
    let backoff =
        |pool: &mut LanePool, scale: &mut f32, skipped: &mut usize| -> Result<(), String> {
            *skipped += 1;
            *scale = (*scale / 2.0).max(1.0 / 1024.0);
            pool.set_seed(*scale);
            Ok(())
        };

    let result: Result<(), String> = std::thread::scope(|scope| {
        // Overlap: the previous sync's reduce, still in flight.
        let mut pending: Option<(
            std::thread::ScopedJoinHandle<'_, Result<Vec<f32>, String>>,
            f32,
        )> = None;
        for epoch in start_epoch..epochs {
            let lr = spec.lr_per_epoch[epoch];
            let mut epoch_loss = 0.0f64;
            for s in 0..plan.syncs_per_epoch {
                let step = (epoch - start_epoch) * plan.syncs_per_epoch + s;
                let ramp = if step < warmup {
                    (step + 1) as f32 / warmup as f32
                } else {
                    1.0
                };
                let lr = lr * lr_factor * ramp;
                // Compute this sync's contribution.
                let t = Instant::now();
                let weight = base_weight / scale;
                let mut bucket = vec![0.0f32; total];
                let mut loss_sum = 0.0f64;
                // Every lane's micro-batches, run concurrently; a lane that
                // fails mid-run has its share re-run on another lane.
                let batches: Vec<Vec<usize>> = (0..me.lane_micro.len())
                    .map(|li| {
                        (0..me.lane_micro[li])
                            .map(|m| s * per_sync + offs[li] + m)
                            .collect()
                    })
                    .collect();
                pool.sync(&batches, |loss, grads| {
                    loss_sum += loss as f64;
                    for (a, g) in bucket.iter_mut().zip(grads) {
                        *a += weight * g;
                    }
                })?;
                compute_s += t.elapsed().as_secs_f64();
                epoch_loss += loss_sum / per_sync as f64;
                samples += me.samples_per_sync(batch);

                // 16-bit wire: round here (so the wire encode is exact), carry
                // the rounding error into the next sync, and record how much
                // rounding cost on these real gradients.
                if half_wire && opts.local_steps == 0 {
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for (i, x) in bucket.iter_mut().enumerate() {
                        let v = *x + residual.get(i).copied().unwrap_or(0.0);
                        let q = wire.round(v);
                        if let Some(r) = residual.get_mut(i) {
                            *r = v - q;
                        }
                        num += ((v - q) as f64).powi(2);
                        den += (v as f64).powi(2);
                        *x = q;
                    }
                    if den > 0.0 {
                        rel_err_sum += (num / den).sqrt();
                        rel_err_n += 1;
                    }
                }
                let t = Instant::now();
                if opts.local_steps > 0 {
                    // Local step on own gradients; average params periodically.
                    if finite(&bucket) {
                        vel.apply(optimizer, spec.momentum, &mut flat, &bucket, 0..total, lr);
                    } else {
                        backoff(&mut pool, &mut scale, &mut skipped)?;
                    }
                    syncs_done += 1;
                    let last = s + 1 == plan.syncs_per_epoch;
                    if syncs_done.is_multiple_of(opts.local_steps) || last {
                        let w = my_samples / s_total;
                        for p in flat.iter_mut() {
                            *p *= w;
                        }
                        // Parameters average in f32: rounding them would
                        // accumulate into the model itself.
                        all_reduce_sum(group, &mut flat, WireDtype::F32)?;
                        wire_bytes += ring(total, 4);
                    }
                } else if opts.zero {
                    let r = group
                        .reduce_scatter_owned(&mut bucket, ReduceKind::Sum)
                        .map_err(cerr("gradient reduce-scatter"))?;
                    // Slices differ per rank: agree on an overflow with one value.
                    let mut bad = [if finite(&bucket[r.clone()]) {
                        0.0f32
                    } else {
                        1.0
                    }];
                    if scale != 1.0 || opts.number_format.is_some() {
                        group
                            .all_reduce(&mut bad, ReduceKind::Sum)
                            .map_err(cerr("overflow vote"))?;
                    }
                    if bad[0] == 0.0 {
                        vel.apply(optimizer, spec.momentum, &mut flat, &bucket, r, lr);
                    } else {
                        backoff(&mut pool, &mut scale, &mut skipped)?;
                    }
                    group
                        .all_gather_owned(&mut flat)
                        .map_err(cerr("param all-gather"))?;
                    wire_bytes += ring(total, 4);
                    syncs_done += 1;
                } else if opts.overlap {
                    if let Some((h, plr)) = pending.take() {
                        let g = h
                            .join()
                            .map_err(|_| "reduce thread panicked".to_string())??;
                        if finite(&g) {
                            vel.apply(optimizer, spec.momentum, &mut flat, &g, 0..total, plr);
                        } else {
                            backoff(&mut pool, &mut scale, &mut skipped)?;
                        }
                        syncs_done += 1;
                    }
                    wire_bytes += ring(total, wire.bytes());
                    pending = Some((
                        scope.spawn(move || {
                            all_reduce_sum(group, &mut bucket, wire)?;
                            Ok(bucket)
                        }),
                        lr,
                    ));
                } else {
                    all_reduce_sum(group, &mut bucket, wire)?;
                    wire_bytes += ring(total, wire.bytes());
                    // Every rank sees the same reduced bucket → same verdict.
                    if finite(&bucket) {
                        vel.apply(optimizer, spec.momentum, &mut flat, &bucket, 0..total, lr);
                    } else {
                        residual.iter_mut().for_each(|r| *r = 0.0);
                        backoff(&mut pool, &mut scale, &mut skipped)?;
                    }
                    syncs_done += 1;
                }
                comm_s += t.elapsed().as_secs_f64();
                pool.set_params(Arc::new(unflatten(&flat, &lens)));
            }
            // Drain an in-flight reduce at epoch end so the epoch's params
            // (and its checkpoint) include every gradient.
            if let Some((h, plr)) = pending.take() {
                let t = Instant::now();
                let g = h
                    .join()
                    .map_err(|_| "reduce thread panicked".to_string())??;
                comm_s += t.elapsed().as_secs_f64();
                if finite(&g) {
                    vel.apply(optimizer, spec.momentum, &mut flat, &g, 0..total, plr);
                } else {
                    backoff(&mut pool, &mut scale, &mut skipped)?;
                }
                syncs_done += 1;
                pool.set_params(Arc::new(unflatten(&flat, &lens)));
            }
            record_epoch(
                epoch,
                (epoch_loss / plan.syncs_per_epoch as f64) as f32,
                &mut first_loss,
                &mut last_loss,
                log,
                epochs,
            );
            if let Some(dir) = &opts.checkpoint_dir {
                // ZeRO: assemble the full momentum for the file.
                // Every state buffer, whole (ZeRO: gathered from the slices).
                let mut full_vel = Vec::with_capacity(total * nbuf);
                for b in &vel.bufs {
                    if opts.zero {
                        let mut v = vec![0.0f32; total];
                        v[own.clone()].copy_from_slice(b);
                        group
                            .all_gather_owned(&mut v)
                            .map_err(cerr("checkpoint gather"))?;
                        full_vel.extend(v);
                    } else {
                        full_vel.extend_from_slice(b);
                    }
                }
                if rank == 0 {
                    save_checkpoint(dir, epoch + 1, &names, &lens, &flat, &full_vel, vel.step)?;
                }
            }
        }
        Ok(())
    });

    if log && skipped > 0 {
        eprintln!(
            "  rank {rank}: {skipped} of {syncs_done} sync(s) skipped on a non-finite gradient; \
             loss scale now {scale}"
        );
    }
    // Every rank sees the same reduced gradients, so this verdict is uniform.
    let running = pool.devices();
    let upload_s = pool.stop();
    if syncs_done > 0 && skipped == syncs_done {
        return Err(format!(
            "every sync's gradient was NaN/inf ({skipped} of {syncs_done}), so nothing trained. \
             A lane's backward produced non-finite values: lanes {:?} on rank {rank}. Try another \
             device (DEVICE / RLX_TRAIN_LANES), RLX_DEBUG_NANS=1 to locate the op, or a smaller LR",
            running
                .iter()
                .map(|d| crate::device_label(*d))
                .collect::<Vec<_>>()
        ));
    }
    result?;

    let metrics = TrainMetrics {
        device: crate::device_ext::fastest_among(&running),
        lanes: running,
        inventory,
        samples,
        wall_s: wall.elapsed().as_secs_f64(),
        compute_s,
        comm_s,
        first_loss,
        last_loss,
        upload_s,
        syncs: syncs_done,
        micro_steps: per_sync,
        wire: format!("{wire:?}").to_lowercase(),
        wire_bytes,
        skipped,
        wire_rel_err: if rel_err_n > 0 {
            rel_err_sum / rel_err_n as f64
        } else {
            0.0
        },
    };
    Ok((
        metrics,
        names.into_iter().zip(unflatten(&flat, &lens)).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_balances_by_throughput_and_tiles_the_data() {
        // rank 0: one slow lane; rank 1: a 3× faster lane; rank 2: a slow CPU
        // lane next to a 6× faster GPU lane.
        let lanes = vec![vec![0.003], vec![0.001], vec![0.003, 0.0005]];
        let p = plan_training(&lanes, 64, 2, true, 60_000).unwrap();
        let k: Vec<Vec<usize>> = p.ranks.iter().map(|r| r.lane_micro.clone()).collect();
        assert_eq!(k, vec![vec![2], vec![6], vec![2, 12]]);
        // Every lane finishes its share in the same time.
        for r in &p.ranks {
            assert!((r.sync_s() - 0.006).abs() < 1e-9, "{r:?}");
        }
        assert_eq!(p.samples_per_sync, (2 + 6 + 2 + 12) * 64);
        assert_eq!(p.syncs_per_epoch, 60_000 / p.samples_per_sync);
        let mut at = 0;
        for r in &p.ranks {
            assert_eq!(r.data_start, at);
            assert_eq!(r.data_len, p.syncs_per_epoch * r.samples_per_sync(64));
            at += r.data_len;
        }
        assert!(at <= 60_000);
        // Unbalanced: every lane runs accum.
        let u = plan_training(&lanes, 64, 2, false, 60_000).unwrap();
        assert!(u.ranks.iter().flat_map(|r| &r.lane_micro).all(|&k| k == 2));
    }

    #[test]
    fn lr_scaling_grows_with_the_update_batch() {
        assert_eq!(LrScaling::None.factor(2752.0, 64.0), 1.0);
        assert!((LrScaling::Sqrt.factor(2560.0, 64.0) - 40f32.sqrt()).abs() < 1e-5);
        assert_eq!(LrScaling::Linear.factor(256.0, 64.0), 4.0);
        // Never below the spec's rate.
        assert_eq!(LrScaling::Sqrt.factor(32.0, 64.0), 1.0);
        assert_eq!("sqrt".parse::<LrScaling>(), Ok(LrScaling::Sqrt));
        assert!("cubic".parse::<LrScaling>().is_err());
    }

    #[test]
    fn a_lane_far_slower_than_its_ranks_fastest_sits_out() {
        // Rank 1's third lane (Core ML recompiling per upload) is 40× its CPU.
        let lanes = vec![vec![0.002], vec![0.004, 0.001, 0.04]];
        let p = plan_training(&lanes, 64, 1, true, 60_000).unwrap();
        let k: Vec<Vec<usize>> = p.ranks.iter().map(|r| r.lane_micro.clone()).collect();
        assert_eq!(k[1][2], 0, "the slow lane gets no work: {k:?}");
        assert!(k[1][0] >= 1 && k[1][1] >= 1);
        // The pace is the slowest *kept* lane (rank 1's 4 ms), not the 40 ms one.
        assert_eq!(k[1][0], 1, "{k:?}");
        // A rank's only lane always stays, however slow.
        let p = plan_training(&[vec![0.001], vec![0.5]], 64, 1, true, 60_000).unwrap();
        assert!(p.ranks[1].lane_micro[0] >= 1);
        // Unbalanced plans keep every lane.
        let p = plan_training(&lanes, 64, 1, false, 60_000).unwrap();
        assert!(p.ranks[1].lane_micro.iter().all(|&m| m == 1));
    }

    #[test]
    fn a_balanced_plan_shrinks_to_fit_a_small_dataset() {
        // A lane 40× faster than the other would want 81 micro-batches of 4:
        // more than 96 samples. The shares shrink together and still fit.
        let lanes = vec![vec![0.04], vec![0.001]];
        let p = plan_training(&lanes, 4, 2, true, 96).unwrap();
        assert!(p.samples_per_sync <= 96 && p.syncs_per_epoch >= 1, "{p:?}");
        let k: Vec<usize> = p.ranks.iter().flat_map(|r| r.lane_micro.clone()).collect();
        assert!(k[0] >= 1 && k[1] > 10 * k[0], "the balance survives: {k:?}");
    }

    #[test]
    fn plan_rejects_a_dataset_smaller_than_one_sync() {
        assert!(plan_training(&[vec![1.0]], 64, 4, false, 100).is_err());
    }

    #[test]
    fn options_validate_rejects_undefined_combinations() {
        let ok = TrainOptions::default();
        assert!(ok.validate().is_ok());
        let bad = |f: fn(&mut TrainOptions)| {
            let mut o = TrainOptions::default();
            f(&mut o);
            o.validate().is_err()
        };
        assert!(bad(|o| o.accum = 0));
        assert!(bad(|o| {
            o.local_steps = 4;
            o.zero = true
        }));
        assert!(bad(|o| {
            o.zero = true;
            o.wire = WireDtype::Bf16
        }));
        assert!(bad(|o| {
            o.zero = true;
            o.overlap = true
        }));
    }

    #[test]
    fn checkpoint_round_trip_and_job_mismatch() {
        let dir = std::env::temp_dir().join(format!("rlx_ckpt_test_{}", std::process::id()));
        let d = dir.to_string_lossy().to_string();
        let names = vec!["w".to_string(), "b".to_string()];
        let lens = vec![3, 1];
        save_checkpoint(&d, 2, &names, &lens, &[1.0, 2.0, 3.0, 4.0], &[0.5; 4], 0).unwrap();
        let (done, p, v, _) = load_checkpoint(&d, &names, &lens).unwrap();
        assert_eq!((done, p, v), (2, vec![1.0, 2.0, 3.0, 4.0], vec![0.5; 4]));
        assert!(
            load_checkpoint(&d, &names, &[2, 2]).is_none(),
            "shape change"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn auto_activations_pick_f16_in_range_and_bf16_outside() {
        // Small messages stay exact.
        assert_eq!(WireDtype::Auto.for_activation(&[1.0, -3e4]), WireDtype::F32);
        // Large ones narrow: f16 in range, else bf16.
        let big = |v: f32| vec![v; EXACT_ACTIVATION_BYTES / 4 + 1];
        assert_eq!(WireDtype::Auto.for_activation(&big(-3e4)), WireDtype::F16);
        assert_eq!(WireDtype::Auto.for_activation(&big(7e4)), WireDtype::Bf16);
        assert_eq!(
            WireDtype::Auto.for_activation(&big(f32::NAN)),
            WireDtype::Bf16
        );
        assert_eq!(WireDtype::F32.for_activation(&[7e4]), WireDtype::F32);
        assert_eq!(WireDtype::Bf16.round(1.0 + 2f32.powi(-9)), 1.0);
        assert_eq!(
            WireDtype::F16.round(1.0 + 2f32.powi(-9)),
            1.0 + 2f32.powi(-9)
        );
    }

    #[test]
    fn wire_dtype_parses() {
        assert_eq!("bf16".parse::<WireDtype>().unwrap(), WireDtype::Bf16);
        assert_eq!("FP16".parse::<WireDtype>().unwrap(), WireDtype::F16);
        assert!("int4".parse::<WireDtype>().is_err());
    }
}
