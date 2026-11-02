// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data parallelism across the devices of one machine — a GPU, an NPU and the
//! CPU computing one step together.
//!
//! Each [`Replica`] is the same model traced at its own share of the batch and
//! pinned to one [`Device`]. A replica runs on its own long-lived thread (the
//! compile cache is per thread, and the thread keeps its compiled graph, so a
//! step after the first compiles nothing). [`DataParallel::train_step`] sends
//! every replica the current parameters and its slice of the batch, waits for
//! all of them, combines the gradients weighted by batch share — the gradient
//! of the mean loss over the whole batch — clips, and applies the optimizer
//! once on the host. [`DataParallel::run`] is the forward-only counterpart for
//! inference.
//!
//! A step takes as long as its slowest replica, so the shares should follow
//! device speed: [`split_batch`] turns measured per-sample times into shares
//! (dropping a device too slow to help),
//! and [`StepReport::secs`] gives the per-replica times to measure with.
//!
//! ```ignore
//! let replicas = vec![
//!     Replica::new(Device::Rocm, build(24).init_params(copy_from(&master)), 24),
//!     Replica::new(Device::Cpu, build(8).init_params(copy_from(&master)), 8),
//! ];
//! let dp = DataParallel::new(replicas);
//! let (next, report) = dp.train_step(&master, &mut opt, &sched, step, 1.0, feeds);
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::Instant;

use rlx_ir::Graph;
use rlx_runtime::{CompiledGraph, Device};

use crate::Func;

/// Stack for replica threads: compile passes and some backend runtimes
/// recurse once per graph node.
const WORKER_STACK: usize = 256 << 20;

/// Named host buffers fed to one replica (`feats`, `target`, …), each holding
/// that replica's slice of the batch.
pub type Feeds = Vec<(String, Vec<f32>)>;

/// One device's copy of the model, traced at its batch share.
pub struct Replica {
    /// Where this replica runs.
    pub device: Device,
    /// The model traced at `batch` samples. Its parameter names must match the
    /// master's; the bound values are replaced every step.
    pub func: Func,
    /// Samples this replica takes from each step's batch.
    pub batch: usize,
}

impl Replica {
    /// A replica of `func` (traced at `batch` samples) on `device`.
    pub fn new(device: Device, func: Func, batch: usize) -> Self {
        Self {
            device,
            func,
            batch,
        }
    }
}

enum Job {
    Grad(Arc<Vec<(String, Vec<f32>)>>, Feeds),
    Run(Arc<Vec<(String, Vec<f32>)>>, Feeds),
}

type Reply = Result<(Vec<Vec<f32>>, f64), String>;

struct Worker {
    device: Device,
    batch: usize,
    /// Replica gradient output `1 + i` → master parameter index.
    to_master: Vec<usize>,
    jobs: Option<mpsc::Sender<Job>>,
    replies: mpsc::Receiver<Reply>,
    thread: Option<JoinHandle<()>>,
}

/// Per-step outcome of [`DataParallel::train_step`].
#[derive(Clone, Debug)]
pub struct StepReport {
    /// Batch-weighted mean loss.
    pub loss: f32,
    /// Wall time of each replica's value-and-gradient run, in replica order.
    pub secs: Vec<f64>,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
}

/// Several [`Replica`]s computing one step together. Dropping it stops the
/// worker threads.
pub struct DataParallel {
    workers: Vec<Worker>,
    names: Vec<String>,
    total: usize,
}

/// Compile `graph` once on this thread and keep it, so later steps skip even
/// the cache fingerprint (which hashes the whole graph).
fn exec(
    slot: &mut Option<Rc<RefCell<CompiledGraph>>>,
    graph: &Graph,
    device: Device,
    params: &[(String, Vec<f32>)],
    feeds: &Feeds,
) -> Vec<Vec<f32>> {
    let compiled = slot
        .get_or_insert_with(|| crate::cache::compiled(graph, device))
        .clone();
    let mut c = compiled.borrow_mut();
    for (name, data) in params {
        c.set_param(name, data);
    }
    let refs: Vec<(&str, &[f32])> = feeds
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect();
    c.run(&refs)
}

fn panic_text(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "replica panicked".into())
}

impl DataParallel {
    /// Start one worker thread per replica. The gradient graph is derived on
    /// the worker, and each graph compiles on its first use.
    ///
    /// # Panics
    /// If `replicas` is empty or a replica's parameter names differ from the
    /// first replica's.
    pub fn new(replicas: Vec<Replica>) -> Self {
        assert!(!replicas.is_empty(), "DataParallel needs a replica");
        let names = replicas[0].func.param_names();
        let index: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let total = replicas.iter().map(|r| r.batch).sum();
        let workers = replicas
            .into_iter()
            .map(|r| {
                let own = r.func.param_names();
                assert_eq!(
                    own.len(),
                    names.len(),
                    "replica on {:?} has different parameters",
                    r.device
                );
                let to_master = own
                    .iter()
                    .map(|n| {
                        *index.get(n.as_str()).unwrap_or_else(|| {
                            panic!("replica on {:?}: unknown parameter {n}", r.device)
                        })
                    })
                    .collect();
                let (jobs, job_rx) = mpsc::channel::<Job>();
                let (reply_tx, replies) = mpsc::channel::<Reply>();
                let (device, func) = (r.device, r.func);
                let thread = std::thread::Builder::new()
                    .name(format!("rlx-dp-{device:?}").to_lowercase())
                    // Backends recurse per graph node in places (the HIP
                    // runtime instantiating a ~30k-node hipGraph needs ~31k
                    // frames); the 2 MB default thread stack overflows there.
                    .stack_size(WORKER_STACK)
                    .spawn(move || {
                        let grad = func.value_and_grad_all();
                        let (mut fwd_c, mut grad_c) = (None, None);
                        while let Ok(job) = job_rx.recv() {
                            let t = Instant::now();
                            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                                || match &job {
                                    Job::Grad(p, f) => {
                                        exec(&mut grad_c, grad.graph(), device, p, f)
                                    }
                                    Job::Run(p, f) => exec(&mut fwd_c, func.graph(), device, p, f),
                                },
                            ));
                            let reply = out
                                .map(|o| (o, t.elapsed().as_secs_f64()))
                                .map_err(panic_text);
                            if reply_tx.send(reply).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("spawn data-parallel worker");
                Worker {
                    device,
                    batch: r.batch,
                    to_master,
                    jobs: Some(jobs),
                    replies,
                    thread: Some(thread),
                }
            })
            .collect();
        Self {
            workers,
            names,
            total,
        }
    }

    /// Samples per step across all replicas.
    pub fn total_batch(&self) -> usize {
        self.total
    }

    /// `(device, batch)` per replica, in order.
    pub fn layout(&self) -> Vec<(Device, usize)> {
        self.workers.iter().map(|w| (w.device, w.batch)).collect()
    }

    fn dispatch(&self, model: &Func, feeds: Vec<Feeds>, grad: bool) -> Vec<(Vec<Vec<f32>>, f64)> {
        assert_eq!(feeds.len(), self.workers.len(), "one feed set per replica");
        let params = Arc::new(model.params.clone());
        for (w, f) in self.workers.iter().zip(feeds) {
            let job = if grad {
                Job::Grad(params.clone(), f)
            } else {
                Job::Run(params.clone(), f)
            };
            w.jobs
                .as_ref()
                .expect("worker running")
                .send(job)
                .unwrap_or_else(|_| panic!("replica on {:?} stopped", w.device));
        }
        // Collect every reply before reporting a failure, so no worker is
        // left holding a reply for the next step.
        let replies: Vec<Reply> = self
            .workers
            .iter()
            .map(|w| {
                w.replies
                    .recv()
                    .unwrap_or_else(|_| Err(format!("replica on {:?} stopped", w.device)))
            })
            .collect();
        replies
            .into_iter()
            .zip(&self.workers)
            .map(|(r, w)| r.unwrap_or_else(|e| panic!("replica on {:?}: {e}", w.device)))
            .collect()
    }

    /// Forward pass on every replica at once: each replica's outputs for its
    /// slice, in replica order. `model` supplies the parameter values.
    pub fn run(&self, model: &Func, feeds: Vec<Feeds>) -> Vec<Vec<Vec<f32>>> {
        self.dispatch(model, feeds, false)
            .into_iter()
            .map(|(o, _)| o)
            .collect()
    }

    /// Batch-weighted loss and gradients of `model`'s parameters (master
    /// order, [`Func::param_names`]): `[loss, ∂p…]`, plus per-replica times.
    pub fn value_and_grad(&self, model: &Func, feeds: Vec<Feeds>) -> (Vec<Vec<f32>>, Vec<f64>) {
        let names = model.param_names();
        assert_eq!(
            names, self.names,
            "model parameters do not match the replicas"
        );
        let results = self.dispatch(model, feeds, true);
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(names.len() + 1);
        out.push(vec![0.0]);
        for name in &names {
            out.push(vec![0.0; model.param_binding(name).map_or(0, <[f32]>::len)]);
        }
        let mut secs = Vec::with_capacity(results.len());
        for ((outputs, t), w) in results.into_iter().zip(&self.workers) {
            let share = w.batch as f32 / self.total as f32;
            out[0][0] += share * outputs[0].first().copied().unwrap_or(f32::NAN);
            for (i, g) in outputs.into_iter().skip(1).enumerate() {
                let dst = &mut out[1 + w.to_master[i]];
                assert_eq!(
                    dst.len(),
                    g.len(),
                    "gradient {} size",
                    names[w.to_master[i]]
                );
                for (d, v) in dst.iter_mut().zip(g) {
                    *d += share * v;
                }
            }
            secs.push(t);
        }
        (out, secs)
    }

    /// One optimizer step over the whole batch: per-replica value and
    /// gradients in parallel, batch-weighted combine, global-L2 clipping to
    /// `max_grad_norm` (≤ 0 disables), then `opt` on the host. Returns the
    /// updated model, like [`Func::train_step_all_at_on_clipped`].
    pub fn train_step(
        &self,
        model: &Func,
        opt: &mut dyn rlx_optim::Optimizer,
        schedule: &crate::LrSchedule,
        step: usize,
        max_grad_norm: f32,
        feeds: Vec<Feeds>,
    ) -> (Func, StepReport) {
        let (outputs, secs) = self.value_and_grad(model, feeds);
        let (next, loss, grad_norm) =
            apply_gradients(model, opt, schedule, step, max_grad_norm, outputs);
        (
            next,
            StepReport {
                loss,
                secs,
                grad_norm,
            },
        )
    }
}

/// The optimizer half of a step, for gradients combined elsewhere (summed
/// across machines, say): `outputs` is `[loss, ∂p…]` in [`Func::param_names`]
/// order. Clips to the global L2 norm `max_grad_norm` (≤ 0 disables), sets the
/// scheduled learning rate and applies `opt`. Returns the updated model, the
/// loss and the gradient norm before clipping. Deterministic: ranks that apply
/// the same gradients to the same parameters stay identical.
pub fn apply_gradients(
    model: &Func,
    opt: &mut dyn rlx_optim::Optimizer,
    schedule: &crate::LrSchedule,
    step: usize,
    max_grad_norm: f32,
    mut outputs: Vec<Vec<f32>>,
) -> (Func, f32, f32) {
    opt.set_lr(schedule.lr_at(step));
    let grad_norm = outputs
        .iter()
        .skip(1)
        .flatten()
        .map(|&g| f64::from(g) * f64::from(g))
        .sum::<f64>()
        .sqrt() as f32;
    let scale = crate::transform::grad_clip_scale(&outputs, max_grad_norm);
    if scale < 1.0 {
        for g in outputs.iter_mut().skip(1) {
            for x in g.iter_mut() {
                *x *= scale;
            }
        }
    }
    let names = model.param_names();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let (next, loss) = model.apply_optimizer_step(opt, &refs, &outputs);
    (next, loss.first().copied().unwrap_or(f32::NAN), grad_norm)
}

impl Drop for DataParallel {
    fn drop(&mut self) {
        for w in &mut self.workers {
            w.jobs.take();
        }
        for w in &mut self.workers {
            if let Some(t) = w.thread.take() {
                let _ = t.join();
            }
        }
    }
}

/// Split `total` samples across devices to minimize the step time, given
/// each device's measured seconds per sample: each sample goes to the device
/// that would finish it soonest (`(share + 1) · secs`). A device too slow to
/// help gets `0` — drop it from the replicas.
///
/// # Panics
/// If `secs_per_sample` is empty or a time is not positive and finite.
pub fn split_batch(total: usize, secs_per_sample: &[f64]) -> Vec<usize> {
    assert!(!secs_per_sample.is_empty(), "split_batch: no devices");
    assert!(
        secs_per_sample.iter().all(|s| s.is_finite() && *s > 0.0),
        "split_batch: times must be positive"
    );
    let mut out = vec![0usize; secs_per_sample.len()];
    for _ in 0..total {
        let (best, _) = secs_per_sample
            .iter()
            .enumerate()
            .map(|(i, s)| (i, (out[i] + 1) as f64 * s))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty");
        out[best] += 1;
    }
    out
}

/// Most neighbouring splits [`tune`] measures after the predicted one.
pub const MAX_PROBES: usize = 8;

/// Outcome of [`tune`]: the fastest layout found and what was measured.
pub struct Tuned {
    /// Replicas in the chosen layout.
    pub dp: DataParallel,
    /// `(device, batch)` per replica of `dp`.
    pub layout: Vec<(Device, usize)>,
    /// Every layout tried, with its seconds per batch.
    pub tried: Vec<(Vec<(Device, usize)>, f64)>,
}

/// Pick the fastest way to spread `total` samples over `devices` by measuring:
/// each device alone with the whole batch, then the [`split_batch`] layout from
/// those times, then up to [`MAX_PROBES`] neighbouring splits (samples moved
/// between devices, kept while faster); the fastest layout measured wins. The
/// predicted split alone is not trusted because a device's time is not
/// proportional to its share (fixed per-run overhead, and a CPU replica
/// competes for cores with the host side of the GPU and NPU backends).
///
/// `build(device, batch)` traces a replica; `measure` runs one layout and
/// returns seconds per batch (warm it up first — the first run compiles).
pub fn tune(
    devices: &[Device],
    total: usize,
    mut build: impl FnMut(Device, usize) -> Func,
    mut measure: impl FnMut(&DataParallel, &[usize]) -> f64,
) -> Tuned {
    assert!(!devices.is_empty(), "tune: no devices");
    let mut tried = Vec::new();
    let mut alone = Vec::with_capacity(devices.len());
    let mut best: Option<(DataParallel, Vec<(Device, usize)>, f64)> = None;
    for &d in devices {
        let dp = DataParallel::new(vec![Replica::new(d, build(d, total), total)]);
        let t = measure(&dp, &[total]);
        tried.push((vec![(d, total)], t));
        alone.push(t / total as f64);
        if best.as_ref().is_none_or(|b| t < b.2) {
            best = Some((dp, vec![(d, total)], t));
        }
    }
    let shares = split_batch(total, &alone);
    let layout: Vec<(Device, usize)> = devices
        .iter()
        .zip(&shares)
        .filter(|(_, b)| **b > 0)
        .map(|(d, b)| (*d, *b))
        .collect();
    if layout.len() > 1 {
        let mut try_layout = |layout: &[(Device, usize)],
                              tried: &mut Vec<(Vec<(Device, usize)>, f64)>,
                              best: &mut Option<(DataParallel, Vec<(Device, usize)>, f64)>|
         -> f64 {
            let dp = DataParallel::new(
                layout
                    .iter()
                    .map(|&(d, b)| Replica::new(d, build(d, b), b))
                    .collect(),
            );
            let bs: Vec<usize> = layout.iter().map(|l| l.1).collect();
            let t = measure(&dp, &bs);
            tried.push((layout.to_vec(), t));
            if best.as_ref().is_none_or(|b| t < b.2) {
                *best = Some((dp, layout.to_vec(), t));
            }
            t
        };
        // Walk from the predicted split: move `step` samples from one device
        // to another, keep a move while it is faster, then halve the step.
        let mut cur = layout;
        let mut cur_t = try_layout(&cur, &mut tried, &mut best);
        let mut step = (total / 16).max(1);
        let mut probes = 0;
        while step > 0 && probes < MAX_PROBES {
            let mut moved = false;
            'pairs: for from in 0..cur.len() {
                for to in 0..cur.len() {
                    if from == to || cur[from].1 <= step || probes >= MAX_PROBES {
                        continue;
                    }
                    let mut next = cur.clone();
                    next[from].1 -= step;
                    next[to].1 += step;
                    if tried.iter().any(|(l, _)| *l == next) {
                        continue;
                    }
                    probes += 1;
                    let t = try_layout(&next, &mut tried, &mut best);
                    if t < cur_t {
                        (cur, cur_t, moved) = (next, t, true);
                        break 'pairs;
                    }
                }
            }
            if !moved {
                step /= 2;
            }
        }
    }
    let (dp, layout, _) = best.expect("at least one layout");
    Tuned { dp, layout, tried }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_follows_speed() {
        assert_eq!(split_batch(32, &[1.0, 1.0]), vec![16, 16]);
        assert_eq!(split_batch(32, &[1.0, 3.0]), vec![24, 8]);
        // A device 10x slower than the rest of the step is dropped.
        assert_eq!(split_batch(8, &[0.05, 0.5]), vec![8, 0]);
        assert_eq!(split_batch(3, &[1.0, 1.0, 1.0]), vec![1, 1, 1]);
        let s = split_batch(33, &[0.5, 1.0, 2.0]);
        assert_eq!(s.iter().sum::<usize>(), 33);
        assert!(s[0] > s[1] && s[1] > s[2]);
    }
}
