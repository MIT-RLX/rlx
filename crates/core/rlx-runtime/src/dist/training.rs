// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Ship-graph **training**: ship a `TrainSpec` (backward graph + data plan +
//! optimizer) to a generic worker that trains any model on its local hardware,
//! with a sample-weighted, optionally-deterministic gradient reduce.

use super::{WeightCache, WeightRef, resolve_device};
use crate::{Device, Session};
use rlx_driver::ProcessGroup;
use rlx_ir::Graph;
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════
// Ship-graph *training* — build the worker once, train any model.
//
// The inference half above ships a `StageSpec`; this half ships a `TrainSpec`:
// a serialized **backward** graph + trainable-param init URIs + a **data plan**
// + an optimizer directive. A generic worker ([`run_train`]) resolves its params
// and data node-locally (weights/data never cross the wire — only the KB-sized
// spec does), compiles on the node's chosen device, and runs the data-parallel
// training loop. Gradients are averaged across workers by a caller-supplied
// `reduce` closure, so this stays free of any collective dependency (the worker
// binary plugs `rlx-collectives` in) and the crate layering is preserved.
// ════════════════════════════════════════════════════════════════════════════

/// Wire tag for a shipped [`TrainSpec`] (clear of the collective range and the
/// inference `TAG_SPEC`/`TAG_ACT`).
const TAG_TRAIN: u32 = 42;

/// One data tensor the worker feeds to a graph input every step. Resolved
/// node-locally from `uri` (a `file://` raw-f32 blob, or gguf/safetensors),
/// then row-sharded: this rank owns samples `[shard_start, shard_start+len)`,
/// each `elem` floats wide.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DataRef {
    /// Graph input node name this tensor feeds (e.g. `"x"`, `"labels"`).
    pub input: String,
    /// Where the full (unsharded) tensor lives — resolved on the worker.
    pub uri: String,
    /// Per-sample element count (row stride into the resolved flat array).
    pub elem: usize,
    /// First sample index of this rank's shard.
    pub shard_start: usize,
    /// Number of samples in this rank's shard.
    pub shard_len: usize,
}

/// A self-describing **training** job for a generic worker: everything needed to
/// train a model the worker has no code for.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainSpec {
    /// Backward graph. Outputs are `[.. , grad_0, grad_1, …]`, with the
    /// gradients starting at `grad_start` and matching `params` 1:1.
    pub graph: Graph,
    /// Trainable params + where to fetch their initial values (node-local URI).
    pub params: Vec<WeightRef>,
    /// Index in `graph.outputs` where the gradients begin (`params[i]` ↔
    /// `outputs[grad_start + i]`).
    pub grad_start: usize,
    /// Index in `graph.outputs` of the scalar loss (for logging).
    pub loss_index: usize,
    /// Inputs fed each step from node-local data shards.
    pub data: Vec<DataRef>,
    /// Name of the backward seed input to feed `1.0` (e.g. `"d_output"`), if any.
    pub seed_input: Option<String>,
    /// SGD momentum.
    pub momentum: f32,
    /// Learning rate per epoch (`len()` = number of epochs). The master owns the
    /// schedule; the worker just applies `lr_per_epoch[epoch]`.
    pub lr_per_epoch: Vec<f32>,
    /// Mini-batch size (matches the graph's input batch dim).
    pub batch: usize,
    /// Device directive: `auto` (node's fastest), or a name (`metal`, `cuda`, …).
    pub device: String,
    /// The collective group id the worker's `reduce` closure averages over
    /// (informational — the closure owns the actual all-reduce).
    pub grad_group: u64,
    /// If set, the master **pushes** each worker its data shard over the wire
    /// (for nodes with no shared filesystem); the worker rewrites its `DataRef`s
    /// to the received local files. When false, workers resolve the URIs
    /// node-locally (shared FS / pre-staged data).
    #[serde(default)]
    pub push_data: bool,
}

/// Per-worker training result.
#[derive(Clone, Debug)]
pub struct TrainMetrics {
    /// Device the worker resolved and trained on (the primary lane).
    pub device: Device,
    /// The backends actually used this run — one for a single device, several
    /// for `device: "all"` (intra-node data-parallel across the node's hardware).
    pub lanes: Vec<Device>,
    /// Every live backend on this node (what `device: "all"` could use).
    pub inventory: Vec<Device>,
    pub samples: usize,
    pub wall_s: f64,
    /// Time in backward compute (`run()`).
    pub compute_s: f64,
    /// Time in the cross-worker gradient reduce (the `reduce` closure).
    pub comm_s: f64,
    pub first_loss: f32,
    pub last_loss: f32,
}

/// Coordinator: ship a training job to worker `rank`.
pub fn ship_train(group: &ProcessGroup, rank: u32, spec: &TrainSpec) -> Result<(), String> {
    let bytes = serde_json::to_vec(spec).map_err(|e| e.to_string())?;
    group
        .transport()
        .send_bytes(rank, TAG_TRAIN, &bytes)
        .map_err(|e| format!("ship_train: {e}"))
}

/// Worker: receive a [`TrainSpec`] from the coordinator (rank 0).
pub fn recv_train(group: &ProcessGroup) -> Result<TrainSpec, String> {
    let bytes = group
        .transport()
        .recv_bytes(0, TAG_TRAIN)
        .map_err(|e| format!("recv_train: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("TrainSpec: {e}"))
}

/// Base wire tag for pushed data shards (`TAG_DATA_BASE + input_index`).
const TAG_DATA_BASE: u32 = 45;
/// Base wire tag for pushed **init params** (`TAG_PARAM_BASE + param_index`).
const TAG_PARAM_BASE: u32 = 60;

/// Master: **push** worker `rank` everything it needs but can't resolve locally
/// — the **init params** (training-from-scratch has no node-local weights, so
/// the `WeightRef` file:// URIs only exist on the master) and each **data
/// shard** (only this rank's samples). Use when the worker has no shared
/// filesystem. Pair with [`pull_shards`] on the worker.
pub fn push_shards<W: FnMut(&str) -> Vec<f32>>(
    group: &ProcessGroup,
    rank: u32,
    spec: &TrainSpec,
    mut resolve: W,
) -> Result<(), String> {
    let mut cache = WeightCache::new();
    // Init params — small, but the whole reason cross-machine training used to
    // silently not learn: without them the worker starts from zero weights, so
    // its gradients vanish (h = relu(x·0) = 0) and it never updates.
    for (i, w) in spec.params.iter().enumerate() {
        let vals = cache.f32(&w.uri).unwrap_or_else(|_| resolve(&w.uri));
        group
            .send_f32(rank, TAG_PARAM_BASE + i as u32, &vals)
            .map_err(|e| format!("push param {} → {rank}: {e}", w.name))?;
    }
    // Data shards — only this rank's samples.
    for (i, d) in spec.data.iter().enumerate() {
        let shard = resolve_shard(
            &d.uri,
            d.elem,
            d.shard_start,
            d.shard_len,
            &mut cache,
            &mut resolve,
        );
        group
            .send_f32(rank, TAG_DATA_BASE + i as u32, &shard)
            .map_err(|e| format!("push_shards[{i}] → {rank}: {e}"))?;
    }
    Ok(())
}

/// Worker: **receive** the pushed init params + data shards from the master
/// (rank 0), write them to node-local files under `dir`, and rewrite
/// `spec.params`/`spec.data` to point at those files (data offset reset to 0).
/// After this the generic [`run_train`] resolves everything locally exactly as
/// in the shared-FS case.
pub fn pull_shards(
    group: &ProcessGroup,
    spec: &mut TrainSpec,
    dir: &std::path::Path,
) -> Result<(), String> {
    let write_bin = |name: &str, vals: &[f32]| -> Result<String, String> {
        let path = dir.join(name);
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &bytes).map_err(|e| format!("write {name}: {e}"))?;
        Ok(format!("file://{}", path.display()))
    };
    // Init params.
    for i in 0..spec.params.len() {
        let vals = group
            .recv_f32(0, TAG_PARAM_BASE + i as u32)
            .map_err(|e| format!("pull param[{i}]: {e}"))?;
        spec.params[i].uri = write_bin(&format!("rlx_pushed_param_{i}.bin"), &vals)?;
    }
    // Data shards.
    for i in 0..spec.data.len() {
        let vals = group
            .recv_f32(0, TAG_DATA_BASE + i as u32)
            .map_err(|e| format!("pull_shards[{i}]: {e}"))?;
        spec.data[i].uri = write_bin(&format!("rlx_pushed_shard_{i}.bin"), &vals)?;
        spec.data[i].shard_start = 0; // the pushed file is exactly this rank's shard
    }
    Ok(())
}

/// Pick this node's training lanes. `device:"all"` fans the backward out over
/// **every live local backend at once** — intra-node data parallelism, so the
/// CPU trains alongside the GPU on each connected node instead of sitting idle;
/// any other directive is a single lane. ANE is excluded (its training path is a
/// separate gated route).
///
/// This is safe across a **heterogeneous** cluster because [`run_train`] combines
/// gradients with a **sample-count-weighted** reduce on a **uniform sync cadence**
/// (see there): a node casts one weighted vote per local lane, so a 3-lane Mac
/// and a 1-lane CUDA box average correctly and never desync. The two failure
/// modes of naïve multi-lane DP — deadlock from unequal per-node sync counts, and
/// gradient bias from equal-weighting unequal lane counts — are handled in the
/// reduce, not by forcing every node down to a single lane.
fn training_lanes(spec_device: &str, available: &[Device], graph: &Graph) -> Vec<Device> {
    if spec_device.eq_ignore_ascii_case("all") {
        let native = crate::devices_for(graph);
        let mut ds: Vec<Device> = available
            .iter()
            .copied()
            .filter(|d| !matches!(d, Device::Ane))
            .collect();
        // Native-supporting backends first (they run without host fallback).
        ds.sort_by_key(|d| !native.contains(d));
        if ds.is_empty() {
            vec![crate::fastest_device()]
        } else {
            ds
        }
    } else {
        vec![resolve_device(spec_device)]
    }
}

/// Run one worker's full data-parallel training from a shipped [`TrainSpec`] —
/// **the generic worker's whole job, with no model code.**
///
/// * `resolve` — node-local weight/data URI → f32 (built-in schemes handled by
///   the internal [`WeightCache`]; this is the fallback for custom schemes).
/// * `reduce` — average a flat gradient buffer across all workers (the worker
///   binary implements this with `rlx-collectives`; keeps this crate
///   collective-free). Called once per step with all gradients flattened into
///   one bucket (DDP-style), and must return the same-length averaged buffer.
///   **Every rank must call `reduce` the same number of times, in lockstep** —
///   the collective is a barrier, so a rank that syncs fewer times deadlocks the
///   rest. `run_train` guarantees this by keeping each rank's per-epoch sync
///   count driven only by cluster-uniform quantities (`shard_len / batch`), not
///   by node-local hardware (see [`training_lanes`]).
///
/// `world` is the number of ranks in the cross-worker group. It gates the
/// intra-node multi-lane (`device:"all"`) path, which is single-node-only:
/// mixing it with cross-node DP on heterogeneous hardware desyncs the reduce.
///
/// Returns the metrics and the final (synced) parameters as `(name, values)`.
pub fn run_train<W, R>(
    spec: &TrainSpec,
    world: u32,
    mut resolve: W,
    mut reduce: R,
    log: bool,
) -> Result<(TrainMetrics, Vec<(String, Vec<f32>)>), String>
where
    W: FnMut(&str) -> Vec<f32>,
    R: FnMut(&[f32]) -> Vec<f32>,
{
    use std::time::Instant;

    let inventory = crate::available_devices();

    // Resolve initial params node-locally (never off the wire).
    let mut cache = WeightCache::new();
    let mut params: Vec<Vec<f32>> = spec
        .params
        .iter()
        .map(|w| cache.f32(&w.uri).unwrap_or_else(|_| resolve(&w.uri)))
        .collect();
    let names: Vec<String> = spec.params.iter().map(|w| w.name.clone()).collect();
    let mut vel: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();

    // Resolve ONLY this rank's data shard (IO opt: `file://` is read with a
    // seek to the shard offset, so the worker never loads the whole dataset).
    // `data` here is already sharded — batch `b` is `data[b*batch*elem ..]`.
    let feeds: Vec<Feed> = spec
        .data
        .iter()
        .map(|d| Feed {
            input: d.input.clone(),
            elem: d.elem,
            data: resolve_shard(
                &d.uri,
                d.elem,
                d.shard_start,
                d.shard_len,
                &mut cache,
                &mut resolve,
            ),
        })
        .collect();
    let shard_len = spec.data.first().map(|d| d.shard_len).unwrap_or(0);
    let batches = shard_len / spec.batch.max(1);

    // Device set (see [`training_lanes`]): `all` → every live backend on this
    // node at once (CPU trains alongside the GPU), else the single resolved
    // device. Multi-lane is cluster-safe here thanks to the weighted reduce below.
    let devices: Vec<Device> =
        training_lanes(&spec.device, &crate::available_devices(), &spec.graph);

    // Fail fast (identically on every rank) if the master sharded unevenly.
    if world > 1 {
        assert_uniform_shards(&mut reduce, batches)?;
    }

    let np = params.len();
    let (mut compute_s, mut comm_s) = (0.0f64, 0.0f64);
    let (mut first_loss, mut last_loss) = (f32::NAN, f32::NAN);
    let mut samples = 0usize;
    let wall = Instant::now();

    if devices.len() > 1 {
        // ── intra-node multi-lane DP: one lane per local backend (CPU + GPU),
        // each running a full independent mini-batch concurrently. Sync count is
        // fixed to `batches` (cluster-uniform), NOT `batches / lanes`, so a node
        // with more lanes never desyncs the collective — it just covers its shard
        // more times per epoch (`lanes` passes) and contributes proportionally
        // more, correctly weighted by `weighted_sync`. ──
        if log {
            let names: Vec<&str> = devices.iter().map(|d| crate::device_label(*d)).collect();
            eprintln!("  intra-node lanes engaged: {names:?} (sample-weighted DP)");
        }
        let mut lanes = spawn_lanes(spec, &devices);
        let ndev = lanes.len();
        for (epoch, &lr) in spec.lr_per_epoch.iter().enumerate() {
            let mut epoch_loss = 0.0f64;
            for s in 0..batches {
                let t = Instant::now();
                for (i, lane) in lanes.iter().enumerate() {
                    // Rotating window so the lanes cover distinct batches; over
                    // `batches` syncs each of the `ndev` lanes makes one pass, so
                    // the node makes `ndev` passes total (weighted accordingly).
                    let b = (s * ndev + i) % batches;
                    let fdata: Vec<Vec<f32>> = feeds
                        .iter()
                        .map(|f| {
                            let off = b * spec.batch * f.elem;
                            f.data[off..off + spec.batch * f.elem].to_vec()
                        })
                        .collect();
                    lane.job.send(LaneJob::Step(params.clone(), fdata)).ok();
                }
                // SUM (not mean) grads across lanes — the per-lane sample counts
                // are equal (`batch` each), so `weighted_sync` divides by the
                // cluster-wide lane total to get the true global mean gradient.
                let mut sum_grads: Vec<Vec<f32>> =
                    params.iter().map(|p| vec![0.0f32; p.len()]).collect();
                let mut loss_sum = 0.0f64;
                for lane in &lanes {
                    let (loss, grads) = lane
                        .res
                        .recv()
                        .map_err(|e| format!("lane {}: {e}", crate::device_label(lane.dev)))?;
                    loss_sum += loss as f64;
                    for i in 0..np {
                        for (a, g) in sum_grads[i].iter_mut().zip(&grads[i]) {
                            *a += g;
                        }
                    }
                }
                compute_s += t.elapsed().as_secs_f64();
                epoch_loss += loss_sum / ndev as f64;
                samples += spec.batch * ndev;
                weighted_sync(
                    &mut reduce,
                    &mut params,
                    &mut vel,
                    &sum_grads,
                    ndev,
                    spec.momentum,
                    lr,
                    &mut comm_s,
                );
            }
            record_epoch(
                epoch,
                (epoch_loss / batches.max(1) as f64) as f32,
                &mut first_loss,
                &mut last_loss,
                log,
                spec.lr_per_epoch.len(),
            );
        }
        for lane in &mut lanes {
            lane.job.send(LaneJob::Stop).ok();
        }
        for lane in lanes {
            lane.handle.join().ok();
        }
    } else {
        // ── single lane (optimized: zero-copy batch slices, reused compiled
        // graph). Contributes one weighted vote (lane_count = 1), so it combines
        // correctly with multi-lane peers on the same cluster. ──
        let mut compiled = Session::new(devices[0]).compile(spec.graph.clone());
        for (n, p) in names.iter().zip(&params) {
            compiled.set_param(n, p);
        }
        let one = [1.0f32];
        let mut sum_grads: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0f32; p.len()]).collect();
        for (epoch, &lr) in spec.lr_per_epoch.iter().enumerate() {
            let mut epoch_loss = 0.0f64;
            for b in 0..batches {
                let mut feed: Vec<(&str, &[f32])> = feeds
                    .iter()
                    .map(|f| {
                        let off = b * spec.batch * f.elem;
                        (f.input.as_str(), &f.data[off..off + spec.batch * f.elem])
                    })
                    .collect();
                if let Some(seed) = &spec.seed_input {
                    feed.push((seed.as_str(), &one));
                }
                let t = Instant::now();
                let outs = compiled.run(&feed);
                compute_s += t.elapsed().as_secs_f64();
                epoch_loss += outs[spec.loss_index][0] as f64;
                samples += spec.batch;

                for i in 0..np {
                    sum_grads[i].copy_from_slice(&outs[spec.grad_start + i]);
                }
                weighted_sync(
                    &mut reduce,
                    &mut params,
                    &mut vel,
                    &sum_grads,
                    1,
                    spec.momentum,
                    lr,
                    &mut comm_s,
                );
                for (n, p) in names.iter().zip(&params) {
                    compiled.set_param(n, p);
                }
            }
            record_epoch(
                epoch,
                (epoch_loss / batches.max(1) as f64) as f32,
                &mut first_loss,
                &mut last_loss,
                log,
                spec.lr_per_epoch.len(),
            );
        }
    }

    let metrics = TrainMetrics {
        device: crate::device_ext::fastest_among(&devices),
        lanes: devices,
        inventory,
        samples,
        wall_s: wall.elapsed().as_secs_f64(),
        compute_s,
        comm_s,
        first_loss,
        last_loss,
    };
    let final_params = names.into_iter().zip(params).collect();
    Ok((metrics, final_params))
}

/// One fed input's node-local **shard** (already sliced to this rank) + its row
/// stride, so batch `b` is `data[b*batch*elem .. (b+1)*batch*elem]`.
struct Feed {
    input: String,
    elem: usize,
    data: Vec<f32>,
}

/// Read only samples `[start, start+len)` of a tensor. For `file://` this is a
/// `seek` + bounded read (the worker never loads the whole dataset — the IO
/// win); other schemes fall back to a full resolve then slice.
fn resolve_shard<W: FnMut(&str) -> Vec<f32>>(
    uri: &str,
    elem: usize,
    start: usize,
    len: usize,
    cache: &mut WeightCache,
    resolve: &mut W,
) -> Vec<f32> {
    if let Some(path) = uri.strip_prefix("file://") {
        use std::io::{Read, Seek, SeekFrom};
        if let Ok(mut f) = std::fs::File::open(path) {
            let mut buf = vec![0u8; len * elem * 4];
            if f.seek(SeekFrom::Start((start * elem * 4) as u64)).is_ok()
                && f.read_exact(&mut buf).is_ok()
            {
                return buf
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
            }
        }
    }
    let full = cache.f32(uri).unwrap_or_else(|_| resolve(uri));
    full.get(start * elem..(start + len) * elem)
        .map(<[f32]>::to_vec)
        .unwrap_or(full)
}

/// Guard that every rank runs the same number of gradient syncs (`= batches`),
/// which the barrier-style all-reduce requires. `batches` is uniform iff the
/// master sharded evenly; we verify it cluster-wide through the `reduce` closure
/// (mean + mean-of-squares → variance) so **all** ranks return the SAME error
/// rather than some erroring while the rest deadlock waiting on them. Lane counts
/// MAY differ across nodes (the weighted reduce handles that); shard sizes may not.
fn assert_uniform_shards<R: FnMut(&[f32]) -> Vec<f32>>(
    reduce: &mut R,
    batches: usize,
) -> Result<(), String> {
    let b = batches as f32;
    let stats = reduce(&[b, b * b]);
    let mean_b = stats.first().copied().unwrap_or(b);
    let mean_b2 = stats.get(1).copied().unwrap_or(b * b);
    let var = (mean_b2 - mean_b * mean_b).max(0.0);
    if var > 0.25 {
        return Err(format!(
            "run_train: uneven shards across the cluster — this rank has {batches} batches, \
             cluster mean {mean_b:.2} (var {var:.2}). The master must ship an equal shard_len \
             (÷ batch) to every rank, else the gradient all-reduce deadlocks."
        ));
    }
    Ok(())
}

/// One data-parallel gradient sync + SGD step, with a **sample-count-weighted**
/// cross-worker reduce that makes heterogeneous lane counts safe.
///
/// `sum_grads` is the element-wise SUM of this rank's per-lane gradients (each
/// lane ran one equal-size `batch`); `lane_count` is how many lanes it ran. The
/// bucket sent to `reduce` is `[sum_grads…, lane_count]`. A Mean- (or Sum-)reduce
/// then gives `[Σ_w Σ_lane grad , Σ_w lane_count_w]` up to a shared 1/W factor,
/// and dividing the gradient part by the count part yields
/// `Σ_all grad_lane / Σ_all lane_count` — the true global mean gradient over
/// every equal-size batch in the cluster. The 1/W cancels, so a 1-lane node
/// (`lane_count = 1`) and a 3-lane node (`lane_count = 3`) combine **without
/// bias**, and because the bucket length and call count are identical on every
/// rank the collective never deadlocks.
fn weighted_sync<R: FnMut(&[f32]) -> Vec<f32>>(
    reduce: &mut R,
    params: &mut [Vec<f32>],
    vel: &mut [Vec<f32>],
    sum_grads: &[Vec<f32>],
    lane_count: usize,
    momentum: f32,
    lr: f32,
    comm_s: &mut f64,
) {
    use std::time::Instant;
    let total: usize = sum_grads.iter().map(|g| g.len()).sum();
    let mut flat = Vec::with_capacity(total + 1);
    for g in sum_grads {
        flat.extend_from_slice(g);
    }
    flat.push(lane_count as f32);
    let t = Instant::now();
    let reduced = reduce(&flat);
    *comm_s += t.elapsed().as_secs_f64();
    let denom = reduced.get(total).copied().unwrap_or(lane_count as f32);
    let inv = if denom != 0.0 { 1.0 / denom } else { 0.0 };
    apply_sgd(params, vel, &reduced, inv, momentum, lr);
}

/// Fold one epoch's mean loss into the run metrics (first/last loss) and log it —
/// the bookkeeping shared by the single-lane and multi-lane training loops.
fn record_epoch(
    epoch: usize,
    mean: f32,
    first_loss: &mut f32,
    last_loss: &mut f32,
    log: bool,
    total_epochs: usize,
) {
    if epoch == 0 {
        *first_loss = mean;
    }
    *last_loss = mean;
    if log {
        eprintln!("  epoch {}/{total_epochs}: mean loss {mean:.4}", epoch + 1);
    }
}

/// In-place momentum-SGD update, scattering a flat reduced-gradient bucket back
/// across the per-parameter tensors and applying `scale` (the weighted reduce's
/// 1/lane_total) to each gradient on the fly — no intermediate scaled copy.
fn apply_sgd(
    params: &mut [Vec<f32>],
    vel: &mut [Vec<f32>],
    reduced: &[f32],
    scale: f32,
    momentum: f32,
    lr: f32,
) {
    let mut off = 0usize;
    for (p, v) in params.iter_mut().zip(vel.iter_mut()) {
        for j in 0..p.len() {
            let g = reduced[off + j] * scale;
            v[j] = momentum * v[j] + g;
            p[j] -= lr * v[j];
        }
        off += p.len();
    }
}

/// A device lane: a thread that owns a `CompiledGraph` on one backend and runs
/// mini-batches fed over a channel (keeps the non-`Send` compiled graph
/// thread-local; only f32 buffers cross the boundary).
struct DevLane {
    dev: Device,
    job: std::sync::mpsc::Sender<LaneJob>,
    res: std::sync::mpsc::Receiver<(f32, Vec<Vec<f32>>)>,
    handle: std::thread::JoinHandle<()>,
}

enum LaneJob {
    /// (params snapshot, per-input batch data) → run one step, return grads.
    Step(Vec<Vec<f32>>, Vec<Vec<f32>>),
    Stop,
}

/// Spawn one [`DevLane`] per device, each compiling `spec.graph` on its backend.
fn spawn_lanes(spec: &TrainSpec, devices: &[Device]) -> Vec<DevLane> {
    devices
        .iter()
        .map(|&dev| {
            let (jtx, jrx) = std::sync::mpsc::channel::<LaneJob>();
            let (rtx, rrx) = std::sync::mpsc::channel();
            let graph = spec.graph.clone();
            let names: Vec<String> = spec.params.iter().map(|w| w.name.clone()).collect();
            let inputs: Vec<String> = spec.data.iter().map(|d| d.input.clone()).collect();
            let seed = spec.seed_input.clone();
            let (grad_start, loss_index, np) =
                (spec.grad_start, spec.loss_index, spec.params.len());
            let handle = std::thread::spawn(move || {
                let mut compiled = Session::new(dev).compile(graph);
                let one = [1.0f32];
                while let Ok(job) = jrx.recv() {
                    match job {
                        LaneJob::Step(params, fdata) => {
                            for (n, p) in names.iter().zip(&params) {
                                compiled.set_param(n, p);
                            }
                            let mut feed: Vec<(&str, &[f32])> = inputs
                                .iter()
                                .zip(&fdata)
                                .map(|(n, d)| (n.as_str(), d.as_slice()))
                                .collect();
                            if let Some(s) = &seed {
                                feed.push((s.as_str(), &one));
                            }
                            let outs = compiled.run(&feed);
                            let loss = outs[loss_index][0];
                            let grads: Vec<Vec<f32>> =
                                (0..np).map(|i| outs[grad_start + i].clone()).collect();
                            let _ = rtx.send((loss, grads));
                        }
                        LaneJob::Stop => break,
                    }
                }
            });
            DevLane {
                dev,
                job: jtx,
                res: rrx,
                handle,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_lanes_all_fans_out_cpu_and_gpu() {
        use rlx_ir::{DType, Graph, Shape};
        // A tiny graph so `devices_for` has an op set to inspect.
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[2, 2], DType::F32));
        let w = g.param("w", Shape::new(&[2, 2], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 2], DType::F32));
        g.set_outputs(vec![mm]);

        let avail = [Device::Cpu, Device::Metal];
        // `all` engages EVERY local backend (ANE excluded) — CPU trains alongside
        // the GPU. This holds on a cluster too: the sample-weighted reduce in
        // `run_train` keeps heterogeneous lane counts from desyncing.
        assert_eq!(training_lanes("all", &avail, &g).len(), 2);
        assert_eq!(training_lanes("all", &[Device::Cpu], &g).len(), 1);
        // An explicit device directive is always a single lane.
        assert_eq!(training_lanes("cpu", &avail, &g).len(), 1);
    }

    #[test]
    fn weighted_sync_unbiased_across_uneven_lane_counts() {
        // Two "ranks": rank A ran 3 lanes (grads summed to gA_sum), rank B ran 1
        // lane (gB). The weighted reduce must yield the global MEAN over all 4
        // equal-size batches: (gA_sum + gB) / 4 — NOT the naïve node-mean
        // (gA_sum/3 + gB) / 2, which over-weights the 1-lane node.
        let p = 2usize; // params per bucket
        let ga_sum = vec![6.0f32, 12.0]; // 3 lanes summed (e.g. lanes 1+2+3, 3+4+5)
        let gb = [10.0f32, 20.0]; // 1 lane
        // Emulate an all-reduce Mean over the two ranks' [grads…, lane_count]
        // buckets (this is exactly what ProcessGroup::all_reduce(Mean) does).
        let bucket_a = [ga_sum[0], ga_sum[1], 3.0];
        let bucket_b = [gb[0], gb[1], 1.0];
        let mean: Vec<f32> = (0..p + 1)
            .map(|i| (bucket_a[i] + bucket_b[i]) / 2.0)
            .collect();
        let mut mean_iter = mean.clone();
        let mut reduce = move |buf: &[f32]| -> Vec<f32> {
            // The buffer each rank passes is identical after a real all-reduce;
            // return the pre-computed cross-rank mean, ignoring the local buf
            // contents beyond asserting the length/protocol.
            assert_eq!(buf.len(), p + 1);
            std::mem::take(&mut mean_iter)
        };
        let mut params = vec![vec![0.0f32; p]];
        let mut vel = vec![vec![0.0f32; p]];
        let sum_grads = vec![ga_sum.clone()]; // this rank = rank A (3 lanes)
        let mut comm = 0.0;
        weighted_sync(
            &mut reduce,
            &mut params,
            &mut vel,
            &sum_grads,
            3,
            0.0,
            1.0,
            &mut comm,
        );
        // global mean grad = (gA_sum + gB) / (3 + 1)
        let expect = [(6.0 + 10.0) / 4.0, (12.0 + 20.0) / 4.0];
        // params -= lr(1.0) * grad, starting from 0 → params == -grad
        assert!(
            (params[0][0] + expect[0]).abs() < 1e-6,
            "got {:?}",
            params[0]
        );
        assert!(
            (params[0][1] + expect[1]).abs() < 1e-6,
            "got {:?}",
            params[0]
        );
    }
}
