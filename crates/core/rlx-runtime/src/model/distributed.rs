// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Train or run **any package function** across a cluster, with nothing
//! about the model it is.
//!
//! - **Training** ([`train_job`]): the package trainer's preparation
//!   ([`super::trainer::prepare`]: the function, its dims, a loss, which
//!   params — all, a glob, LoRA, an SAE) differentiated into a
//!   [`TrainSpec`] that [`crate::dist::run_train_group`] runs on every rank:
//!   data-parallel, the dataset sliced per rank, every cluster option
//!   (accumulation, wire, balance, local SGD, ZeRO, number format,
//!   checkpoints), AdamW as the package asks. Rank 0 holds the package and the
//!   dataset and sends the rest everything they need.
//! - **Inference** ([`InferJob`]): the function split into contiguous stages
//!   balanced by estimated FLOPs (and each rank's speed), one per rank; each
//!   stage's weights are sent once, then any number of requests stream
//!   through, a stage forwarding only the values later stages read. With an
//!   LM's `forward`, [`InferJob::generate`] decodes greedily.
//! - **Workers** run [`serve`]: they wait for rank 0's job and do it — no
//!   model code, no package on disk.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result, bail};
use rlx_driver::ProcessGroup;
use rlx_ir::{Graph, NodeId, Op};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::rlxp::{Opened, params_of};
use super::trainer::{Prepared, prepare, specialize_function};
use crate::dist::{
    DataRef, TrainOptions, TrainSpec, WeightRef, WireDtype, recv_activation_any, send_activation_as,
};
use crate::{CompileOptions, Device, Session};

const TAG_JOB: u32 = 90;
const TAG_STAGE: u32 = 91;
const TAG_VALUES: u32 = 92;
const TAG_TYPED: u32 = 93;
/// A step input no graph reads: run under it, a stage holds its state update
/// until the next step's `KEEP` says how many of its rows to keep (all of
/// them, or 0 to drop it) — how a speculative step is taken back.
pub const KEEP: &str = "__keep";

/// Each stage adds its compute seconds to the values it passes on, under
/// this prefix and its index, so rank 0 sees every stage's time each step.
pub const TIME: &str = "__time.";

/// Sent down the pipeline by [`InferJob::calibrate`]: each worker passes it
/// on, then takes a new stage (plan and weights) from rank 0.
const RESTAGE: &str = "__restage";

/// A function of a package, as [`describe`] reports it.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionInfo {
    pub name: String,
    /// `(name, dtype, dims)`; a symbolic dim is its symbol's name.
    pub inputs: Vec<(String, String, Vec<String>)>,
    pub outputs: Vec<String>,
    pub params: usize,
    pub param_bytes: usize,
}

/// The functions of the package at `path` and the symbols they are sized by.
pub fn describe(path: &str) -> Result<(Vec<FunctionInfo>, Vec<String>)> {
    let pkg = Opened::open_merged(std::path::Path::new(path))?;
    let sym_name = |id: u32| {
        pkg.dims
            .iter()
            .find(|(_, s)| *s == id)
            .map(|(n, _)| n.clone())
            .unwrap_or_else(|| format!("s{id}"))
    };
    let dims_of = |s: &rlx_ir::Shape| -> Vec<String> {
        s.dims()
            .iter()
            .map(|d| match d {
                rlx_ir::Dim::Static(n) => n.to_string(),
                rlx_ir::Dim::Dynamic(sym) => sym_name(*sym),
            })
            .collect()
    };
    let infos = pkg
        .graphs
        .iter()
        .map(|g| {
            let inputs = g
                .nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Input { name } => Some((
                        name.clone(),
                        format!("{:?}", n.shape.dtype()).to_lowercase(),
                        dims_of(&n.shape),
                    )),
                    _ => None,
                })
                .collect();
            let names: HashMap<usize, String> = g.output_names().iter().cloned().collect();
            let outputs = (0..g.outputs.len())
                .map(|i| names.get(&i).cloned().unwrap_or_else(|| format!("#{i}")))
                .collect();
            let ps = params_of(g);
            FunctionInfo {
                name: g.name.split(':').next().unwrap_or(&g.name).to_string(),
                inputs,
                outputs,
                params: ps.len(),
                param_bytes: ps
                    .iter()
                    .map(|(_, dt, d)| d.iter().product::<usize>() * dt.size_bytes())
                    .sum(),
            }
        })
        .collect();
    Ok((infos, pkg.dims.iter().map(|(n, _)| n.clone()).collect()))
}

/// `"seq=64,batch=4"` → `{seq: 64, batch: 4}`.
pub fn parse_dims(s: &str) -> Result<Map<String, Value>> {
    let mut m = Map::new();
    for kv in s.split(',').map(str::trim).filter(|kv| !kv.is_empty()) {
        let (k, v) = kv
            .split_once('=')
            .with_context(|| format!("dim `{kv}`: expected name=size"))?;
        let n: u64 = v
            .trim()
            .parse()
            .with_context(|| format!("dim `{kv}`: size is not a number"))?;
        m.insert(k.trim().to_string(), json!(n));
    }
    Ok(m)
}

// ── training ────────────────────────────────────────────────────────────────

/// A package training job: the spec every rank runs, and on rank 0 the
/// values it hands out and what saves the result.
pub struct TrainJob {
    pub spec: TrainSpec,
    prepared: Prepared,
    data: HashMap<String, Vec<f32>>,
    /// Samples in the dataset.
    pub samples: usize,
}

/// Build the job that trains `path` as `options` says (the package trainer's
/// options: `function`, `dims`, `loss`, `train`, `optimizer` — see
/// [`super::trainer`]) on the safetensors dataset `data` (a tensor per
/// function input, samples along its first axis) for `epochs` epochs, every
/// rank on `device` (`auto`, `all`, a name, or a list). `cluster` carries the
/// distributed options; its optimizer becomes AdamW (from `options.optimizer`)
/// unless `options.optimizer.kind` is `sgd`.
pub fn train_job(
    path: &str,
    options: &Value,
    data: &str,
    epochs: usize,
    device: &str,
    cluster: TrainOptions,
) -> Result<TrainJob> {
    let prepared = prepare(path, options)?;
    let mut cache = crate::dist::WeightCache::new();
    job_from(
        prepared,
        |name| {
            cache
                .f32(&format!("safetensors://{data}#{name}"))
                .map_err(|e| anyhow::anyhow!("dataset {data}: input `{name}`: {e}"))
        },
        epochs,
        device,
        cluster,
    )
}

/// [`train_job`] on plain text: the package's tokenizer turns `text` into
/// next-token samples, one per window of the token input's length, every
/// `stride` tokens. The function's token input (its first largest) gets the window;
/// a one-value input (a position) the window's last index; a window-long one
/// (a mask) ones. The loss's labels get the token after the window, or the
/// window shifted by one when they are window-long.
pub fn train_job_text(
    path: &str,
    options: &Value,
    text: &str,
    stride: usize,
    epochs: usize,
    device: &str,
    cluster: TrainOptions,
) -> Result<TrainJob> {
    let ids = package_codec(path)?.encode(text)?;
    let prepared = prepare(path, options)?;
    let function = options.get("function").and_then(Value::as_str);
    let (infos, _) = describe(path)?;
    let forward: HashSet<String> = infos
        .iter()
        .find(|f| function.is_none_or(|n| f.name == n))
        .map(|f| f.inputs.iter().map(|(n, _, _)| n.clone()).collect())
        .unwrap_or_default();
    let values = text_samples(&prepared.inputs, &forward, &ids, stride)?;
    job_from(
        prepared,
        |name| {
            values
                .get(name)
                .cloned()
                .with_context(|| format!("no text samples for input `{name}`"))
        },
        epochs,
        device,
        cluster,
    )
}

/// Next-token samples from `ids` for graph `inputs` (name, dims with the
/// batch first); `forward` names the function's own inputs (the rest are
/// the loss's labels). See [`train_job_text`].
pub fn text_samples(
    inputs: &[(String, Vec<usize>)],
    forward: &HashSet<String>,
    ids: &[u32],
    stride: usize,
) -> Result<HashMap<String, Vec<f32>>> {
    let elem = |d: &[usize]| d[1..].iter().product::<usize>().max(1);
    let (tok, seq) = inputs
        .iter()
        .filter(|(n, _)| forward.contains(n))
        .map(|(n, d)| (n.clone(), elem(d)))
        // The first of the largest: token ids come before a same-length mask.
        .reduce(|a, b| if b.1 > a.1 { b } else { a })
        .context("the function has no token input")?;
    let stride = stride.max(1);
    let starts: Vec<usize> = (0..ids.len().saturating_sub(seq)).step_by(stride).collect();
    if starts.is_empty() {
        bail!(
            "the text is {} tokens: fewer than one window of {seq} plus its next token",
            ids.len()
        );
    }
    let mut out = HashMap::new();
    for (name, dims) in inputs {
        let e = elem(dims);
        let v: Vec<f32> = if *name == tok {
            starts
                .iter()
                .flat_map(|&i| ids[i..i + seq].iter().map(|&t| t as f32))
                .collect()
        } else if forward.contains(name) {
            match e {
                1 => vec![(seq - 1) as f32; starts.len()],
                _ if e == seq => vec![1.0; starts.len() * seq],
                _ => bail!(
                    "input `{name}` ({e} values per sample) is neither a position,                      a mask nor the tokens: pass a dataset file instead"
                ),
            }
        } else if e == 1 {
            starts.iter().map(|&i| ids[i + seq] as f32).collect()
        } else if e == seq {
            starts
                .iter()
                .flat_map(|&i| ids[i + 1..i + seq + 1].iter().map(|&t| t as f32))
                .collect()
        } else {
            bail!("labels `{name}` have {e} values per sample; the window is {seq}");
        };
        out.insert(name.clone(), v);
    }
    Ok(out)
}

/// The package's tokenizer: the one its first entry naming one points at
/// (`entries.<name>.tokenizer`). Needs a tokenizer installed
/// ([`super::set_tokenizer_loader`]).
pub fn package_codec(path: &str) -> Result<Box<dyn super::TextCodec>> {
    let pkg = Opened::open_merged(std::path::Path::new(path))?;
    let reference = super::rlxp::entries_of(&pkg.meta)
        .into_iter()
        .find_map(|(_, _, e)| e.get("tokenizer").and_then(Value::as_str).map(String::from))
        .with_context(|| format!("{path} names no tokenizer (entries.*.tokenizer)"))?;
    super::load_tokenizer(&pkg.file(&reference)?, &Value::Null)
}

/// The job from prepared options and a loader of each input's samples.
fn job_from(
    prepared: Prepared,
    mut load: impl FnMut(&str) -> Result<Vec<f32>>,
    epochs: usize,
    device: &str,
    mut cluster: TrainOptions,
) -> Result<TrainJob> {
    let g = &prepared.graph;
    // Differentiate w.r.t. the trained params (each name's first node).
    let mut wrt: Vec<NodeId> = Vec::new();
    for name in &prepared.trainable {
        let id = g
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Param { name: p } if p == name))
            .map(|n| n.id)
            .with_context(|| format!("trained param {name} is not in the graph"))?;
        wrt.push(id);
    }
    let backward = rlx_opt::autodiff::grad_with_loss(g, &wrt);
    let trained: HashSet<&String> = prepared.trainable.iter().collect();
    let frozen: Vec<WeightRef> = params_of(&backward)
        .into_iter()
        .filter(|(n, _, _)| !trained.contains(n))
        .map(|(name, _, _)| WeightRef {
            uri: format!("init://{name}"),
            name,
            packed: false,
        })
        .collect();
    let params: Vec<WeightRef> = prepared
        .trainable
        .iter()
        .map(|name| WeightRef {
            name: name.clone(),
            uri: format!("init://{name}"),
            packed: false,
        })
        .collect();

    // The dataset: one tensor per input, samples first.
    let mut values = HashMap::new();
    let mut refs = Vec::new();
    let (mut batch, mut samples) = (0usize, usize::MAX);
    for (name, dims) in &prepared.inputs {
        let (b, elem) = (dims[0], dims[1..].iter().product::<usize>().max(1));
        if batch != 0 && b != batch {
            bail!("inputs disagree on the batch (first) dim: {b} vs {batch}");
        }
        batch = b;
        let v = load(name)?;
        if v.len() % elem != 0 {
            bail!(
                "dataset `{name}` has {} values, not a multiple of one sample's {elem}",
                v.len()
            );
        }
        samples = samples.min(v.len() / elem);
        refs.push((name.clone(), elem));
        values.insert(name.clone(), v);
    }
    let data_refs = refs
        .into_iter()
        .map(|(name, elem)| DataRef {
            uri: format!("data://{name}"),
            input: name,
            elem,
            shard_start: 0,
            shard_len: samples,
        })
        .collect();

    let opt = &prepared.optimizer;
    let num = |k: &str, d: f32| opt.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32);
    let lr = num("lr", 1e-4);
    if opt.get("kind").and_then(Value::as_str) != Some("sgd") {
        cluster.optimizer = crate::dist::Optimizer::AdamW {
            beta1: num("beta1", 0.9),
            beta2: num("beta2", 0.999),
            eps: num("eps", 1e-8),
            weight_decay: num("weight_decay", 0.0),
        };
    }
    let spec = TrainSpec {
        graph: backward,
        params,
        grad_start: 1,
        loss_index: 0,
        data: data_refs,
        seed_input: Some("d_output".into()),
        momentum: num("momentum", 0.9),
        lr_per_epoch: vec![lr; epochs.max(1)],
        batch,
        device: device.into(),
        grad_group: 0,
        push_data: true,
        options: Some(cluster),
        frozen,
    };
    Ok(TrainJob {
        spec,
        prepared,
        data: values,
        samples,
    })
}

impl TrainJob {
    /// Rank 0 (the coordinator): send every worker the job, train its share
    /// with them, and return the metrics and the trained params.
    pub fn run(
        &self,
        group: &ProcessGroup,
        log: bool,
    ) -> Result<(crate::dist::TrainMetrics, HashMap<String, Vec<f32>>)> {
        announce(group, "train")?;
        for r in 1..group.world_size() {
            crate::dist::ship_train(group, r, &self.spec).map_err(anyhow::Error::msg)?;
        }
        let initial = &self.prepared.initial;
        let data = &self.data;
        let resolve = |uri: &str| -> Vec<f32> {
            if let Some(n) = uri.strip_prefix("init://") {
                initial.get(n).cloned().unwrap_or_default()
            } else if let Some(n) = uri.strip_prefix("data://") {
                data.get(n).cloned().unwrap_or_default()
            } else {
                Vec::new()
            }
        };
        let (m, params) = crate::dist::run_train_group(&self.spec, group, resolve, log)
            .map_err(anyhow::Error::msg)?;
        Ok((m, params.into_iter().collect()))
    }

    /// Write `trained` as the package trainer writes it (the package with the
    /// new values, a LoRA adapter, or an SAE package) at `out`.
    pub fn save(
        &self,
        trained: &HashMap<String, Vec<f32>>,
        out: &std::path::Path,
        name: &str,
    ) -> Result<()> {
        self.prepared.save(trained, out, name)
    }

    /// The loss of the trained function's first `batch` samples with `params`
    /// in place of what the package had, on `device` — to compare before and
    /// after.
    pub fn loss(&self, params: &HashMap<String, Vec<f32>>, device: Device) -> f32 {
        let mut c = Session::new(device).compile(self.prepared.graph.clone());
        for (n, v) in &self.prepared.initial {
            c.set_param(n, params.get(n).unwrap_or(v));
        }
        let feeds: Vec<(String, Vec<f32>)> = self
            .prepared
            .inputs
            .iter()
            .map(|(n, d)| {
                let len: usize = d.iter().product();
                (n.clone(), self.data[n][..len].to_vec())
            })
            .collect();
        let f: Vec<(&str, &[f32])> = feeds.iter().map(|(n, v)| (n.as_str(), &v[..])).collect();
        c.run(&f)[0].first().copied().unwrap_or(f32::NAN)
    }

    /// The mean loss over the first `batches` batches of this job's data with
    /// `params` (missing ones at their initial values) — e.g. a held-out set:
    /// build a job on held-out text ([`train_job_text`]) and evaluate the
    /// params another job trained. `exp` of it is the perplexity for an LM loss.
    pub fn eval_loss(
        &self,
        params: &HashMap<String, Vec<f32>>,
        device: Device,
        batches: usize,
    ) -> f32 {
        let mut c = Session::new(device).compile(self.prepared.graph.clone());
        for (n, v) in &self.prepared.initial {
            c.set_param(n, params.get(n).unwrap_or(v));
        }
        let per_batch = |d: &[usize]| d.iter().product::<usize>();
        let batch = self.spec.batch.max(1);
        let n = (self.samples / batch).min(batches.max(1));
        let mut sum = 0.0f64;
        for b in 0..n {
            let feeds: Vec<(String, Vec<f32>)> = self
                .prepared
                .inputs
                .iter()
                .map(|(name, d)| {
                    let len = per_batch(d);
                    (
                        name.clone(),
                        self.data[name][b * len..(b + 1) * len].to_vec(),
                    )
                })
                .collect();
            let f: Vec<(&str, &[f32])> = feeds.iter().map(|(n, v)| (n.as_str(), &v[..])).collect();
            sum += c.run(&f)[0].first().copied().unwrap_or(f32::NAN) as f64;
        }
        (sum / n.max(1) as f64) as f32
    }
}

// ── inference ───────────────────────────────────────────────────────────────

/// One rank's share of a function: its subgraph and how values reach it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePlan {
    pub graph: Graph,
    /// Values the stage reads (its subgraph inputs, by name).
    pub reads: Vec<String>,
    /// Params it binds.
    pub params: Vec<String>,
    /// Values it produces (its subgraph outputs, in order).
    pub writes: Vec<String>,
    /// Values handed to the next rank afterwards.
    pub carry: Vec<String>,
    /// Number format the stage compiles in.
    #[serde(default)]
    pub number_format: Option<String>,
    #[serde(default)]
    pub device: String,
    /// Width the stage sends its values in (rank 0's choice).
    #[serde(default)]
    pub wire: WireDtype,
    /// Recurrent state this stage keeps for itself between steps (a decode
    /// step's KV caches, gated delta-net states): each layer's state is read
    /// and rewritten by the stage that holds the layer, so it never crosses
    /// the network.
    #[serde(default)]
    pub state: Vec<StateSlot>,
    /// The ranks this stage receives from and sends to; `None`: the ones
    /// before and after it in rank order.
    #[serde(default)]
    pub prev: Option<u32>,
    #[serde(default)]
    pub next: Option<u32>,
    /// This stage's place in the pipeline: the key its compute time travels
    /// under ([`TIME`]).
    #[serde(default)]
    pub index: usize,
    /// The package the weights come from, as `path:bytes:mtime`: a worker that
    /// still holds this plan from an earlier job (same bytes) skips them.
    #[serde(default)]
    pub weights_key: String,
    /// The speeds and pipeline order the whole step was laid out by. A
    /// persistent worker remembers them, and the next job plans by them while
    /// the ranks are steady ([`exchange_speeds_links`]): the same plan again,
    /// so the worker keeps its stage, calibrated layouts included.
    #[serde(default)]
    pub layout: Option<PlannedLayout>,
}

/// What a step was laid out by: each rank's speed (GFLOP/s), and the ranks in
/// pipeline order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedLayout {
    pub speeds: Vec<f64>,
    pub order: Vec<u32>,
}

/// One stage-local state: the input it feeds and the output that updates it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSlot {
    pub input: String,
    pub output: String,
    /// `append`: the output holds the cache's rows then the new row last, and
    /// the next cache is the first `past` rows plus that row (a KV cache).
    /// Otherwise the output replaces the state (recurrent state).
    pub append: bool,
    /// Elements the input takes (the cache is zero-padded to it).
    pub numel: usize,
    /// Elements in one cache row (`append` only).
    pub width: usize,
}

/// Rough FLOPs of a node: a matmul's `2·m·k·n` (from its shapes), else its
/// output's element count.
fn node_cost(g: &Graph, n: &rlx_ir::Node) -> f64 {
    let numel = |s: &rlx_ir::Shape| s.num_elements().unwrap_or(0) as f64;
    match &n.op {
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => 0.0,
        Op::MatMul => {
            let k = g
                .shape(n.inputs[0])
                .dims()
                .last()
                .map(|d| d.unwrap_static())
                .unwrap_or(1);
            2.0 * numel(&n.shape) * k as f64
        }
        Op::Attention { .. } => {
            // ~4·(rows)·(keys)·d: two products over the keys.
            let q = numel(g.shape(n.inputs[0]));
            let kv = g.shape(n.inputs[1]).dims();
            let keys = kv
                .get(kv.len().saturating_sub(2))
                .map(|d| d.unwrap_static())
                .unwrap_or(1);
            4.0 * q * keys as f64
        }
        _ => numel(&n.shape),
    }
}

fn value_name(g: &Graph, id: NodeId) -> String {
    match &g.node(id).op {
        Op::Input { name } => name.clone(),
        _ => format!("v{}", id.0),
    }
}

/// Split `graph` into `weights.len()` contiguous stages, stage `r` getting
/// a share of the estimated FLOPs proportional to `weights[r]` (a rank's
/// speed). Params and constants are copied into each stage that reads them;
/// every value crossing a cut becomes a named input.
pub fn split_function(graph: &Graph, weights: &[f64]) -> Result<Vec<StagePlan>> {
    split_function_keeping(graph, weights, &[])
}

/// The node-id span of each stage-local state `(input, output index, _)`:
/// from its first reader to the update it is replaced by and that update's
/// last reader. A cut inside one
/// would put a layer's read of its state and its write on two ranks.
fn state_spans(graph: &Graph, state: &[(String, usize, bool)]) -> Vec<(usize, usize)> {
    state
        .iter()
        .filter_map(|(input, idx, _)| {
            let inp = graph
                .nodes()
                .iter()
                .find(|nd| matches!(&nd.op, Op::Input { name } if name == input))?
                .id;
            let out_id = *graph.outputs.get(*idx)?;
            // The update is often read in its layer too (attention over the
            // appended cache): its readers stay with it, since the stage
            // keeps the update rather than sending it on.
            let ids = graph
                .nodes()
                .iter()
                .filter(|nd| nd.inputs.contains(&inp) || nd.inputs.contains(&out_id))
                .map(|nd| nd.id.0 as usize)
                .chain([out_id.0 as usize]);
            let (lo, hi) = ids.fold((usize::MAX, 0), |(lo, hi), i| (lo.min(i), hi.max(i)));
            Some((lo, hi))
        })
        .collect()
}

/// [`split_function`], never cutting inside one of `together`'s node-id
/// spans: a stage grows past its share until the span closes.
pub fn split_function_keeping(
    graph: &Graph,
    weights: &[f64],
    together: &[(usize, usize)],
) -> Result<Vec<StagePlan>> {
    split_function_cut(graph, weights, together, &HashSet::new())
}

/// How far a cut may move from its share's point, as a fraction of the
/// stage's share, to where fewer bytes cross.
const CUT_WINDOW: f64 = 0.25;

/// The bytes (f32) crossing a cut before each node: every value made before
/// it and read at or after it, but params and constants (each stage holds its
/// own) and `local` values (stage-local state). A graph output crosses every
/// cut after it.
fn crossing_bytes(graph: &Graph, local: &HashSet<String>) -> Vec<f64> {
    let n = graph.len();
    let mut last = vec![0usize; n];
    for nd in graph.nodes() {
        for &i in &nd.inputs {
            last[i.0 as usize] = last[i.0 as usize].max(nd.id.0 as usize);
        }
    }
    for &o in &graph.outputs {
        last[o.0 as usize] = n;
    }
    let mut diff = vec![0f64; n + 2];
    for nd in graph.nodes() {
        let j = nd.id.0 as usize;
        if matches!(nd.op, Op::Param { .. } | Op::Constant { .. })
            || last[j] <= j
            || local.contains(&value_name(graph, nd.id))
        {
            continue;
        }
        let bytes = 4.0 * nd.shape.num_elements().unwrap_or(0) as f64;
        diff[j + 1] += bytes;
        diff[last[j].min(n) + 1] -= bytes;
    }
    let mut out = vec![0f64; n + 1];
    let mut run = 0.0;
    for (c, o) in out.iter_mut().enumerate() {
        run += diff[c];
        *o = run;
    }
    out
}

/// [`split_function_keeping`], each cut moved up to [`CUT_WINDOW`] of its
/// stage's share to where the fewest bytes cross ([`crossing_bytes`], not
/// counting `local` values): a cut between a cache and the attention that
/// reads it sent the whole cache.
pub fn split_function_cut(
    graph: &Graph,
    weights: &[f64],
    together: &[(usize, usize)],
    local: &HashSet<String>,
) -> Result<Vec<StagePlan>> {
    if !graph.vars().is_empty() {
        bail!(
            "'{}' carries state (caches) between calls; split a stateless function",
            graph.name
        );
    }
    let weights: &[f64] = if weights.is_empty() { &[1.0] } else { weights };
    let n = weights.len();
    let is_source = |op: &Op| {
        matches!(
            op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        )
    };
    let costs: Vec<f64> = graph
        .nodes()
        .iter()
        .map(|nd| node_cost(graph, nd))
        .collect();
    let total: f64 = costs.iter().sum::<f64>().max(1.0);
    // Where stage `s` (starting at cost `start`) ends: its weight's share of
    // what is left, so a stage held past its share takes from every later
    // stage, not just the next.
    let bound = |s: usize, start: f64| {
        let rest: f64 = weights[s..].iter().sum::<f64>().max(1e-12);
        start + (total - start).max(0.0) * weights[s] / rest
    };
    // The compute nodes in order, the cost before each, and where a cut may go.
    let compute: Vec<usize> = graph
        .nodes()
        .iter()
        .filter(|nd| !is_source(&nd.op))
        .map(|nd| nd.id.0 as usize)
        .collect();
    let mut before = Vec::with_capacity(compute.len() + 1);
    let mut run = 0.0;
    for &c in &compute {
        before.push(run);
        run += costs[c];
    }
    let allowed = |c: usize| !together.iter().any(|&(a, b)| a < c && c <= b);
    let bytes = crossing_bytes(graph, local);
    // Each cut: near its share's point, where the fewest bytes cross; with no
    // candidate there, the first allowed one past it (a span holds it).
    let mut starts = vec![0usize];
    let mut start = 0.0;
    for st in 0..n - 1 {
        let end = bound(st, start);
        let share = (end - start).max(0.0);
        let (lo, hi) = (end - CUT_WINDOW * share, end + CUT_WINDOW * share);
        let from = *starts.last().unwrap() + 1;
        let near = (from..compute.len())
            .filter(|&k| before[k] >= lo && before[k] <= hi && allowed(compute[k]))
            .min_by(|&a, &b| {
                let key = |k: usize| (bytes[compute[k]], (before[k] - end).abs());
                key(a)
                    .partial_cmp(&key(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let k = near
            .or_else(|| {
                (from..compute.len()).find(|&k| before[k] > end - 1e-9 && allowed(compute[k]))
            })
            .unwrap_or(compute.len());
        starts.push(k.max(from).min(compute.len()));
        start = before.get(k).copied().unwrap_or(run);
    }
    // Stage of each compute node.
    let mut stage_of: Vec<Option<usize>> = vec![None; graph.len()];
    let mut st = 0usize;
    for (k, &c) in compute.iter().enumerate() {
        while st + 1 < n && k >= starts[st + 1] {
            st += 1;
        }
        stage_of[c] = Some(st);
    }
    // Where each value is read: the stages, and whether the caller wants it.
    let mut readers: Vec<HashSet<usize>> = vec![HashSet::new(); graph.len()];
    for nd in graph.nodes() {
        if let Some(st) = stage_of[nd.id.0 as usize] {
            for &i in &nd.inputs {
                readers[i.0 as usize].insert(st);
            }
        }
    }
    let outputs: HashSet<NodeId> = graph.outputs.iter().copied().collect();
    let mut plans = Vec::with_capacity(n);
    for st in 0..n {
        let mut sub = Graph::new(format!("{}#stage{st}", graph.name));
        let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
        let (mut reads, mut params) = (Vec::new(), Vec::new());
        for nd in graph.nodes() {
            if stage_of[nd.id.0 as usize] != Some(st) {
                continue;
            }
            let mut ins = Vec::with_capacity(nd.inputs.len());
            for &i in &nd.inputs {
                if let Some(&m) = remap.get(&i) {
                    ins.push(m);
                    continue;
                }
                let src = graph.node(i);
                let id = match &src.op {
                    Op::Param { name } => {
                        params.push(name.clone());
                        sub.append_node(src.op.clone(), vec![], src.shape.clone(), src.name.clone())
                    }
                    Op::Constant { .. } => {
                        sub.append_node(src.op.clone(), vec![], src.shape.clone(), src.name.clone())
                    }
                    _ => {
                        let name = value_name(graph, i);
                        reads.push(name.clone());
                        sub.append_node(
                            Op::Input { name: name.clone() },
                            vec![],
                            src.shape.clone(),
                            Some(name),
                        )
                    }
                };
                remap.insert(i, id);
                ins.push(id);
            }
            let id = sub.append_node(nd.op.clone(), ins, nd.shape.clone(), nd.name.clone());
            remap.insert(nd.id, id);
        }
        // Outputs: what a later stage reads, or the caller wants.
        let mut writes = Vec::new();
        let mut outs = Vec::new();
        for nd in graph.nodes() {
            if stage_of[nd.id.0 as usize] == Some(st)
                && (outputs.contains(&nd.id) || readers[nd.id.0 as usize].iter().any(|&r| r > st))
            {
                writes.push(value_name(graph, nd.id));
                outs.push(remap[&nd.id]);
            }
        }
        sub.set_outputs(outs);
        plans.push(StagePlan {
            graph: sub,
            reads,
            params,
            writes,
            carry: Vec::new(),
            number_format: None,
            device: "auto".into(),
            wire: WireDtype::F32,
            state: Vec::new(),
            prev: None,
            next: None,
            index: st,
            weights_key: String::new(),
            layout: None,
        });
    }
    // Carry after stage s: every value read past s, and the caller's outputs.
    let wanted: HashSet<String> = graph
        .outputs
        .iter()
        .map(|&o| value_name(graph, o))
        .collect();
    for st in 0..n {
        let mut later: HashSet<String> = plans[st + 1..]
            .iter()
            .flat_map(|p| p.reads.iter().cloned())
            .collect();
        later.extend(wanted.iter().cloned());
        let mut have: HashSet<String> = graph
            .nodes()
            .iter()
            .filter(|nd| matches!(nd.op, Op::Input { .. }))
            .map(|nd| value_name(graph, nd.id))
            .collect();
        for p in &plans[..=st] {
            have.extend(p.writes.iter().cloned());
        }
        let mut carry: Vec<String> = have.intersection(&later).cloned().collect();
        carry.sort();
        plans[st].carry = carry;
    }
    Ok(plans)
}

type Values = BTreeMap<String, Vec<f32>>;

/// A stage's weights: f32 params as values, and the params its graph
/// declares in another dtype (bf16/f16, packed GGUF blocks behind a
/// `DequantMatMul`) as their bytes — shipped and bound as stored.
#[derive(Default, Clone)]
struct Weights {
    f32: Values,
    typed: BTreeMap<String, (rlx_ir::DType, Vec<u8>)>,
}

impl Weights {
    /// `names` of `graph`, read the package's way (layouts, references).
    fn read(pkg: &super::rlxp::PackageWeights, graph: &Graph, names: &[String]) -> Result<Self> {
        let decl: HashMap<String, (rlx_ir::DType, Vec<usize>)> = params_of(graph)
            .into_iter()
            .map(|(n, dt, d)| (n, (dt, d)))
            .collect();
        let mut w = Self::default();
        for name in names {
            let (dt, dims) = decl.get(name).with_context(|| format!("param {name}"))?;
            if *dt == rlx_ir::DType::F32 {
                w.f32.insert(name.clone(), pkg.values(name, dims)?);
            } else {
                w.typed
                    .insert(name.clone(), (*dt, pkg.typed(name, *dt, dims)?));
            }
        }
        Ok(w)
    }

    /// Only `names` (a stage's own), copied.
    fn subset(&self, names: &[String]) -> Self {
        let mut w = Self::default();
        for n in names {
            if let Some(v) = self.f32.get(n) {
                w.f32.insert(n.clone(), v.clone());
            }
            if let Some(v) = self.typed.get(n) {
                w.typed.insert(n.clone(), v.clone());
            }
        }
        w
    }
}

/// Send a stage's weights: the f32 values in `wire`, then the typed params
/// (names and dtypes, then one frame of bytes each).
fn send_weights(group: &ProcessGroup, to: u32, w: &Weights, wire: WireDtype) -> Result<()> {
    send_values(group, to, Some(&w.f32), wire)?;
    let t = group.transport();
    let header = json!({
        "names": w.typed.keys().collect::<Vec<_>>(),
        "dtypes": w.typed.values().map(|(d, _)| super::dtype_name(*d)).collect::<Vec<_>>(),
    });
    t.send_bytes(to, TAG_TYPED, &serde_json::to_vec(&header)?)
        .map_err(|e| anyhow::anyhow!("send typed params → {to}: {e}"))?;
    for (_, b) in w.typed.values() {
        t.send_bytes(to, TAG_TYPED, b)
            .map_err(|e| anyhow::anyhow!("send typed params → {to}: {e}"))?;
    }
    Ok(())
}

fn recv_weights(group: &ProcessGroup, from: u32) -> Result<Weights> {
    let f32 = recv_values(group, from)?.unwrap_or_default();
    let t = group.transport();
    let recv = || {
        t.recv_bytes(from, TAG_TYPED)
            .map_err(|e| anyhow::anyhow!("receive typed params ← {from}: {e}"))
    };
    let h: Value = serde_json::from_slice(&recv()?)?;
    let names: Vec<String> = serde_json::from_value(h["names"].clone())?;
    let dtypes: Vec<String> = serde_json::from_value(h["dtypes"].clone())?;
    let mut typed = BTreeMap::new();
    for (n, d) in names.into_iter().zip(dtypes) {
        typed.insert(n, (super::parse_dtype(&d)?, recv()?));
    }
    Ok(Weights { f32, typed })
}

/// Send a set of named tensors (the names and lengths, then one payload in
/// `wire`); `None` tells the receiver to stop.
fn send_values(
    group: &ProcessGroup,
    to: u32,
    values: Option<&Values>,
    wire: WireDtype,
) -> Result<()> {
    let header = match values {
        None => json!({"stop": true}),
        Some(v) => json!({
            "names": v.keys().collect::<Vec<_>>(),
            "lens": v.values().map(Vec::len).collect::<Vec<_>>(),
        }),
    };
    group
        .transport()
        .send_bytes(to, TAG_VALUES, &serde_json::to_vec(&header)?)
        .map_err(|e| anyhow::anyhow!("send values → {to}: {e}"))?;
    if let Some(v) = values {
        let flat: Vec<f32> = v.values().flatten().copied().collect();
        send_activation_as(group, to, &flat, wire).map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

fn recv_values(group: &ProcessGroup, from: u32) -> Result<Option<Values>> {
    let h: Value = serde_json::from_slice(
        &group
            .transport()
            .recv_bytes(from, TAG_VALUES)
            .map_err(|e| anyhow::anyhow!("receive values ← {from}: {e}"))?,
    )?;
    if h.get("stop").is_some() {
        return Ok(None);
    }
    let names: Vec<String> = serde_json::from_value(h["names"].clone())?;
    let lens: Vec<usize> = serde_json::from_value(h["lens"].clone())?;
    let flat = recv_activation_any(group, from).map_err(anyhow::Error::msg)?;
    let mut out = Values::new();
    let mut at = 0;
    for (n, l) in names.into_iter().zip(lens) {
        out.insert(
            n,
            flat.get(at..at + l)
                .context("short values payload")?
                .to_vec(),
        );
        at += l;
    }
    Ok(Some(out))
}

fn announce(group: &ProcessGroup, kind: &str) -> Result<()> {
    let b = serde_json::to_vec(&json!({"kind": kind}))?;
    for r in 1..group.world_size() {
        group
            .transport()
            .send_bytes(r, TAG_JOB, &b)
            .map_err(|e| anyhow::anyhow!("announce job → {r}: {e}"))?;
    }
    Ok(())
}

/// A compiled stage with its plan.
struct Stage {
    plan: StagePlan,
    compiled: crate::CompiledGraph,
    device: Device,
    /// The stage's weights, kept while it runs off the CPU: a GPU that fails
    /// (out of memory, a kernel fault) is replaced by the CPU with them.
    params: Option<Weights>,
    /// One cache per `plan.state` slot, and the cache rows so far.
    caches: Vec<Vec<f32>>,
    past: usize,
    /// A step run under [`KEEP`]: each slot's update (an append slot's new
    /// rows) and the row count, kept or dropped by the next step's `KEEP`.
    pending: Option<(Vec<Vec<f32>>, usize)>,
    /// `RLX_STAGE_PROFILE`: seconds preparing, running, extracting; steps.
    profile: (f64, f64, f64, usize),
}

impl Stage {
    /// Forget every request's state (caches, the pending update), keeping the
    /// compiled stage and its weights: a new job on a kept stage.
    fn reset(&mut self) {
        self.caches = vec![Vec::new(); self.plan.state.len()];
        self.past = 0;
        self.pending = None;
        self.profile = Default::default();
    }

    fn compile(plan: StagePlan, params: &Weights) -> Result<Self> {
        let device = crate::dist::resolve_device_pub(&plan.device);
        let mut s = match Self::build(&plan, params, device) {
            Ok(compiled) => Self {
                caches: vec![Vec::new(); plan.state.len()],
                plan,
                compiled,
                device,
                params: None,
                past: 0,
                pending: None,
                profile: Default::default(),
            },
            Err(e) if device != Device::Cpu => {
                eprintln!(
                    "  stage on {} failed to compile ({e:#}) — running it on the CPU",
                    crate::device_label(device)
                );
                Self {
                    compiled: Self::build(&plan, params, Device::Cpu)?,
                    caches: vec![Vec::new(); plan.state.len()],
                    plan,
                    device: Device::Cpu,
                    params: None,
                    past: 0,
                    pending: None,
                    profile: Default::default(),
                }
            }
            Err(e) => return Err(e),
        };
        if s.device != Device::Cpu {
            s.params = Some(params.subset(&s.plan.params));
        }
        Ok(s)
    }

    fn build(plan: &StagePlan, params: &Weights, device: Device) -> Result<crate::CompiledGraph> {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let opts = CompileOptions {
            number_format: plan
                .number_format
                .as_deref()
                .map(str::parse)
                .transpose()
                .map_err(|e: String| anyhow::anyhow!(e))?,
            ..Default::default()
        };
        catch_unwind(AssertUnwindSafe(|| -> Result<crate::CompiledGraph> {
            let mut compiled = Session::new(device).compile_with(plan.graph.clone(), &opts);
            for n in &plan.params {
                if let Some((dt, b)) = params.typed.get(n) {
                    compiled.set_param_typed(n, b, *dt);
                    continue;
                }
                let v = params
                    .f32
                    .get(n)
                    .with_context(|| format!("no value for param {n}"))?;
                compiled.set_param(n, v);
            }
            Ok(compiled)
        }))
        .map_err(|p| anyhow::anyhow!("backend panic: {}", panic_text(p)))?
    }

    /// Run on `values`, then keep only what goes on.
    fn step(&mut self, mut values: Values) -> Result<Values> {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let t_in = std::time::Instant::now();
        // Under `KEEP`, settle the previous step's update first.
        let keep = values.get(KEEP).cloned();
        if let Some(k) = &keep {
            self.settle(k.first().copied().unwrap_or(0.0) as usize)?;
        }
        // Stage-local state in, padded to its input.
        for (slot, cache) in self.plan.state.iter().zip(&self.caches) {
            let mut v = cache.clone();
            v.resize(slot.numel, 0.0);
            values.insert(slot.input.clone(), v);
        }
        let feeds: Vec<(&str, &[f32])> = self
            .plan
            .reads
            .iter()
            .map(|n| {
                values
                    .get(n)
                    .map(|v| (n.as_str(), v.as_slice()))
                    .with_context(|| format!("stage input {n} never arrived"))
            })
            .collect::<Result<_>>()?;
        let t_run = std::time::Instant::now();
        let prep = t_run - t_in;
        let outs = match catch_unwind(AssertUnwindSafe(|| self.compiled.run(&feeds))) {
            Ok(o) => o,
            Err(p) => {
                let params = self
                    .params
                    .take()
                    .ok_or_else(|| anyhow::anyhow!("stage failed on the CPU: {}", panic_text(p)))?;
                eprintln!(
                    "  stage on {} failed while running — rerunning it on the CPU",
                    crate::device_label(self.device)
                );
                let cpu = Self::build(&self.plan, &params, Device::Cpu)?;
                // The failed device's state may be broken; never drop it.
                std::mem::forget(std::mem::replace(&mut self.compiled, cpu));
                self.device = Device::Cpu;
                self.compiled.run(&feeds)
            }
        };
        let run = t_run.elapsed();
        let t_out = std::time::Instant::now();
        for (n, v) in self.plan.writes.iter().zip(outs) {
            values.insert(n.clone(), v);
        }
        // Stage-local state out: kept here, not forwarded. An append slot's
        // output is its padded cache with the step's new rows after it.
        let mut updates = Vec::with_capacity(self.plan.state.len());
        let mut rows_new = 1;
        for slot in &self.plan.state {
            values.remove(&slot.input);
            let out = values
                .remove(&slot.output)
                .with_context(|| format!("state output {} was not produced", slot.output))?;
            if !slot.append {
                updates.push(out);
                continue;
            }
            let w = slot.width.max(1);
            let rows = out.len() / w;
            let n = rows.saturating_sub(slot.numel / w).max(1);
            if rows < self.past + n {
                bail!(
                    "cache output {} has {rows} rows, needs {}",
                    slot.output,
                    self.past + n
                );
            }
            rows_new = n;
            updates.push(out[(rows - n) * w..rows * w].to_vec());
        }
        if keep.is_some() {
            self.pending = Some((updates, rows_new));
        } else {
            self.commit(updates, rows_new, rows_new)?;
        }
        if rlx_ir::env::flag("RLX_STAGE_PROFILE") {
            let p = &mut self.profile;
            p.0 += prep.as_secs_f64();
            p.1 += run.as_secs_f64();
            p.2 += t_out.elapsed().as_secs_f64();
            p.3 += 1;
            if p.3.is_multiple_of(16) {
                let ms = |x: f64| 1e3 * x / p.3 as f64;
                eprintln!(
                    "  stage profile ({} steps): state in {:.2} ms, run {:.2} ms, state out {:.2} ms",
                    p.3,
                    ms(p.0),
                    ms(p.1),
                    ms(p.2)
                );
            }
        }
        let carry: HashSet<&String> = self.plan.carry.iter().collect();
        values.retain(|k, _| carry.contains(k) || k.starts_with(TIME));
        values.insert(
            format!("{TIME}{}", self.plan.index),
            vec![run.as_secs_f32()],
        );
        // The next stage settles its own update the same way.
        if let Some(k) = keep {
            values.insert(KEEP.into(), k);
        }
        Ok(values)
    }

    /// Keep `rows` of the pending step's new rows: all of them, or 0 to drop
    /// the step.
    fn settle(&mut self, rows: usize) -> Result<()> {
        let Some((updates, n)) = self.pending.take() else {
            return Ok(());
        };
        if rows == 0 {
            return Ok(());
        }
        self.commit(updates, n, rows.min(n))
    }

    /// Apply a step's `updates`: append slots keep the first `rows` of their
    /// `n` new rows, the others take theirs whole (only with all `n` kept:
    /// a recurrent state has no row to stop at).
    fn commit(&mut self, updates: Vec<Vec<f32>>, n: usize, rows: usize) -> Result<()> {
        if self.plan.state.is_empty() {
            return Ok(());
        }
        if rows < n && self.plan.state.iter().any(|s| !s.append) {
            bail!("kept {rows} of {n} rows, but a replaced state cannot keep part of a step");
        }
        let past = self.past;
        for ((slot, cache), update) in self.plan.state.iter().zip(&mut self.caches).zip(updates) {
            if !slot.append {
                *cache = update;
                continue;
            }
            let w = slot.width.max(1);
            cache.truncate(past * w);
            cache.extend_from_slice(&update[..rows * w]);
        }
        self.past += rows;
        Ok(())
    }
}

fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "backend panic".into())
}

/// A package's stateful `generate` entry, read for distributed decoding:
/// the step function, its states, and how to feed each step.
pub struct DecodeEntry {
    pub function: String,
    /// The dims to specialize the step at (`upper`, `cap`).
    pub dims: Map<String, Value>,
    /// `(input, output index, append)`, as [`InferJob::start_with_state`] takes.
    pub state: Vec<(String, usize, bool)>,
    pub vocab: usize,
    pub stop: Vec<u32>,
    pub upper: usize,
    /// The output holding a draft head's logits for the token after the
    /// next (Qwen3.5's MTP), when the step has one.
    pub draft: Option<usize>,
    /// A speculative step (`generate.verify`): its function, and how many
    /// tokens it takes (the last one, then the draft).
    pub verify: Option<(String, usize)>,
    token: String,
    mask: Option<String>,
    /// `(input, table, row width)`.
    tables: Vec<(String, Vec<f32>, usize)>,
}

/// Read `path`'s `generate` entry for a run of `needed` cache rows (prompt +
/// new tokens): the smallest of its sizes that fits.
pub fn decode_entry(path: &str, needed: usize) -> Result<DecodeEntry> {
    let mut pkg = Opened::open_merged(std::path::Path::new(path))?;
    let entry = super::rlxp::entry_of_kind(&pkg.meta, "generate")
        .context("the package has no generate entry")?;
    let state: Vec<(String, usize, bool)> = entry
        .get("state")
        .and_then(Value::as_array)
        .context("the generate entry has no state (use generate, not decode)")?
        .iter()
        .map(|s| {
            let a = s.as_array().context("state entry")?;
            Ok((
                a.first()
                    .and_then(Value::as_str)
                    .context("state input")?
                    .to_string(),
                a.get(1).and_then(Value::as_u64).context("state output")? as usize,
                a.get(2).and_then(Value::as_str) == Some("append"),
            ))
        })
        .collect::<Result<_>>()?;
    let sizes: Vec<(u64, u64)> = entry
        .get("sizes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|s| Some((s.get("upper")?.as_u64()?, s.get("cap")?.as_u64()?)))
                .collect()
        })
        .unwrap_or_default();
    let (upper, cap) = sizes
        .iter()
        .copied()
        .filter(|(u, _)| *u as usize >= needed)
        .min()
        .or_else(|| sizes.iter().copied().max())
        .context("the generate entry lists no sizes")?;
    let function = entry
        .get("functions")
        .and_then(Value::as_array)
        .and_then(|f| f.first())
        .and_then(Value::as_str)
        .context("the generate entry names no function")?
        .to_string();
    let mut dims = Map::new();
    dims.insert("upper".into(), json!(upper));
    dims.insert("cap".into(), json!(cap));
    let (mut token, mut mask, mut tables) = (None, None, Vec::new());
    if let Some(inputs) = entry.get("inputs").and_then(Value::as_object) {
        for (name, kind) in inputs {
            match kind {
                Value::String(k) if k == "token" || k == "tokens" => token = Some(name.clone()),
                Value::String(k) if k == "decode_mask" => mask = Some(name.clone()),
                Value::Object(o) => {
                    let t = o
                        .get("table")
                        .and_then(Value::as_str)
                        .with_context(|| format!("input {name}: {{table: <tensor>}}"))?;
                    let data = pkg.weights.take_table(t)?;
                    tables.push((name.clone(), data, 0));
                }
                other => bail!("input {name}: feed {other} is not supported distributed"),
            }
        }
    }
    // A table row is as wide as the step's input for it.
    let graph = specialize_function(&mut pkg, &function, Some(&dims))?;
    for (name, _, width) in &mut tables {
        *width = graph
            .nodes()
            .iter()
            .find(|nd| matches!(&nd.op, Op::Input { name: n } if n == name))
            .and_then(|nd| nd.shape.num_elements())
            .with_context(|| format!("table input {name}"))?;
    }
    Ok(DecodeEntry {
        function,
        dims,
        state,
        vocab: entry.get("vocab_size").and_then(Value::as_u64).unwrap_or(0) as usize,
        stop: entry
            .get("stop_ids")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|v| v as u32))
                    .collect()
            })
            .unwrap_or_default(),
        upper: upper as usize,
        draft: entry
            .get("draft_logits")
            .and_then(Value::as_u64)
            .map(|d| d as usize),
        verify: entry.get("verify").and_then(|v| {
            Some((
                v.get("function")?.as_str()?.to_string(),
                v.get("tokens")?.as_u64()? as usize,
            ))
        }),
        token: token.context("the generate entry has no token input")?,
        mask,
        tables,
    })
}

impl DecodeEntry {
    /// The step's inputs at cache length `past` for `token`.
    pub fn feeds(&self, past: usize, token: u32) -> Vec<(String, Vec<f32>)> {
        self.feeds_n(past, &[token])
    }

    /// The inputs of a step that takes `tokens` at once (the verify step) at
    /// cache length `past`: the keys mask over the padded cache and the new
    /// rows after it, and a table row per token.
    pub fn feeds_n(&self, past: usize, tokens: &[u32]) -> Vec<(String, Vec<f32>)> {
        let n = tokens.len();
        let mut f = vec![(
            self.token.clone(),
            tokens.iter().map(|&t| t as f32).collect(),
        )];
        if let Some(m) = &self.mask {
            let keys: Vec<f32> = (0..self.upper + n)
                .map(|i| {
                    if i < past || i >= self.upper {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            f.push((m.clone(), keys));
        }
        for (name, table, w) in &self.tables {
            let rows = (0..n)
                .flat_map(|r| {
                    let at = (past + r) * w;
                    table
                        .get(at..at + w)
                        .map(<[f32]>::to_vec)
                        .unwrap_or_else(|| vec![0.0; *w])
                })
                .collect();
            f.push((name.clone(), rows));
        }
        f
    }
}

/// What a speculative decode drafted and kept.
#[derive(Debug, Clone, Copy, Default)]
pub struct DraftStats {
    /// Verify steps run.
    pub passes: usize,
    /// Draft tokens checked, and of them the ones kept.
    pub drafts: usize,
    pub accepted: usize,
}

/// Greedy speculative decoding through a two-token verify step: each pass
/// runs the last token and the draft head's guess at the next one, and keeps
/// both rows (two new tokens) if the guess was the trunk's own pick, else
/// drops the pass and runs the last token with the right one next, which
/// gives the token after it. Every token is the trunk's argmax after the kept
/// ones, as in one-token decoding. The prompt runs two tokens a pass.
///
/// `pass(past, tokens, keep)` runs the step at cache length `past` and returns
/// the trunk's pick at each of its two rows and the draft head's;
/// `keep` is how many rows of the previous pass to keep.
fn verify_decode(
    prompt: &[u32],
    new: usize,
    stop: &[u32],
    mut pass: impl FnMut(usize, &[u32], usize) -> Result<([u32; 2], u32)>,
) -> Result<(Vec<u32>, DraftStats)> {
    #[derive(Clone, Copy, PartialEq)]
    enum Pair {
        /// Two prompt tokens, more after them.
        Prompt,
        /// Two tokens known right (the prompt's last two, or a dropped pass's
        /// last token and the pick it gave): keep, read the second row.
        Known,
        /// The last token and a guess (`counted`: the draft head's).
        Guess { counted: bool },
    }
    let (l, mut stats) = (prompt.len(), DraftStats::default());
    if l == 0 {
        bail!("decode needs at least one prompt token");
    }
    let first = |i: usize| -> ([u32; 2], Pair) {
        match l - i {
            1 => ([prompt[i], 0], Pair::Guess { counted: false }),
            2 => ([prompt[i], prompt[i + 1]], Pair::Known),
            _ => ([prompt[i], prompt[i + 1]], Pair::Prompt),
        }
    };
    let (mut out, mut past, mut keep, mut i) = (Vec::new(), 0usize, 0usize, 0usize);
    let (mut pair, mut kind) = first(0);
    // Push a token; `true` when decoding is done.
    let emit = |out: &mut Vec<u32>, t: u32| {
        out.push(t);
        out.len() >= new || stop.contains(&t)
    };
    while out.len() < new {
        let ([a0, a1], next_draft) = pass(past, &pair, keep)?;
        stats.passes += 1;
        match kind {
            Pair::Prompt => {
                (keep, past, i) = (2, past + 2, i + 2);
                (pair, kind) = first(i);
            }
            Pair::Known => {
                (keep, past) = (2, past + 2);
                if emit(&mut out, a1) {
                    break;
                }
                (pair, kind) = ([a1, next_draft], Pair::Guess { counted: true });
            }
            Pair::Guess { counted } => {
                stats.drafts += usize::from(counted);
                if a0 == pair[1] {
                    stats.accepted += usize::from(counted);
                    (keep, past) = (2, past + 2);
                    if emit(&mut out, a0) || emit(&mut out, a1) {
                        break;
                    }
                    (pair, kind) = ([a1, next_draft], Pair::Guess { counted: true });
                } else {
                    keep = 0;
                    if emit(&mut out, a0) {
                        break;
                    }
                    (pair, kind) = ([pair[0], a0], Pair::Known);
                }
            }
        }
    }
    Ok((out, stats))
}

/// The same decode on one device, unsplit — the reference a split run is
/// checked against. Runs through the same stage-local state code.
pub fn decode_local(
    path: &str,
    entry: &DecodeEntry,
    device: Device,
    prompt: &[u32],
    new: usize,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    decode_local_drafts(path, entry, device, prompt, new).map(|(out, rows, _)| (out, rows))
}

/// [`decode_local`], plus the draft head's pick after each new token: the
/// guess for the one after it (`drafts[i]` against `tokens[i + 1]`). Empty
/// when the step has no draft output.
pub fn decode_local_drafts(
    path: &str,
    entry: &DecodeEntry,
    device: Device,
    prompt: &[u32],
    new: usize,
) -> Result<(Vec<u32>, Vec<Vec<f32>>, Vec<u32>)> {
    let mut pkg = Opened::open_merged(std::path::Path::new(path))?;
    let graph = specialize_function(&mut pkg, &entry.function, Some(&entry.dims))?;
    let mut plan = split_function(&graph, &[1.0])?.remove(0);
    plan.device = device.as_arg().to_string();
    let params = Weights::read(&pkg.weights, &graph, &plan.params)?;
    for (input, idx, append) in &entry.state {
        let output = value_name(&graph, graph.outputs[*idx]);
        let numel = graph
            .nodes()
            .iter()
            .find(|nd| matches!(&nd.op, Op::Input { name: n } if n == input))
            .and_then(|nd| nd.shape.num_elements())
            .with_context(|| format!("state input {input}"))?;
        plan.state.push(StateSlot {
            input: input.clone(),
            output,
            append: *append,
            numel,
            width: if *append {
                numel / entry.upper.max(1)
            } else {
                0
            },
        });
    }
    let logits_name = value_name(&graph, graph.outputs[0]);
    let draft_name = entry.draft.map(|d| value_name(&graph, graph.outputs[d]));
    let mut stage = Stage::compile(plan, &params)?;
    let (mut out, mut rows, mut drafts) = (Vec::new(), Vec::new(), Vec::new());
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    };
    let mut token = prompt[0];
    for past in 0..prompt.len() + new - 1 {
        let values: Values = entry.feeds(past, token).into_iter().collect();
        let mut v = stage.step(values)?;
        let logits = v.remove(&logits_name).context("logits")?;
        let row = logits[logits.len().saturating_sub(entry.vocab)..].to_vec();
        if past + 1 < prompt.len() {
            token = prompt[past + 1];
            continue;
        }
        let next = argmax(&row);
        if let Some(d) = &draft_name {
            let draft = v.remove(d).context("draft logits")?;
            drafts.push(argmax(&draft[draft.len().saturating_sub(entry.vocab)..]));
        }
        rows.push(row);
        out.push(next);
        if entry.stop.contains(&next) {
            break;
        }
        token = next;
    }
    Ok((out, rows, drafts))
}

/// The pick of each of `rows` rows of `out`: the value itself when the step
/// already reduced them to ids (one value a row, [`StepLayout::argmax`]),
/// else the argmax of the row's last `vocab` logits.
fn picks(out: &[f32], rows: usize, vocab: usize) -> Vec<u32> {
    if out.len() == rows {
        return out.iter().map(|&v| v as u32).collect();
    }
    let w = out.len() / rows.max(1);
    (0..rows)
        .map(|r| {
            out[r * w + w.saturating_sub(vocab)..(r + 1) * w]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        })
        .collect()
}

/// [`verify_decode`] on one device, unsplit, through the entry's verify step
/// — the reference a split speculative run is checked against, and with the
/// draft head the speculative counterpart of [`decode_local`].
pub fn decode_local_verify(
    path: &str,
    entry: &DecodeEntry,
    device: Device,
    prompt: &[u32],
    new: usize,
) -> Result<(Vec<u32>, DraftStats)> {
    let (function, tokens) = entry
        .verify
        .clone()
        .context("the generate entry has no verify step")?;
    if tokens != 2 {
        bail!("a verify step of {tokens} tokens: only 2 (the last token and a draft) run");
    }
    let mut pkg = Opened::open_merged(std::path::Path::new(path))?;
    let graph = specialize_function(&mut pkg, &function, Some(&entry.dims))?;
    let mut plan = split_function(&graph, &[1.0])?.remove(0);
    plan.device = device.as_arg().to_string();
    let params = Weights::read(&pkg.weights, &graph, &plan.params)?;
    for (input, idx, append) in &entry.state {
        let output = value_name(&graph, graph.outputs[*idx]);
        let numel = graph
            .nodes()
            .iter()
            .find(|nd| matches!(&nd.op, Op::Input { name: n } if n == input))
            .and_then(|nd| nd.shape.num_elements())
            .with_context(|| format!("state input {input}"))?;
        plan.state.push(StateSlot {
            input: input.clone(),
            output,
            append: *append,
            numel,
            width: if *append {
                numel / entry.upper.max(1)
            } else {
                0
            },
        });
    }
    let logits_name = value_name(&graph, graph.outputs[0]);
    let draft_name = value_name(
        &graph,
        graph.outputs[entry
            .draft
            .context("the generate entry has no draft output")?],
    );
    let mut stage = Stage::compile(plan, &params)?;
    verify_decode(prompt, new, &entry.stop, |past, toks, keep| {
        let mut values: Values = entry.feeds_n(past, toks).into_iter().collect();
        values.insert(KEEP.into(), vec![keep as f32]);
        let mut v = stage.step(values)?;
        let rows = picks(&v.remove(&logits_name).context("logits")?, 2, entry.vocab);
        let draft = picks(
            &v.remove(&draft_name).context("draft logits")?,
            1,
            entry.vocab,
        );
        Ok(([rows[0], rows[1]], draft[0]))
    })
}

/// A function split across the group, ready to run requests (rank 0).
pub struct InferJob {
    first: Stage,
    /// The function's input names and dims, and its outputs' value names.
    pub inputs: Vec<(String, Vec<usize>)>,
    outputs: Vec<String>,
    /// Output names as the package names them.
    pub output_names: Vec<String>,
    wire: WireDtype,
    world: u32,
    /// How many layers' worth of FLOPs each rank got (fractions).
    pub shares: Vec<f64>,
    /// Seconds in rank 0's own stage, seconds waiting on the rest of the
    /// pipeline (send → the result back), and requests run — after the
    /// first, whose seconds (every stage compiling its kernels) are `warmup`.
    pub timing: (f64, f64, usize),
    pub warmup: f64,
    /// The ranks in pipeline order (rank 0 first), and the layout's
    /// estimated seconds a step against rank order's, when links were known.
    pub order: Vec<u32>,
    pub estimate: Option<(f64, f64)>,
    /// Each stage's FLOPs, and its compute seconds summed over `timing.2`
    /// requests (from the values' [`TIME`] entries).
    pub stage_flops: Vec<f64>,
    pub stage_seconds: Vec<f64>,
    /// Workers that kept their stage from an earlier job, weights and all.
    pub kept: usize,
    /// What laying the step out again takes ([`InferJob::calibrate`]).
    ctx: Option<Box<LayoutCtx>>,
}

/// A step's layout inputs, kept to lay it out again with measured speeds.
struct LayoutCtx {
    pkg: Opened,
    graph: Graph,
    state: Vec<(String, usize, bool)>,
    spans: Vec<(usize, usize)>,
    local: HashSet<String>,
    links: Option<Links>,
    upper: usize,
    device: String,
    number_format: Option<String>,
    wire: WireDtype,
    /// The plans now on the ranks: what calibration measures and must beat.
    deployed: Vec<StagePlan>,
    /// [`StagePlan::weights_key`] for this package.
    weights_key: String,
}

/// A deployed layout: rank 0's stage, each stage's FLOPs, and the state
/// outputs (kept by their stages, not returned).
struct Deployed {
    first: Stage,
    costs: Vec<f64>,
    state_outputs: HashSet<String>,
    /// Workers that still held their stage from an earlier job.
    kept: usize,
}

/// Seconds the weights of the stages that `order`/`plans` would move take to
/// reach their ranks from rank 0, the layout deployed now being `was`
/// ([`LayoutCtx::deployed`]): a worker whose stage is the same keeps it, and
/// rank 0 needs no transfer.
fn move_seconds(ctx: &LayoutCtx, was: &[u32], order: &[u32], plans: &[StagePlan]) -> f64 {
    let bytes = |p: &StagePlan| -> f64 {
        p.graph
            .nodes()
            .iter()
            .filter(|nd| matches!(nd.op, Op::Param { .. }))
            .filter_map(|nd| nd.shape.size_bytes())
            .sum::<usize>() as f64
    };
    let same =
        |a: &StagePlan, b: &StagePlan| serde_json::to_vec(a).ok() == serde_json::to_vec(b).ok();
    order
        .iter()
        .zip(plans)
        .filter(|(r, _)| **r != 0)
        .filter(|(r, p)| {
            !was.iter()
                .position(|w| w == *r)
                .and_then(|k| ctx.deployed.get(k))
                .is_some_and(|old| same(old, p))
        })
        .map(|(&r, p)| {
            ctx.links
                .as_ref()
                .map_or(0.0, |l| l.time(0, r as usize, bytes(p)))
        })
        .sum()
}

/// Put `plans` on ranks `order`: stage-local state assigned, each worker
/// sent its stage and weights, rank 0's compiled here.
fn deploy(
    group: &ProcessGroup,
    ctx: &LayoutCtx,
    mut plans: Vec<StagePlan>,
    order: &[u32],
    speeds: &[f64],
) -> Result<Deployed> {
    let n = order.len();
    let graph = &ctx.graph;
    let costs: Vec<f64> = plans
        .iter()
        .map(|p| {
            p.graph
                .nodes()
                .iter()
                .map(|nd| node_cost(&p.graph, nd))
                .sum()
        })
        .collect();
    let layout = PlannedLayout {
        speeds: speeds.to_vec(),
        order: order.to_vec(),
    };
    remember_layout(&layout);
    for p in &mut plans {
        p.device = ctx.device.clone();
        p.number_format = ctx.number_format.clone();
        p.wire = ctx.wire;
        p.layout = Some(layout.clone());
    }
    // Each state belongs to the stage that reads it, which must also be
    // the one that writes its update (both ends of one layer).
    let input_numel = |name: &str| {
        graph
            .nodes()
            .iter()
            .find(|nd| matches!(&nd.op, Op::Input { name: n } if n == name))
            .and_then(|nd| nd.shape.num_elements())
    };
    let upper = ctx.upper;
    let mut state_outputs = HashSet::new();
    for (input, idx, append) in &ctx.state {
        let &out = graph
            .outputs
            .get(*idx)
            .with_context(|| format!("state {input}: no output {idx}"))?;
        let output = value_name(graph, out);
        let numel = input_numel(input).with_context(|| format!("state input {input}"))?;
        let p = plans
            .iter_mut()
            .find(|p| p.reads.contains(input))
            .with_context(|| format!("no stage reads state {input}"))?;
        if !p.writes.contains(&output) {
            bail!(
                "state {input} is read by one stage and updated by another: the split \
                 cut through its layer"
            );
        }
        if *append && upper == 0 {
            bail!("state {input} appends rows, but `upper` is not among the dims");
        }
        p.state.push(StateSlot {
            input: input.clone(),
            output: output.clone(),
            append: *append,
            numel,
            width: if *append { numel / upper } else { 0 },
        });
        state_outputs.insert(output);
    }
    let mut first = None;
    let mut kept = 0;
    for (k, mut plan) in plans.into_iter().enumerate() {
        // Stage k runs on rank order[k], between its neighbours in order.
        let r = order[k] as usize;
        plan.prev = Some(order[(k + n - 1) % n]);
        plan.next = Some(order[(k + 1) % n]);
        plan.index = k;
        plan.weights_key = ctx.weights_key.clone();
        // Each stage's weights, read the package's way (layouts,
        // references), in the dtype its graph declares.
        let params = Weights::read(&ctx.pkg.weights, graph, &plan.params)?;
        if r == 0 {
            first = Some(Stage::compile(plan, &params)?);
            continue;
        }
        group
            .transport()
            .send_bytes(r as u32, TAG_STAGE, &serde_json::to_vec(&plan)?)
            .map_err(|e| anyhow::anyhow!("send stage → {r}: {e}"))?;
        // A worker kept from an earlier job may hold this stage already.
        let need = group
            .transport()
            .recv_bytes(r as u32, TAG_STAGE)
            .map_err(|e| anyhow::anyhow!("stage ack ← {r}: {e}"))?;
        if need != [1] {
            kept += 1;
            continue;
        }
        // Weights must arrive exactly. bf16 is exact for weights that are
        // bf16 values to begin with (a BF16 checkpoint), and half the
        // bytes; anything else goes f32.
        let exact_bf16 = params
            .f32
            .values()
            .all(|v| v.iter().all(|x| x.to_bits() & 0xffff == 0));
        let wire = if exact_bf16 {
            WireDtype::Bf16
        } else {
            WireDtype::F32
        };
        send_weights(group, r as u32, &params, wire)?;
    }
    Ok(Deployed {
        first: first.context("no stage for rank 0")?,
        costs,
        state_outputs,
        kept,
    })
}

/// How [`InferJob::start_with_layout`] lays a step out.
#[derive(Debug, Clone, Default)]
pub struct StepLayout {
    /// Outputs to reduce to the argmax of their last axis (their last
    /// `vocab` entries) before splitting: a greedy decoder needs the token,
    /// and the last stage then sends ids, not rows of logits.
    pub argmax: Vec<usize>,
    pub vocab: usize,
    /// The links between ranks ([`exchange_speeds_links`]): with them the
    /// workers' pipeline order is the one the estimate of compute (by speed)
    /// plus every hop (latency + bytes / bandwidth) puts fastest.
    pub links: Option<Links>,
}

/// Replace each of `outputs` by the argmax of its last axis, over its last
/// `vocab` entries (all of them for 0).
fn argmax_outputs(graph: &mut Graph, outputs: &[usize], vocab: usize) {
    use rlx_ir::{DType, Shape};
    let mut outs = graph.outputs.clone();
    for &i in outputs {
        let Some(&l) = outs.get(i) else { continue };
        let shape = graph.node(l).shape.clone();
        let rank = shape.rank();
        let w = shape.dim(rank - 1).unwrap_static();
        let mut src = l;
        if vocab > 0 && vocab < w {
            let mut dims: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
            dims[rank - 1] = vocab;
            src = graph.add_node(
                Op::Narrow {
                    axis: rank - 1,
                    start: w - vocab,
                    len: vocab,
                },
                vec![l],
                Shape::new(&dims, shape.dtype()),
            );
        }
        let dims: Vec<usize> = shape.dims()[..rank - 1]
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        outs[i] = graph.add_node(
            Op::ArgMax {
                axis: rank - 1,
                keep_dim: false,
            },
            vec![src],
            Shape::new(&dims, DType::F32),
        );
    }
    graph.set_outputs(outs);
}

/// Seconds a step takes laid out as `plans` on ranks `order`: each stage's
/// FLOPs at its rank's speed (GFLOP/s), plus each hop's latency and bytes
/// over its bandwidth ([`Links::time`]; none without links), back to rank 0
/// included.
fn layout_estimate(
    graph: &Graph,
    order: &[u32],
    plans: &[StagePlan],
    speeds: &[f64],
    local: &HashSet<String>,
    links: Option<&Links>,
) -> f64 {
    let n = order.len();
    let bytes: HashMap<String, f64> = graph
        .nodes()
        .iter()
        .map(|nd| {
            (
                value_name(graph, nd.id),
                4.0 * nd.shape.num_elements().unwrap_or(0) as f64,
            )
        })
        .collect();
    plans
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let r = order[k] as usize;
            let flops: f64 = p
                .graph
                .nodes()
                .iter()
                .map(|nd| node_cost(&p.graph, nd))
                .sum();
            let sent: f64 = p
                .carry
                .iter()
                .filter(|c| !local.contains(*c))
                .map(|c| bytes.get(c).copied().unwrap_or(0.0))
                .sum();
            let hop = links.map_or(0.0, |l| l.time(r, order[(k + 1) % n] as usize, sent));
            flops / (speeds[r].max(1e-3) * 1e9) + hop
        })
        .sum()
}

/// `order` lists each of `n` ranks once, rank 0 first.
fn is_pipeline_order(order: &[u32], n: usize) -> bool {
    let mut seen = vec![false; n];
    order.len() == n
        && order.first() == Some(&0)
        && order
            .iter()
            .all(|&r| (r as usize) < n && !std::mem::replace(&mut seen[r as usize], true))
}

/// Split `graph` with the workers in each order (all of them up to 7
/// workers, else rank order), and keep the one the estimate puts fastest:
/// each stage's FLOPs at its rank's speed, plus each hop's latency and bytes
/// over its bandwidth ([`Links::time`]), back to rank 0 included. Returns
/// the order, its plans, and its estimate against rank order's.
fn choose_layout(
    graph: &Graph,
    speeds: &[f64],
    spans: &[(usize, usize)],
    local: &HashSet<String>,
    links: Option<&Links>,
) -> Result<(Vec<u32>, Vec<StagePlan>, Option<(f64, f64)>)> {
    let n = speeds.len();
    let identity: Vec<u32> = (0..n as u32).collect();
    let split = |order: &[u32]| {
        let w: Vec<f64> = order.iter().map(|&r| speeds[r as usize]).collect();
        split_function_cut(graph, &w, spans, local)
    };
    // The order the last job ended with, when its speeds are the ones here.
    if let Some(order) = links
        .and_then(|l| l.order.as_ref())
        .filter(|o| is_pipeline_order(o, n))
    {
        return Ok((order.clone(), split(order)?, None));
    }
    let Some(links) = links.filter(|l| l.latency.len() == n && n > 2) else {
        return Ok((identity.clone(), split(&identity)?, None));
    };
    let estimate = |order: &[u32], plans: &[StagePlan]| {
        layout_estimate(graph, order, plans, speeds, local, Some(links))
    };
    let base_plans = split(&identity)?;
    let base = estimate(&identity, &base_plans);
    let mut best = (base, identity, base_plans);
    if n - 1 <= 7 {
        let mut workers: Vec<u32> = (1..n as u32).collect();
        // Every order of the workers (Heap's algorithm), rank 0 first.
        let mut c = vec![0usize; workers.len()];
        let mut i = 0;
        while i < workers.len() {
            if c[i] < i {
                if i % 2 == 0 {
                    workers.swap(0, i);
                } else {
                    workers.swap(c[i], i);
                }
                let order: Vec<u32> = std::iter::once(0).chain(workers.iter().copied()).collect();
                let plans = split(&order)?;
                let t = estimate(&order, &plans);
                if t < best.0 {
                    best = (t, order, plans);
                }
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
    }
    Ok((best.1, best.2, Some((best.0, base))))
}

impl InferJob {
    /// Open `function` of `path` sized by `dims`, split it over the group
    /// (stage `r`'s share ∝ `speeds[r]`), and send each worker its stage and
    /// weights. `device` and `number_format` apply on every rank.
    pub fn start(
        group: &ProcessGroup,
        path: &str,
        function: &str,
        dims: &Map<String, Value>,
        speeds: &[f64],
        device: &str,
        number_format: Option<String>,
        wire: WireDtype,
    ) -> Result<Self> {
        Self::start_with_state(
            group,
            path,
            function,
            dims,
            speeds,
            device,
            number_format,
            wire,
            &[],
        )
    }

    /// [`Self::start`] for a step function that carries state: `state` lists
    /// `(input, output index, append)` — each input fed from, and its output
    /// written back to, a cache kept by the stage that holds that layer.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_state(
        group: &ProcessGroup,
        path: &str,
        function: &str,
        dims: &Map<String, Value>,
        speeds: &[f64],
        device: &str,
        number_format: Option<String>,
        wire: WireDtype,
        state: &[(String, usize, bool)],
    ) -> Result<Self> {
        Self::start_with_layout(
            group,
            path,
            function,
            dims,
            speeds,
            device,
            number_format,
            wire,
            state,
            &StepLayout::default(),
        )
    }

    /// [`Self::start_with_state`], laid out by `layout`: outputs reduced to
    /// their argmax, the workers in the order the links make fastest.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_layout(
        group: &ProcessGroup,
        path: &str,
        function: &str,
        dims: &Map<String, Value>,
        speeds: &[f64],
        device: &str,
        number_format: Option<String>,
        wire: WireDtype,
        state: &[(String, usize, bool)],
        layout: &StepLayout,
    ) -> Result<Self> {
        let mut pkg = Opened::open_merged(std::path::Path::new(path))?;
        let mut graph = specialize_function(&mut pkg, function, Some(dims))?;
        if !layout.argmax.is_empty() {
            argmax_outputs(&mut graph, &layout.argmax, layout.vocab);
        }
        let n = group.world_size() as usize;
        // At f32 precision, which is what the speed exchange carries: a job
        // planned by speeds a worker remembers then lays the step out exactly
        // as the job before it did.
        let speeds: Vec<f64> = if speeds.len() == n {
            speeds.iter().map(|&v| v as f32 as f64).collect()
        } else {
            vec![1.0; n]
        };
        let local: HashSet<String> = state
            .iter()
            .flat_map(|(i, o, _)| {
                let out = graph.outputs.get(*o).map(|&id| value_name(&graph, id));
                std::iter::once(i.clone()).chain(out)
            })
            .collect();
        let spans = state_spans(&graph, state);
        let (order, plans, estimate) =
            choose_layout(&graph, &speeds, &spans, &local, layout.links.as_ref())?;
        let ctx = LayoutCtx {
            pkg,
            graph,
            state: state.to_vec(),
            spans,
            local,
            // Without the remembered order: calibration searches them all.
            links: layout.links.clone().map(|l| Links { order: None, ..l }),
            upper: dims.get("upper").and_then(Value::as_u64).unwrap_or(0) as usize,
            device: device.into(),
            number_format,
            wire,
            deployed: plans.clone(),
            weights_key: {
                let meta = std::fs::metadata(path).ok();
                let mtime = meta
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_nanos());
                format!("{path}:{}:{mtime}", meta.map_or(0, |m| m.len()))
            },
        };
        announce(group, "infer")?;
        let Deployed {
            first,
            costs,
            state_outputs,
            kept,
        } = deploy(group, &ctx, plans, &order, &speeds)?;
        let graph = &ctx.graph;
        let inputs = graph
            .nodes()
            .iter()
            .filter_map(|nd| match &nd.op {
                Op::Input { name } => Some((
                    name.clone(),
                    nd.shape.dims().iter().map(|d| d.unwrap_static()).collect(),
                )),
                _ => None,
            })
            .collect();
        let names: HashMap<usize, String> = graph.output_names().iter().cloned().collect();
        let outputs = graph
            .outputs
            .iter()
            .map(|&o| value_name(graph, o))
            .filter(|n| !state_outputs.contains(n))
            .collect();
        let output_names = (0..graph.outputs.len())
            .map(|i| names.get(&i).cloned().unwrap_or_else(|| format!("#{i}")))
            .collect();
        let mut job = Self {
            first,
            inputs,
            outputs,
            output_names,
            wire,
            world: n as u32,
            shares: Vec::new(),
            timing: (0.0, 0.0, 0),
            warmup: 0.0,
            order,
            estimate,
            stage_flops: Vec::new(),
            stage_seconds: Vec::new(),
            kept,
            ctx: Some(Box::new(ctx)),
        };
        job.set_costs(costs);
        Ok(job)
    }

    /// Each rank's share (stage k runs on order[k]) from each stage's FLOPs.
    fn set_costs(&mut self, costs: Vec<f64>) {
        let tc: f64 = costs.iter().sum::<f64>().max(1.0);
        let mut sh = vec![0.0; self.world as usize];
        for (k, c) in costs.iter().enumerate() {
            sh[self.order[k] as usize] = c / tc;
        }
        self.shares = sh;
        self.stage_seconds = vec![0.0; costs.len()];
        self.stage_flops = costs;
    }

    /// Each rank's measured speed (GFLOP/s): its stage's FLOPs over its mean
    /// compute seconds a request. `None` before any request was timed.
    pub fn measured_speeds(&self) -> Option<Vec<f64>> {
        let reqs = self.timing.2;
        if reqs == 0 {
            return None;
        }
        let mut speeds = vec![0.0; self.world as usize];
        for (k, (&f, &t)) in self.stage_flops.iter().zip(&self.stage_seconds).enumerate() {
            let per = (t / reqs as f64).max(1e-6);
            speeds[self.order[k] as usize] = (f / per / 1e9).max(1e-3);
        }
        Some(speeds)
    }

    /// Calibrate on the running cluster: run `passes` throwaway requests of
    /// `inputs` (under [`KEEP`], so no stage keeps their state), time every
    /// stage, and lay the step out again with the measured speeds (order and
    /// cuts, against the links). Where that is estimated at least `gain`
    /// (e.g. 0.1) faster, the workers take their new stages; state starts
    /// over, so this belongs before the first real request. Returns the
    /// measured speeds and whether the layout changed.
    pub fn calibrate(
        &mut self,
        group: &ProcessGroup,
        inputs: &[(String, Vec<f32>)],
        passes: usize,
        gain: f64,
    ) -> Result<(Vec<f64>, bool)> {
        self.calibrate_for(group, inputs, passes, gain, None)
    }

    /// [`Self::calibrate`] for a job of about `steps` requests: the step is
    /// laid out again only where the time it is estimated to save over them
    /// beats the time its new stages' weights take to reach their ranks
    /// (each moved stage's parameter bytes over the link from rank 0, an
    /// unchanged stage moving nothing). On a short job a re-layout costs more
    /// than it saves: 60 tokens over Wi-Fi saved ~1.5 s for ~15 s of weights,
    /// and the calibration's own noise had picked a slower layout.
    pub fn calibrate_for(
        &mut self,
        group: &ProcessGroup,
        inputs: &[(String, Vec<f32>)],
        passes: usize,
        gain: f64,
        steps: Option<usize>,
    ) -> Result<(Vec<f64>, bool)> {
        let mut feed: Vec<(String, Vec<f32>)> = inputs.to_vec();
        feed.push((KEEP.into(), vec![0.0]));
        let refs: Vec<(&str, &[f32])> = feed
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        // The first request compiles every stage's kernels: not timed.
        for _ in 0..passes.max(1) + 1 {
            self.run(group, &refs)?;
        }
        let speeds: Vec<f64> = self
            .measured_speeds()
            .context("calibration timed nothing")?
            .iter()
            .map(|&v| v as f32 as f64)
            .collect();
        let mut ctx = self
            .ctx
            .take()
            .context("this job keeps no layout to redo")?;
        let result = (|| -> Result<bool> {
            // The deployed layout, timed at the measured speeds (not split
            // again: that would already move work off a slow rank).
            let now = layout_estimate(
                &ctx.graph,
                &self.order,
                &ctx.deployed,
                &speeds,
                &ctx.local,
                ctx.links.as_ref(),
            );
            let (order, plans, _) = choose_layout(
                &ctx.graph,
                &speeds,
                &ctx.spans,
                &ctx.local,
                ctx.links.as_ref(),
            )?;
            let next = layout_estimate(
                &ctx.graph,
                &order,
                &plans,
                &speeds,
                &ctx.local,
                ctx.links.as_ref(),
            );
            if next > now * (1.0 - gain) {
                return Ok(false);
            }
            if let Some(steps) = steps
                && (now - next) * steps as f64 <= move_seconds(&ctx, &self.order, &order, &plans)
            {
                return Ok(false);
            }
            // Every worker passes the marker on, then waits for its stage.
            if self.world > 1 {
                let marker: Values = [(RESTAGE.to_string(), vec![1.0])].into_iter().collect();
                send_values(group, self.order[1], Some(&marker), self.wire)?;
                recv_values(group, self.order[self.order.len() - 1])?
                    .context("the last stage stopped")?;
            }
            ctx.deployed = plans.clone();
            let Deployed { first, costs, .. } = deploy(group, &ctx, plans, &order, &speeds)?;
            self.first = first;
            self.order = order;
            self.estimate = Some((next, now));
            self.set_costs(costs);
            self.timing = (0.0, 0.0, 0);
            self.warmup = 0.0;
            Ok(true)
        })();
        self.ctx = Some(ctx);
        let relaid = result?;
        // The throwaway passes are not the job's: its counters start here
        // (a kept layout keeps its compiled kernels, so no warm-up again).
        self.timing = (0.0, 0.0, 0);
        self.stage_seconds.iter_mut().for_each(|t| *t = 0.0);
        Ok((speeds, relaid))
    }

    /// Run one request: the function's inputs by name → its outputs in order.
    pub fn run(
        &mut self,
        group: &ProcessGroup,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>> {
        let mut values: Values = inputs
            .iter()
            .map(|(n, v)| (n.to_string(), v.to_vec()))
            .collect();
        let t0 = std::time::Instant::now();
        values = self.first.step(values)?;
        let own = t0.elapsed().as_secs_f64();
        let mut rest = 0.0;
        if self.world > 1 {
            let t = std::time::Instant::now();
            let (head, tail) = (self.order[1], self.order[self.order.len() - 1]);
            send_values(group, head, Some(&values), self.wire)?;
            values = recv_values(group, tail)?.context("the last stage stopped")?;
            rest = t.elapsed().as_secs_f64();
        }
        if self.warmup == 0.0 {
            self.warmup = t0.elapsed().as_secs_f64();
        } else {
            self.timing.0 += own;
            self.timing.1 += rest;
            self.timing.2 += 1;
            for (k, t) in self.stage_seconds.iter_mut().enumerate() {
                if let Some(v) = values.get(&format!("{TIME}{k}")) {
                    *t += v.first().copied().unwrap_or(0.0) as f64;
                }
            }
        }
        self.outputs
            .iter()
            .map(|n| {
                values
                    .get(n)
                    .cloned()
                    .with_context(|| format!("output {n} missing"))
            })
            .collect()
    }

    /// Greedy decoding for an LM function: its token input is its largest
    /// input (`[1, seq]` or `[seq]`); any one-element input is the position to
    /// read (`last`), fed the last real token's index; the first output is the
    /// logits — one row (`[…, vocab]`, read at `last`) or every position's
    /// (`[…, seq, vocab]`). The prompt is right-padded to the function's `seq`.
    /// Returns the new tokens.
    pub fn generate(
        &mut self,
        group: &ProcessGroup,
        prompt: &[u32],
        new: usize,
    ) -> Result<Vec<u32>> {
        let tok = self
            .inputs
            .iter()
            .max_by_key(|(_, d)| d.iter().product::<usize>())
            .cloned()
            .context("the function has no inputs")?;
        let seq = *tok.1.last().context("token input has no seq axis")?;
        let positions: Vec<String> = self
            .inputs
            .iter()
            .filter(|(n, d)| *n != tok.0 && d.iter().product::<usize>() == 1)
            .map(|(n, _)| n.clone())
            .collect();
        if positions.len() + 1 != self.inputs.len() {
            bail!("generate takes a token input and at most one-element position inputs");
        }
        let mut tokens: Vec<u32> = prompt.to_vec();
        let mut out = Vec::new();
        for _ in 0..new {
            if tokens.len() >= seq {
                break;
            }
            let mut feed = vec![0.0f32; seq];
            for (f, &t) in feed.iter_mut().zip(&tokens) {
                *f = t as f32;
            }
            let last = [(tokens.len() - 1) as f32];
            let mut feeds: Vec<(&str, &[f32])> = vec![(tok.0.as_str(), &feed)];
            for p in &positions {
                feeds.push((p.as_str(), &last));
            }
            let logits = self.run(group, &feeds)?.remove(0);
            let row = next_token_row(&logits, seq, tokens.len(), !positions.is_empty());
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            tokens.push(next);
            out.push(next);
        }
        Ok(out)
    }

    /// Greedy decoding through a stateful step (a package's `generate` entry
    /// with `state`): one token per step, each stage keeping its layers' state.
    /// `feed(past, token)` gives the step's inputs (token, mask, table rows);
    /// the first output is the logits, of which the last `vocab` count. The
    /// prompt is run token by token. Returns the new tokens and each step's
    /// logits (one row per generated token).
    pub fn decode(
        &mut self,
        group: &ProcessGroup,
        prompt: &[u32],
        new: usize,
        vocab: usize,
        stop: &[u32],
        mut feed: impl FnMut(usize, u32) -> Vec<(String, Vec<f32>)>,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
        if prompt.is_empty() {
            bail!("decode needs at least one prompt token");
        }
        let (mut out, mut rows) = (Vec::new(), Vec::new());
        let mut token = prompt[0];
        for past in 0..prompt.len() + new - 1 {
            let inputs = feed(past, token);
            let refs: Vec<(&str, &[f32])> = inputs
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let logits = self.run(group, &refs)?.remove(0);
            let row = logits[logits.len().saturating_sub(vocab)..].to_vec();
            if past + 1 < prompt.len() {
                token = prompt[past + 1];
                continue;
            }
            // The token itself when the step already took the argmax.
            let next = picks(&row, 1, vocab)[0];
            rows.push(row);
            out.push(next);
            if stop.contains(&next) {
                break;
            }
            token = next;
        }
        Ok((out, rows))
    }

    /// Greedy speculative decoding through a verify step (the job started on
    /// the entry's verify function): [`verify_decode`] across the group, each
    /// stage holding a pass's state until the next one says to keep or drop
    /// it ([`KEEP`]). `feed(past, tokens)` gives a pass's inputs; the step's
    /// first output is the logits, its second the draft logits.
    pub fn decode_verify(
        &mut self,
        group: &ProcessGroup,
        prompt: &[u32],
        new: usize,
        vocab: usize,
        stop: &[u32],
        mut feed: impl FnMut(usize, &[u32]) -> Vec<(String, Vec<f32>)>,
    ) -> Result<(Vec<u32>, DraftStats)> {
        if self.outputs.len() < 2 {
            bail!("the verify step has no draft output");
        }
        verify_decode(prompt, new, stop, |past, toks, keep| {
            let mut inputs = feed(past, toks);
            inputs.push((KEEP.into(), vec![keep as f32]));
            let refs: Vec<(&str, &[f32])> = inputs
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect();
            let out = self.run(group, &refs)?;
            let rows = picks(&out[0], 2, vocab);
            let draft = picks(&out[1], 1, vocab);
            Ok(([rows[0], rows[1]], draft[0]))
        })
    }

    /// Tell the workers the job is over.
    pub fn finish(self, group: &ProcessGroup) -> Result<()> {
        if self.world > 1 {
            send_values(group, self.order[1], None, self.wire)?;
        }
        Ok(())
    }
}

thread_local! {
    /// The stage a worker served last, by its plan's bytes: a persistent
    /// worker given the same plan again (same package, same layout) keeps it,
    /// its weights and its compiled kernels.
    static KEPT_STAGE: std::cell::RefCell<Option<(Vec<u8>, Stage)>> =
        const { std::cell::RefCell::new(None) };
}

/// A worker's stage from rank 0: the one it kept, if the plan is the same
/// (its state reset), else the plan's weights received and compiled. Tells
/// rank 0 which ([`deploy`] waits for it). Returns the plan bytes too, the
/// key to keep it under.
/// `RLX_STAGE_KEEP_WARM=1`: while a worker's stage waits for its next
/// request, a thread keeps its device busy with a small matmul, so the GPU
/// does not clock down between requests. A pipeline stage idles most of a
/// step while the others compute; an integrated GPU (a 780M) dropped to its
/// lowest clock there, and its matmuls ran 3× slower than back to back (the
/// same Q8_0 GEMV: 389 µs after 300 ms idle, 128 µs with the gap kept busy).
/// It costs power for the job's length, so it is opt-in.
struct KeepWarm {
    on: std::sync::Arc<std::sync::atomic::AtomicBool>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl KeepWarm {
    /// A warmer for `device` when `RLX_STAGE_KEEP_WARM` asks for one.
    fn start(device: Device) -> Option<Self> {
        use std::sync::atomic::{AtomicBool, Ordering};
        if device == Device::Cpu || !rlx_ir::env::flag("RLX_STAGE_KEEP_WARM") {
            return None;
        }
        let on = std::sync::Arc::new(AtomicBool::new(false));
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let (on2, stop2) = (on.clone(), stop.clone());
        let thread = std::thread::Builder::new()
            .name("rlx-keep-warm".into())
            .spawn(move || {
                use rlx_ir::{DType, Graph, Shape};
                const N: usize = 1024;
                let mut g = Graph::new("keep_warm");
                let x = g.input("x", Shape::new(&[1, N], DType::F32));
                let w = g.param("w", Shape::new(&[N, N], DType::F32));
                let y = g.matmul(x, w, Shape::new(&[1, N], DType::F32));
                g.set_outputs(vec![y]);
                let Ok(mut c) = std::panic::catch_unwind(|| Session::new(device).compile(g)) else {
                    return;
                };
                c.set_param("w", &vec![1e-3; N * N]);
                let xv = vec![1.0f32; N];
                while !stop2.load(Ordering::Relaxed) {
                    if on2.load(Ordering::Relaxed) {
                        c.run(&[("x", &xv)]);
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                }
            })
            .ok()?;
        Some(Self {
            on,
            stop,
            thread: Some(thread),
        })
    }

    fn set(&self, on: bool) {
        self.on.store(on, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for KeepWarm {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn receive_stage(group: &ProcessGroup) -> Result<(Stage, Vec<u8>)> {
    let bytes = group
        .transport()
        .recv_bytes(0, TAG_STAGE)
        .map_err(|e| anyhow::anyhow!("receive stage: {e}"))?;
    // A kept stage for another plan is dropped here, before the new one's
    // weights arrive: holding both, plus the compile, got an iPad's app
    // killed mid-transfer.
    let kept = KEPT_STAGE.with(|k| match k.borrow_mut().take() {
        Some((key, stage)) if key == bytes => Some(stage),
        _ => None,
    });
    let ack = |need: bool| {
        group
            .transport()
            .send_bytes(0, TAG_STAGE, &[u8::from(need)])
            .map_err(|e| anyhow::anyhow!("stage ack: {e}"))
    };
    if let Some(mut stage) = kept {
        ack(false)?;
        stage.reset();
        if let Some(l) = &stage.plan.layout {
            remember_layout(l);
        }
        return Ok((stage, bytes));
    }
    ack(true)?;
    let plan: StagePlan = serde_json::from_slice(&bytes)?;
    if let Some(l) = &plan.layout {
        remember_layout(l);
    }
    let params = recv_weights(group, 0)?;
    Ok((Stage::compile(plan, &params)?, bytes))
}

/// The logits row that predicts the token after `len` real ones: the whole
/// output when the function returns one row (`one_row`, it read `last`), else
/// row `len − 1` of `seq`.
/// A whole `dist_node MODE=model` worker rank, in the order every rank runs
/// it: a barrier once connected, [`exchange_speeds`], [`serve`], then the two
/// barriers `dist_node` ends with (after the job, and a final one that keeps
/// every rank's links up until all are done). A rank that is not `dist_node` (a phone, an embedded host) must
/// use this rather than [`serve`] alone — any collective out of that order
/// leaves the whole mesh waiting.
pub fn worker(group: &ProcessGroup, device: Device, log: bool) -> Result<String> {
    group
        .barrier()
        .map_err(|e| anyhow::anyhow!("initial barrier: {e}"))?;
    let speeds = exchange_speeds(group, device)?;
    if log {
        eprintln!(
            "rlx-node: rank {}/{} speeds (GFLOP/s) {speeds:.0?}; serving",
            group.rank(),
            group.world_size()
        );
    }
    let did = serve(group, log)?;
    for what in ["post-job", "final"] {
        group
            .barrier()
            .map_err(|e| anyhow::anyhow!("{what} barrier: {e}"))?;
    }
    Ok(did)
}

/// The second step of every `dist_node MODE=model` job, on every rank, after
/// the barrier each rank runs once connected: measure
/// this rank's fp32 GEMM throughput (GFLOP/s, a 1024³ matmul on `device`) and
/// all-gather it, so rank 0 can split a function by speed. Every rank must
/// call this before [`serve`] / [`InferJob::start`] / [`TrainJob::run`] — a
/// rank that skips it leaves the gather waiting forever.
pub fn exchange_speeds(group: &ProcessGroup, device: Device) -> Result<Vec<f64>> {
    exchange_speeds_links(group, device).map(|(s, _)| s)
}

/// [`exchange_speeds`], and the links between every pair of ranks
/// ([`probe_links`], whole on rank 0). Every rank runs both, in this order.
pub fn exchange_speeds_links(group: &ProcessGroup, device: Device) -> Result<(Vec<f64>, Links)> {
    let gflops = gemm_gflops(device) as f32;
    let now = group
        .all_gather(&[gflops, device_code(device)])
        .map_err(|e| anyhow::anyhow!("speed exchange: {e}"))?;
    let (speeds, order) = steady_speeds(group, &now)?;
    let mut links = probe_links(group)?;
    links.order = order;
    Ok((speeds, links))
}

/// How far, as a factor either way, a worker's new speed measurement may
/// move from the one a persistent node's last job planned by before the job
/// plans by the new ones. Only a gross change counts — a GPU fallen back to
/// its CPU, another job on it: the GEMM probe itself swings that much from
/// run to run on an iPad (Yellow iPad 333, 477, 242 GFLOP/s; msi and amd held
/// within 7%), and calibration measures the stages for real.
const SPEED_STEADY: f64 = 3.0;

/// What a persistent node remembers of its last job: the speed probe it
/// was planned against (`[speed, device code]` per rank) and the layout it
/// was deployed with ([`PlannedLayout`], calibrated or not).
#[derive(Default)]
struct Remembered {
    probe: Vec<f64>,
    layout: Option<PlannedLayout>,
}

static REMEMBERED: std::sync::Mutex<Option<Remembered>> = std::sync::Mutex::new(None);

/// Note the layout this process's stage (or, on rank 0, every stage) was
/// deployed with.
fn remember_layout(layout: &PlannedLayout) {
    if let Ok(mut m) = REMEMBERED.lock() {
        m.get_or_insert_with(Remembered::default).layout = Some(layout.clone());
    }
}

/// `device` as a number an `f32` holds exactly, for the speed exchange.
fn device_code(device: Device) -> f32 {
    let h = format!("{device:?}").bytes().fold(0x811c_9dc5u32, |h, b| {
        (h ^ b as u32).wrapping_mul(0x0100_0193)
    });
    (h & 0xff_ffff) as f32
}

/// The speeds to plan by, and the pipeline order when it is the last job's
/// too. The last job's layout — its speeds and order, so calibrated ones —
/// when every worker remembers the same, every rank is on the device it was
/// on, and no worker's new probe is more than a factor [`SPEED_STEADY`] off
/// the one that layout was planned against; else the new probe and no order.
/// A persistent worker keeps its stage only while the plan is the same, and
/// a probe off by a few percent moves the split. Rank 0's own number is not
/// compared: it is the rank started afresh each run, and its first GEMM
/// varied 402–1721 GFLOP/s from run to run on a busy mm4 while the workers'
/// held within 7%. `now` holds `[speed, device code]` per rank. Same answer
/// on every rank: each gets every rank's memory.
fn steady_speeds(group: &ProcessGroup, now: &[f32]) -> Result<(Vec<f64>, Option<Vec<u32>>)> {
    let n = now.len() / 2;
    let fresh: Vec<f64> = now.iter().map(|&v| v as f64).collect();
    // Probe (2n), then the layout's speeds (n) and order (n); zeros for none.
    let mine: Vec<f32> = {
        let m = REMEMBERED.lock().ok();
        let m = m.as_ref().and_then(|m| m.as_ref());
        let mut v = vec![0.0f32; 4 * n];
        if let Some(m) = m.filter(|m| m.probe.len() == 2 * n) {
            for (d, &x) in v.iter_mut().zip(&m.probe) {
                *d = x as f32;
            }
            if let Some(l) = m.layout.as_ref().filter(|l| l.speeds.len() == n) {
                for k in 0..n {
                    v[2 * n + k] = l.speeds[k] as f32;
                    v[3 * n + k] = l.order.get(k).copied().unwrap_or(0) as f32;
                }
            }
        }
        v
    };
    let all = group
        .all_gather(&mine)
        .map_err(|e| anyhow::anyhow!("speed exchange: {e}"))?;
    let remembered = |r: usize| &all[r * 4 * n..(r + 1) * 4 * n];
    let prev: Vec<f64> = remembered(1.min(n - 1)).iter().map(|&v| v as f64).collect();
    let steady = n > 1
        && (1..n).all(|r| remembered(r) == remembered(1))
        && (0..n).all(|r| prev[2 * r] > 0.0 && prev[2 * r + 1] == fresh[2 * r + 1])
        && (1..n)
            .all(|r| (fresh[2 * r] / prev[2 * r]).max(prev[2 * r] / fresh[2 * r]) <= SPEED_STEADY);
    let probe = if steady {
        prev[..2 * n].to_vec()
    } else {
        fresh
    };
    let planned: Vec<f64> = prev[2 * n..3 * n].to_vec();
    let order: Vec<u32> = prev[3 * n..].iter().map(|&v| v as u32).collect();
    let (speeds, order) =
        if steady && planned.iter().all(|&v| v > 0.0) && is_pipeline_order(&order, n) {
            (planned, Some(order))
        } else {
            (probe.iter().step_by(2).copied().collect(), None)
        };
    if let Ok(mut m) = REMEMBERED.lock() {
        m.get_or_insert_with(Remembered::default).probe = probe;
    }
    Ok((speeds, order))
}

/// The link from each rank to each other: one-way latency (half the mean
/// round trip of small messages, so Wi-Fi's jitter counts) and bandwidth
/// (a 64 KiB message each way, then 4 MiB where that would take under a
/// second, less the fastest round trip), in seconds and bytes/s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Links {
    pub latency: Vec<Vec<f64>>,
    pub bandwidth: Vec<Vec<f64>>,
    /// The pipeline order the last job ended with, set when the speeds the
    /// exchange returned are that job's ([`exchange_speeds_links`]): laid out
    /// in it, the step is that job's again.
    #[serde(default)]
    pub order: Option<Vec<u32>>,
}

impl Links {
    /// Seconds to send `bytes` from rank `a` to `b`.
    pub fn time(&self, a: usize, b: usize, bytes: f64) -> f64 {
        if a == b || self.latency.is_empty() {
            return 0.0;
        }
        self.latency[a][b] + bytes / self.bandwidth[a][b].max(1.0)
    }
}

const TAG_PROBE: u32 = 94;

/// Measure every pair's link, the lower rank of each pair timing it, and
/// gather the table on rank 0 (the others return theirs, partial).
pub fn probe_links(group: &ProcessGroup) -> Result<Links> {
    let n = group.world_size() as usize;
    let me = group.rank() as usize;
    let t = group.transport();
    let err = |e: rlx_driver::CollectiveError| anyhow::anyhow!("link probe: {e}");
    const ROUNDS: usize = 8;
    // Timed with a small message first: 4 MiB over a slow link (Bluetooth
    // LE, ~25 KB/s) would take minutes each way.
    const SMALL: usize = 64 << 10;
    const LARGE: usize = 4 << 20;
    let mut links = Links {
        latency: vec![vec![0.0; n]; n],
        bandwidth: vec![vec![0.0; n]; n],
        order: None,
    };
    for a in 0..n {
        for b in a + 1..n {
            let (a32, b32) = (a as u32, b as u32);
            if me == a {
                let (mut sum, mut fastest) = (0.0, f64::MAX);
                for _ in 0..ROUNDS {
                    let t0 = std::time::Instant::now();
                    t.send_bytes(b32, TAG_PROBE, &[0]).map_err(err)?;
                    t.recv_bytes(b32, TAG_PROBE).map_err(err)?;
                    let dt = t0.elapsed().as_secs_f64();
                    sum += dt;
                    fastest = fastest.min(dt);
                }
                let rtt = sum / ROUNDS as f64;
                // `a → b`: send `n` bytes, b acks; `b → a`: ask for `n`.
                let timed = |n: usize| -> Result<(f64, f64)> {
                    let t0 = std::time::Instant::now();
                    t.send_bytes(b32, TAG_PROBE, &vec![0u8; n]).map_err(err)?;
                    t.recv_bytes(b32, TAG_PROBE).map_err(err)?;
                    let up = (t0.elapsed().as_secs_f64() - fastest).max(1e-6);
                    let t0 = std::time::Instant::now();
                    t.send_bytes(b32, TAG_PROBE, &(n as u64).to_le_bytes())
                        .map_err(err)?;
                    t.recv_bytes(b32, TAG_PROBE).map_err(err)?;
                    let down = (t0.elapsed().as_secs_f64() - fastest).max(1e-6);
                    Ok((n as f64 / up, n as f64 / down))
                };
                // After each round, one byte: another (with LARGE) or done.
                let (mut up_bw, mut down_bw) = timed(SMALL)?;
                let again = (LARGE as f64) / up_bw.min(down_bw) < 1.0;
                t.send_bytes(b32, TAG_PROBE, &[u8::from(again)])
                    .map_err(err)?;
                if again {
                    (up_bw, down_bw) = timed(LARGE)?;
                    t.send_bytes(b32, TAG_PROBE, &[0]).map_err(err)?;
                }
                links.latency[a][b] = rtt / 2.0;
                links.latency[b][a] = rtt / 2.0;
                links.bandwidth[a][b] = up_bw;
                links.bandwidth[b][a] = down_bw;
            } else if me == b {
                for _ in 0..ROUNDS {
                    t.recv_bytes(a32, TAG_PROBE).map_err(err)?;
                    t.send_bytes(a32, TAG_PROBE, &[0]).map_err(err)?;
                }
                // Answer each timing round the initiator runs: ack what it
                // sends, send what it asks for, then read whether another.
                loop {
                    t.recv_bytes(a32, TAG_PROBE).map_err(err)?;
                    t.send_bytes(a32, TAG_PROBE, &[0]).map_err(err)?;
                    let ask = t.recv_bytes(a32, TAG_PROBE).map_err(err)?;
                    let n = ask
                        .get(..8)
                        .and_then(|b| b.try_into().ok())
                        .map(u64::from_le_bytes)
                        .unwrap_or(0) as usize;
                    t.send_bytes(a32, TAG_PROBE, &vec![0u8; n]).map_err(err)?;
                    if t.recv_bytes(a32, TAG_PROBE).map_err(err)? != [1] {
                        break;
                    }
                }
            }
        }
    }
    // Rank 0 takes each pair from its lower rank's table.
    if me != 0 {
        t.send_bytes(0, TAG_PROBE, &serde_json::to_vec(&links)?)
            .map_err(err)?;
        return Ok(links);
    }
    for a in 1..n {
        let theirs: Links =
            serde_json::from_slice(&t.recv_bytes(a as u32, TAG_PROBE).map_err(err)?)?;
        for b in a + 1..n {
            for (x, y) in [(a, b), (b, a)] {
                links.latency[x][y] = theirs.latency[x][y];
                links.bandwidth[x][y] = theirs.bandwidth[x][y];
            }
        }
    }
    Ok(links)
}

/// fp32 GFLOP/s of a 1024³ matmul on `device` (after a warm-up run).
pub fn gemm_gflops(device: Device) -> f64 {
    use rlx_ir::{DType, Shape};
    let n = 1024usize;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 97) as f32) * 0.01).collect();
    let mut g = Graph::new("gflops_probe");
    let ain = g.input("a", Shape::new(&[n, n], DType::F32));
    let bp = g.param("b", Shape::new(&[n, n], DType::F32));
    let mm = g.matmul(ain, bp, Shape::new(&[n, n], DType::F32));
    g.set_outputs(vec![mm]);
    let mut c = Session::new(device).compile(g);
    c.set_param("b", &a);
    let _ = c.run(&[("a", a.as_slice())]);
    let iters = 5;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let _ = c.run(&[("a", a.as_slice())]);
    }
    let per = t.elapsed().as_secs_f64() / iters as f64;
    2.0 * (n as f64).powi(3) / per / 1e9
}

pub fn next_token_row(logits: &[f32], seq: usize, len: usize, one_row: bool) -> &[f32] {
    if one_row || !logits.len().is_multiple_of(seq) {
        return logits;
    }
    let vocab = logits.len() / seq;
    &logits[(len - 1) * vocab..len * vocab]
}

/// The function run whole on one device (the reference a split must match).
pub fn run_local(
    path: &str,
    function: &str,
    dims: &Map<String, Value>,
    device: Device,
    inputs: &[(&str, &[f32])],
) -> Result<Vec<Vec<f32>>> {
    let mut pkg = Opened::open_merged(std::path::Path::new(path))?;
    let graph = specialize_function(&mut pkg, function, Some(dims))?;
    let mut c = Session::new(device).compile(graph.clone());
    for (name, _, d) in params_of(&graph) {
        c.set_param(&name, &pkg.weights.values(&name, &d)?);
    }
    Ok(c.run(inputs))
}

/// Release the workers without a job (rank 0 had nothing to distribute).
pub fn dismiss(group: &ProcessGroup) -> Result<()> {
    announce(group, "none")
}

// ── worker ──────────────────────────────────────────────────────────────────

/// A worker's whole job: wait for rank 0's job and do it (a training run, or
/// an inference stage until rank 0 stops it). Returns what it did, for logs.
pub fn serve(group: &ProcessGroup, log: bool) -> Result<String> {
    let rank = group.rank();
    if rank == 0 {
        bail!("rank 0 coordinates (TrainJob / InferJob); serve is for workers");
    }
    let h: Value = serde_json::from_slice(
        &group
            .transport()
            .recv_bytes(0, TAG_JOB)
            .map_err(|e| anyhow::anyhow!("wait for a job: {e}"))?,
    )?;
    match h.get("kind").and_then(Value::as_str) {
        Some("train") => {
            let spec = crate::dist::recv_train(group).map_err(anyhow::Error::msg)?;
            let (m, _) = crate::dist::run_train_group(&spec, group, crate::dist::uri_resolver, log)
                .map_err(anyhow::Error::msg)?;
            Ok(format!(
                "trained {} samples on {:?}: loss {:.4} → {:.4}, {} syncs ({} skipped on overflow)",
                m.samples, m.lanes, m.first_loss, m.last_loss, m.syncs, m.skipped
            ))
        }
        Some("infer") => {
            let (mut stage, mut key) = receive_stage(group)?;
            let mut nodes = stage.plan.graph.len();
            let mut next = stage.plan.next.unwrap_or((rank + 1) % group.world_size());
            let mut prev = stage.plan.prev.unwrap_or(rank - 1);
            let wire = stage.plan.wire;
            let mut served = 0usize;
            // Where a step's time goes on this rank: waiting for the previous
            // stage, computing, sending on.
            let (mut wait, mut compute, mut send, mut bytes) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            let mut first = 0.0f64;
            let warm = KeepWarm::start(stage.device);
            loop {
                let t = std::time::Instant::now();
                if let Some(w) = &warm {
                    w.set(true);
                }
                let got = recv_values(group, prev);
                if let Some(w) = &warm {
                    w.set(false);
                }
                let Some(v) = got? else {
                    break;
                };
                // Calibration laid the step out again: pass the marker on, then
                // take this rank's new stage (its neighbours may change). The
                // current one goes in as the kept stage, so an unchanged plan
                // sends no weights and a changed one frees it first.
                if v.contains_key(RESTAGE) {
                    send_values(group, next, Some(&v), stage.plan.wire)?;
                    KEPT_STAGE.with(|k| *k.borrow_mut() = Some((key, stage)));
                    (stage, key) = receive_stage(group)?;
                    nodes = stage.plan.graph.len();
                    next = stage.plan.next.unwrap_or((rank + 1) % group.world_size());
                    prev = stage.plan.prev.unwrap_or(rank - 1);
                    continue;
                }
                wait += t.elapsed().as_secs_f64();
                bytes += v.values().map(|x| x.len() * 4).sum::<usize>();
                let t = std::time::Instant::now();
                let out = stage.step(v)?;
                let c = t.elapsed().as_secs_f64();
                if served == 0 {
                    // The first step pays one-time costs (kernel compiles,
                    // uploads); reported apart from the steady state.
                    first = c;
                } else {
                    compute += c;
                }
                let t = std::time::Instant::now();
                send_values(group, next, Some(&out), wire)?;
                send += t.elapsed().as_secs_f64();
                served += 1;
            }
            // Pass the stop on (the last stage's next is rank 0, which knows).
            if next != 0 {
                send_values(group, next, None, wire)?;
            }
            // Kept for the next job, should it bring the same plan.
            KEPT_STAGE.with(|k| *k.borrow_mut() = Some((key, stage)));
            let per = |x: f64| 1e3 * x / served.max(1) as f64;
            Ok(format!(
                "served {served} request(s) through a {nodes}-node stage — first step {:.0} ms, \
                 then per step: compute {:.1} ms, send {:.1} ms, waiting {:.1} ms; {:.1} KB in",
                1e3 * first,
                1e3 * compute / served.saturating_sub(1).max(1) as f64,
                per(send),
                per(wait),
                bytes as f64 / served.max(1) as f64 / 1e3
            ))
        }
        Some("none") => Ok("no job".into()),
        other => bail!("unknown job {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    /// The verify loop against greedy one-token decoding, through a model
    /// whose next token is a function of the last (`next`) and whose draft
    /// head is wrong whenever the token it extends is a multiple of 3. Each
    /// pass checks its cache length against the rows kept so far.
    #[test]
    fn verify_decoding_gives_the_greedy_tokens() {
        const V: usize = 10;
        let next = |t: u32| (t * 3 + 1) % V as u32;
        let onehot = |t: u32| {
            let mut r = vec![0.0f32; V];
            r[t as usize] = 1.0;
            r
        };
        let (mut drafts, mut accepted) = (0, 0);
        for prompt in [
            vec![4u32],
            vec![4, 7],
            vec![2, 5, 8],
            vec![1, 2, 3, 4, 5, 6],
        ] {
            let new = 12;
            let mut want = Vec::new();
            let mut t = *prompt.last().unwrap();
            for _ in 0..new {
                t = next(t);
                want.push(t);
            }
            let (mut kept, mut last) = (0usize, 0usize);
            let (got, st) = verify_decode(&prompt, new, &[], |past, toks, keep| {
                kept += keep.min(last);
                assert_eq!(past, kept, "a pass ran at a cache length not kept");
                last = toks.len();
                let mut logits = onehot(next(toks[0]));
                logits.extend(onehot(next(toks[1])));
                let after = next(next(toks[1]));
                let draft = if toks[1] % 3 == 0 {
                    (after + 1) % V as u32
                } else {
                    after
                };
                let rows = picks(&logits, 2, V);
                Ok(([rows[0], rows[1]], picks(&onehot(draft), 1, V)[0]))
            })
            .unwrap();
            assert_eq!(got, want, "prompt {prompt:?}");
            assert!(st.passes < prompt.len() + new, "{st:?}");
            (drafts, accepted) = (drafts + st.drafts, accepted + st.accepted);
        }
        // Both outcomes were taken.
        assert!(accepted > 0 && accepted < drafts, "{accepted}/{drafts}");
    }

    use super::*;
    use rlx_ir::{DType, GraphExt, Shape};

    /// `u = s + relu(x)` is the state's update and `y = u·x` reads it in the
    /// same layer. An even split by cost cuts between them; the state span
    /// keeps them on one stage, so the update never has to travel.
    #[test]
    fn the_layout_routes_round_a_slow_link() {
        let f = Shape::new(&[1, 64], DType::F32);
        let mut g = Graph::new("chain");
        let mut h = g.input("x", f);
        for _ in 0..12 {
            h = g.relu(h);
        }
        g.set_outputs(vec![h]);
        // 0 → 1 takes a second; every other link is quick.
        let mut links = Links {
            latency: vec![vec![1e-4; 3]; 3],
            bandwidth: vec![vec![1e9; 3]; 3],
            order: None,
        };
        links.latency[0][1] = 1.0;
        let (order, plans, est) =
            choose_layout(&g, &[1.0, 1.0, 1.0], &[], &HashSet::new(), Some(&links)).unwrap();
        assert_eq!(order, vec![0, 2, 1], "rank order would cross the slow link");
        assert_eq!(plans.len(), 3);
        let (t, base) = est.unwrap();
        assert!(t < base && base > 1.0, "{t} vs {base}");
        // Without links: rank order.
        let (order, _, est) = choose_layout(&g, &[1.0; 3], &[], &HashSet::new(), None).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
        assert!(est.is_none());
    }

    #[test]
    fn argmax_outputs_give_ids_over_the_last_vocab_entries() {
        let mut g = Graph::new("lm");
        let x = g.input("x", Shape::new(&[1, 2, 6], DType::F32));
        let y = g.relu(x);
        g.set_outputs(vec![y]);
        argmax_outputs(&mut g, &[0], 4);
        let out = g.node(g.outputs[0]);
        assert!(matches!(out.op, Op::ArgMax { axis: 2, .. }), "{:?}", out.op);
        assert_eq!(out.shape.num_elements(), Some(2));
        let mut c = crate::Session::new(Device::Cpu).compile(g);
        // Row 0's largest is in its first two (dropped) entries.
        let x = [9.0, 9.0, 1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 5.0, 1.0, 7.0, 2.0];
        let ids = c.run(&[("x", &x)]).remove(0);
        assert_eq!(ids, vec![1.0, 2.0]);
    }

    #[test]
    fn a_split_keeps_a_state_update_with_its_readers() {
        let f = Shape::new(&[1, 4], DType::F32);
        let mut g = Graph::new("decode");
        let x = g.input("x", f.clone());
        let s = g.input("s", f);
        let a = g.relu(x);
        let u = g.add(s, a);
        let y = g.mul(u, x);
        let z = g.relu(y);
        g.set_outputs(vec![z, u]);
        let u_name = value_name(&g, u);
        let crosses = |plans: &[StagePlan]| plans.iter().any(|p| p.reads.contains(&u_name));

        let plain = split_function(&g, &[1.0, 1.0]).unwrap();
        assert!(
            crosses(&plain),
            "the unconstrained split should cut after the update"
        );

        let spans = state_spans(&g, &[("s".into(), 1, false)]);
        let kept = split_function_keeping(&g, &[1.0, 1.0], &spans).unwrap();
        assert!(
            !crosses(&kept),
            "the update crossed a stage: {:?}",
            kept[1].reads
        );
        assert!(kept[1].graph.len() > 1, "the second stage got no work");
    }
}
