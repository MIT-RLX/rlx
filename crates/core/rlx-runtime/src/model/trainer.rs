// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Training a package: any function of an `.rlxp` (or `.rlxmodel`), its
//! loss, the params it trains and the optimizer, with nothing about the
//! model it is.
//!
//! ```text
//! open_trainer("model.rlxp", {
//!   function: "forward",                 // a package function
//!   dims: {seq: 128},                    // its symbols, sized
//!   loss: {cross_entropy: {logits: "logits", labels: "labels", mask: "mask"}}
//!       | {mse: {output: "y", target: "target"}}
//!       | {output: "loss"},              // the function computes its own
//!   train: "all"                          // every param
//!        | {params: ["*.q_proj.*", …]}    // those whose names match
//!        | {lora: {targets: ["*proj*"], rank: 8, alpha: 16}}
//!        | {sae: {tap: "layer3", width: 4096, l1: 1e-3}},
//!   optimizer: {lr: 1e-4, beta1, beta2, eps, weight_decay},
//! })
//! ```
//!
//! The package's model block may hold the same object as `train`, its
//! defaults; the options given win, key by key. LoRA is a rewrite of the
//! function: each matmul whose weight is a matching param gains `+ (α/r) ·
//! (x·D)·U`, `D` `[in, r]` small and random, `U` `[r, out]` zero, so the
//! model is unchanged before the first step. The trainer runs on
//! [`crate::train::ResidentTrainer`] (forward, backward and AdamW one
//! graph); [`PackageTrainer::save_adapter`] writes the LoRA factors as an
//! adapter package over the trained one, which opens as the base with them
//! merged, in the layout each tensor is stored in.
//!
//! A sparse autoencoder trains on one activation of the function, the node
//! `tap` names (its `name`, or `#id`), the base frozen: `z = relu((h −
//! b_dec)·W_enc + b_enc)`, `ĥ = z·W_dec + b_dec`, the loss `mean((ĥ − h)²) +
//! l1 · mean(Σ|z|)` (no `loss` given). [`PackageTrainer::save_sae`] writes
//! it as a package of its own, its function `sae` (`h` → `features`,
//! `reconstruction`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_ir::{DType, Dim, Graph, GraphExt, NodeId, Op, Shape};
use serde_json::{Map, Value};

use super::Tensor;
use super::rlxp::{LoraUpdate, Opened, params_of, save_adapter};
use super::sae::{SAE_PARAMS, sae_apply};
use crate::train::{AdamSpec, ResidentTrainer, TrainableParam};

/// A LoRA site: the param it adapts, its factors' names, and how the
/// matmul reads the param.
struct Lora {
    param: String,
    /// The matmul reads the param transposed (`x · Pᵀ`).
    transposed: bool,
    down: String,
    up: String,
    /// The matmul's `[in, out]`.
    k: usize,
    n: usize,
}

/// A package's function made trainable ([`prepare`]): the graph whose only
/// output is the loss (its integer inputs fed as f32), the params to train,
/// every param's starting values, the optimizer options — what any trainer
/// takes — and what writes the trained values back as a package.
pub struct Prepared {
    pub graph: Graph,
    pub trainable: Vec<String>,
    pub initial: HashMap<String, Vec<f32>>,
    /// The options' `optimizer` (`{lr, beta1, beta2, eps, weight_decay, …}`).
    pub optimizer: Value,
    /// The graph's inputs (the loss's included), by name, and their dims.
    pub inputs: Vec<(String, Vec<usize>)>,
    saves: Saves,
}

/// How trained values become a package.
struct Saves {
    lora: Vec<Lora>,
    rank: usize,
    scale: f32,
    /// What stores each adapted param, as an adapter names it.
    layout: HashMap<String, (String, bool)>,
    base: PathBuf,
    /// The trained params' dims.
    shapes: HashMap<String, Vec<usize>>,
    /// The SAE trained, if one is: its tap and dims.
    sae: Option<Sae>,
}

/// A package being trained on [`ResidentTrainer`] (see the module docs).
pub struct PackageTrainer {
    trainer: ResidentTrainer,
    inputs: Vec<(String, Vec<usize>)>,
    saves: Saves,
}

/// A sparse autoencoder on an activation.
struct Sae {
    tap: String,
    d: usize,
    width: usize,
    l1: f32,
}

/// `g` cut at its node `tap` and that activation through an SAE, the loss
/// its only output.
fn with_sae(g: &Graph, tap: &str, width: usize, l1: f32) -> Result<(Graph, usize)> {
    let id = super::sae::find_tap(g, tap)?
        .with_context(|| format!("the function has no node named '{tap}'"))?;
    let mut out = g.clone();
    let dims = static_dims(&out.node(id).shape)?;
    let d = *dims.last().context("the tap has a feature axis")?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let h = out.reshape_(id, vec![rows as i64, d as i64]);
    let (z, r) = sae_apply(&mut out, h, d, width);
    let e = out.sub(r, h);
    let e2 = out.mul(e, e);
    let mse = out.mean(e2, vec![0, 1], false);
    let a = out.abs(z);
    let per_row = out.sum(a, vec![1], false);
    let sparsity = out.mean(per_row, vec![0], false);
    let k = scalar(&mut out, l1);
    let sparsity = out.mul(sparsity, k);
    let total = out.add(mse, sparsity);
    let total = out.reshape_(total, vec![1]);
    out.node_mut(total).name = Some("loss".into());
    out.set_outputs(vec![total]);
    Ok((rlx_ir::retarget::prune(&out), d))
}

/// `pattern` (`*` any run of characters) matches `name`.
fn glob(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let mut rest = name;
    for (i, part) in parts.iter().enumerate() {
        if i == 0 {
            match rest.strip_prefix(part) {
                Some(r) => rest = r,
                None => return false,
            }
        } else if i == parts.len() - 1 {
            return rest.ends_with(part);
        } else {
            match rest.find(part) {
                Some(at) => rest = &rest[at + part.len()..],
                None => return false,
            }
        }
    }
    true
}

fn patterns(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

fn static_dims(shape: &Shape) -> Result<Vec<usize>> {
    shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Ok(*n),
            Dim::Dynamic(_) => bail!("a dim is unsized: give the function's `dims`"),
        })
        .collect()
}

/// The defaults of the model block's `train` under `options`, key by key.
fn merged(meta: &Value, options: &Value) -> Value {
    let mut out: Map<String, Value> = meta
        .get("train")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(o) = options.as_object() {
        for (k, v) in o {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

fn output_named(g: &Graph, name: &str) -> Result<NodeId> {
    g.outputs
        .iter()
        .copied()
        .find(|&o| g.node(o).name.as_deref() == Some(name))
        .with_context(|| format!("the function has no output '{name}'"))
}

fn scalar(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

/// `g` with each matmul whose weight is a param matching `targets` adapted
/// by a low-rank update `scale · (x·D)·U`; the sites.
fn with_lora(g: &Graph, targets: &[String], rank: usize, scale: f32) -> Result<(Graph, Vec<Lora>)> {
    let mut sites = Vec::new();
    let mut out = rlx_ir::stateful::rebuild(g, |out, _, node, inputs| {
        if !matches!(node.op, Op::MatMul) {
            return Ok(None);
        }
        let w = g.node(node.inputs[1]);
        let (param, transposed) = match &w.op {
            Op::Param { name } => (name.clone(), false),
            Op::Transpose { perm } if perm == &[1, 0] => match &g.node(w.inputs[0]).op {
                Op::Param { name } => (name.clone(), true),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        if !targets.iter().any(|t| glob(t, &param)) || w.shape.rank() != 2 {
            return Ok(None);
        }
        let dims = static_dims(&w.shape).map_err(|e| e.to_string())?;
        let (k, n) = (dims[0], dims[1]);
        let i = sites.len();
        let (down, up) = (format!("lora.{i}.down"), format!("lora.{i}.up"));
        let y = out.mm(inputs[0], inputs[1]);
        let x_shape = out.node(inputs[0]).shape.clone();
        let x_dims = static_dims(&x_shape).map_err(|e| e.to_string())?;
        let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
        let x2 = out.reshape_(inputs[0], vec![rows as i64, k as i64]);
        let d = out.param(down.clone(), Shape::new(&[k, rank], DType::F32));
        let u = out.param(up.clone(), Shape::new(&[rank, n], DType::F32));
        let z = out.mm(x2, d);
        let z = out.mm(z, u);
        let s = scalar(out, scale);
        let z = out.mul(z, s);
        let mut y_dims: Vec<i64> = x_dims[..x_dims.len() - 1]
            .iter()
            .map(|&v| v as i64)
            .collect();
        y_dims.push(n as i64);
        let z = out.reshape_(z, y_dims);
        sites.push(Lora {
            param,
            transposed,
            down,
            up,
            k,
            n,
        });
        let sum = out.add(y, z);
        out.node_mut(sum).name = node.name.clone();
        Ok(Some(sum))
    })
    .map_err(anyhow::Error::msg)?;
    out.name = g.name.clone();
    if sites.is_empty() {
        bail!("no matmul weight matches the LoRA targets {targets:?}");
    }
    Ok((out, sites))
}

/// `g` with each integer input (token ids, class ids) fed as f32 and cast
/// back inside: the trainer's batches are f32, exact for ids below 2²⁴.
fn with_float_inputs(g: &Graph) -> Result<Graph> {
    let mut out = rlx_ir::stateful::rebuild(g, |out, _, node, _| match &node.op {
        Op::Input { name } if node.shape.dtype().is_int() => {
            let dims = static_dims(&node.shape).map_err(|e| e.to_string())?;
            let x = out.input(name.clone(), Shape::new(&dims, DType::F32));
            Ok(Some(out.add_node(
                Op::Cast {
                    to: node.shape.dtype(),
                },
                vec![x],
                node.shape.clone(),
            )))
        }
        _ => Ok(None),
    })
    .map_err(anyhow::Error::msg)?;
    out.name = g.name.clone();
    Ok(out)
}

/// `g`'s scalar loss as its only output, as `loss` says.
fn with_loss(mut g: Graph, loss: &Value) -> Result<Graph> {
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let total = if let Some(name) = loss.get("output").and_then(Value::as_str) {
        let o = output_named(&g, name)?;
        let dims = static_dims(&g.node(o).shape)?;
        let all: Vec<usize> = (0..dims.len()).collect();
        g.mean(o, all, false)
    } else if let Some(ce) = loss.get("cross_entropy") {
        let field = |k: &str| ce.get(k).and_then(Value::as_str);
        let o = output_named(
            &g,
            field("logits").context("cross_entropy names its `logits`")?,
        )?;
        let dims = static_dims(&g.node(o).shape)?;
        let c = *dims.last().context("logits have a class axis")?;
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let logits = g.reshape_(o, vec![rows as i64, c as i64]);
        let labels = g.input(
            field("labels").context("cross_entropy names its `labels` input")?,
            f32s(&[rows]),
        );
        let per = g.add_node(
            Op::SoftmaxCrossEntropyWithLogits,
            vec![logits, labels],
            f32s(&[rows]),
        );
        match field("mask") {
            // Masked rows (0) count for nothing: `Σ ce·m / Σ m`.
            Some(m) => {
                let mask = g.input(m, f32s(&[rows]));
                let kept = g.mul(per, mask);
                let num = g.sum(kept, vec![0], false);
                let den = g.sum(mask, vec![0], false);
                g.div(num, den)
            }
            None => g.mean(per, vec![0], false),
        }
    } else if let Some(mse) = loss.get("mse") {
        let field = |k: &str| mse.get(k).and_then(Value::as_str);
        let o = output_named(&g, field("output").context("mse names its `output`")?)?;
        let shape = g.node(o).shape.clone();
        let t = g.input(
            field("target").context("mse names its `target` input")?,
            shape.clone(),
        );
        let d = g.sub(o, t);
        let d2 = g.mul(d, d);
        let all: Vec<usize> = (0..shape.rank()).collect();
        g.mean(d2, all, false)
    } else {
        bail!("`loss` is {{output}}, {{cross_entropy}} or {{mse}}");
    };
    let total = g.reshape_(total, vec![1]);
    g.node_mut(total).name = Some("loss".into());
    g.set_outputs(vec![total]);
    Ok(g)
}

/// Uniform in `(-bound, bound)`, seeded (LoRA's `D`).
fn uniform(n: usize, bound: f32, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * bound
        })
        .collect()
}

/// Prepare the package at `path` for training, as `options` (over the model
/// block's `train`) say (see the module docs).
pub fn prepare(path: &str, options: &Value) -> Result<Prepared> {
    let base = PathBuf::from(path);
    let mut pkg = Opened::open_merged(&base)?;
    let o = merged(&pkg.meta, options);
    let function = o
        .get("function")
        .and_then(Value::as_str)
        .context("the trainer names its `function`")?;
    let mut graph = pkg.take_graph(function)?;
    if let Some(dims) = o.get("dims").and_then(Value::as_object) {
        let mut bind = rlx_ir::DimBinding::new();
        for (name, size) in dims {
            let sym = pkg
                .dims
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, s)| *s)
                .with_context(|| format!("the package has no dim '{name}'"))?;
            bind.set(
                sym,
                size.as_u64().context("a dim's size is a whole number")? as usize,
            );
        }
        graph = rlx_ir::dynamic::specialize(&graph, &bind);
    }
    if graph.has_dynamic_dims() {
        bail!("'{function}' still has unsized dims: give them in `dims`");
    }
    let mut graph = with_float_inputs(&graph)?;

    let train = o
        .get("train")
        .cloned()
        .unwrap_or(Value::String("all".into()));
    let (mut rank, mut scale, mut lora) = (0, 1.0, Vec::new());
    if let Some(l) = train.get("lora") {
        rank = l.get("rank").and_then(Value::as_u64).unwrap_or(8) as usize;
        let alpha = l
            .get("alpha")
            .and_then(Value::as_f64)
            .unwrap_or(rank as f64) as f32;
        scale = alpha / rank as f32;
        let targets = patterns(&l["targets"]);
        if targets.is_empty() {
            bail!("lora names its `targets`");
        }
        let (g, sites) = with_lora(&graph, &targets, rank, scale)?;
        graph = g;
        lora = sites;
    }
    let mut sae = None;
    let graph = if let Some(cfg) = train.get("sae") {
        let tap = cfg
            .get("tap")
            .and_then(Value::as_str)
            .context("sae names its `tap`")?
            .to_string();
        let width = cfg
            .get("width")
            .and_then(Value::as_u64)
            .context("sae gives its `width`")? as usize;
        let l1 = cfg.get("l1").and_then(Value::as_f64).unwrap_or(1e-3) as f32;
        let (g, d) = with_sae(&graph, &tap, width, l1)?;
        sae = Some(Sae { tap, d, width, l1 });
        g
    } else {
        let loss = o.get("loss").context("the trainer names its `loss`")?;
        with_loss(graph, loss)?
    };

    // The params: the package's values, the factors initialized.
    let mut initial = HashMap::new();
    let lora_names: HashMap<&str, &Lora> = lora
        .iter()
        .flat_map(|l| [(l.down.as_str(), l), (l.up.as_str(), l)])
        .collect();
    for (name, _, dims) in params_of(&graph) {
        if let Some(sae) = &sae
            && let Some(i) = SAE_PARAMS.iter().position(|p| *p == name)
        {
            let (d, w) = (sae.d, sae.width);
            let enc = uniform(d * w, 1.0 / (d as f32).sqrt(), 0x5ae);
            let values = match i {
                0 => enc,
                // The decoder starts as the encoder's transpose.
                2 => transpose(&enc, d, w),
                1 => vec![0.0; w],
                _ => vec![0.0; d],
            };
            initial.insert(name, values);
            continue;
        }
        let values = match lora_names.get(name.as_str()) {
            Some(l) if name == l.down => {
                let seed = 0x5eed ^ (initial.len() as u64).wrapping_mul(0x9e37_79b9);
                uniform(l.k * rank, 1.0 / (l.k as f32).sqrt(), seed)
            }
            Some(l) => vec![0.0; rank * l.n],
            None => pkg.weights.values(&name, &dims)?,
        };
        initial.insert(name, values);
    }
    let wanted = |name: &str| -> bool {
        if sae.is_some() {
            return SAE_PARAMS.contains(&name);
        }
        if !lora.is_empty() {
            return lora_names.contains_key(name);
        }
        match &train {
            Value::String(s) if s == "all" => true,
            v => patterns(&v["params"]).iter().any(|p| glob(p, name)),
        }
    };
    let mut wrt = Vec::new();
    for n in graph.nodes() {
        if let Op::Param { name } = &n.op
            && wanted(name)
            && !wrt.iter().any(|w: &TrainableParam| &w.name == name)
        {
            wrt.push(TrainableParam {
                name: name.clone(),
                node: n.id,
            });
        }
    }
    if wrt.is_empty() {
        bail!("no param of '{function}' is trained as `train` says");
    }
    let shapes = params_of(&graph)
        .into_iter()
        .filter(|(n, _, _)| wrt.iter().any(|w| &w.name == n))
        .map(|(n, _, d)| (n, d))
        .collect();
    let inputs = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some((name.clone(), static_dims(&n.shape).ok()?)),
            _ => None,
        })
        .collect();
    let layout = lora
        .iter()
        .map(|l| (l.param.clone(), pkg.weights.stored_layout(&l.param)))
        .collect();
    Ok(Prepared {
        trainable: wrt.into_iter().map(|w| w.name).collect(),
        graph,
        initial,
        optimizer: o.get("optimizer").cloned().unwrap_or_default(),
        inputs,
        saves: Saves {
            lora,
            rank,
            scale,
            layout,
            base,
            shapes,
            sae,
        },
    })
}

/// Open `path` to train it (see [`prepare`]) on a [`ResidentTrainer`]
/// (AdamW) on the CPU.
pub fn open_trainer(path: &str, options: &Value) -> Result<PackageTrainer> {
    let p = prepare(path, options)?;
    let opt = &p.optimizer;
    let f = |k: &str, d: f32| opt.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32);
    let d = AdamSpec::default();
    let spec = AdamSpec {
        lr: f("lr", d.lr),
        beta1: f("beta1", d.beta1),
        beta2: f("beta2", d.beta2),
        eps: f("eps", d.eps),
        weight_decay: f("weight_decay", d.weight_decay),
    };
    let wrt: Vec<TrainableParam> = p
        .trainable
        .iter()
        .map(|name| {
            let node = p
                .graph
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, Op::Param { name: m } if m == name))
                .map(|n| n.id)
                .expect("a trainable param is a param of the graph");
            TrainableParam {
                name: name.clone(),
                node,
            }
        })
        .collect();
    let trainer = ResidentTrainer::new(&p.graph, &wrt, &p.initial, &spec, crate::Device::Cpu)
        .map_err(|e| anyhow::anyhow!("'{}' cannot be trained: {e:?}", p.graph.name))?;
    Ok(PackageTrainer {
        trainer,
        inputs: p.inputs,
        saves: p.saves,
    })
}

/// A batch tensor's values as f32: integers (ids) widened.
fn as_f32(t: &Tensor) -> Result<Vec<f32>> {
    let d = &t.data;
    Ok(match t.dtype {
        DType::I32 => d
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect(),
        DType::I64 => d
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as f32)
            .collect(),
        DType::U32 => d
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect(),
        DType::U8 => d.iter().map(|&b| f32::from(b)).collect(),
        _ => t.to_f32()?,
    })
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = v[r * cols + c];
        }
    }
    out
}

impl PackageTrainer {
    /// The function's inputs (the loss's included), by name, and their dims.
    pub fn inputs(&self) -> &[(String, Vec<usize>)] {
        &self.inputs
    }

    /// The params trained.
    pub fn trainable(&self) -> Vec<String> {
        self.trainer.trainable().map(String::from).collect()
    }

    fn feeds(&self, batch: &[Tensor]) -> Result<Vec<(String, Vec<f32>)>> {
        self.inputs
            .iter()
            .map(|(name, dims)| {
                let t = batch
                    .iter()
                    .find(|t| &t.name == name)
                    .with_context(|| format!("the batch has no '{name}'"))?;
                let v = as_f32(t)?;
                let want: usize = dims.iter().product();
                if v.len() != want {
                    bail!(
                        "'{name}' has {} values; the function takes {dims:?}",
                        v.len()
                    );
                }
                Ok((name.clone(), v))
            })
            .collect()
    }

    /// One optimizer step on `batch`; the loss before it.
    pub fn step(&mut self, batch: &[Tensor]) -> Result<f32> {
        let feeds = self.feeds(batch)?;
        let refs: Vec<(&str, &[f32])> = feeds
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let outs = self.trainer.step(&refs);
        outs.first()
            .and_then(|v| v.first())
            .copied()
            .context("the step returned no loss")
    }

    /// The loss on `batch`, nothing trained.
    pub fn loss(&mut self, batch: &[Tensor]) -> Result<f32> {
        let feeds = self.feeds(batch)?;
        let refs: Vec<(&str, &[f32])> = feeds
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let outs = self.trainer.forward_only(&refs);
        outs.first()
            .and_then(|v| v.first())
            .copied()
            .context("the forward returned no loss")
    }

    /// The trained params' values.
    pub fn params(&mut self) -> HashMap<String, Vec<f32>> {
        self.trainer.params()
    }

    /// Write the package trained (see [`Prepared::save`]).
    pub fn save(&mut self, out: &Path) -> Result<()> {
        let trained = self.trainer.params();
        self.saves.package(&trained, out)
    }

    /// Write the SAE trained (see [`Prepared::save`]).
    pub fn save_sae(&mut self, out: &Path, name: &str) -> Result<()> {
        let trained = self.trainer.params();
        self.saves.sae_package(&trained, out, name)
    }

    /// Write the LoRA factors as an adapter package (see [`Prepared::save`]).
    pub fn save_adapter(&mut self, out: &Path, name: &str) -> Result<()> {
        let trained = self.trainer.params();
        self.saves.adapter(&trained, out, name)
    }
}

/// What writes a prepared package's trained values back ([`Prepared::save`]),
/// kept by a trainer once the graph and values are its own.
pub struct Saver(Saves);

impl Saver {
    /// See [`Prepared::save`].
    pub fn save(&self, trained: &HashMap<String, Vec<f32>>, out: &Path, name: &str) -> Result<()> {
        self.0.save(trained, out, name)
    }
}

impl Prepared {
    /// The graph, the params to train, the starting values and the
    /// optimizer options for a trainer, and what saves its results.
    #[allow(clippy::type_complexity)]
    pub fn split(self) -> (Graph, Vec<String>, HashMap<String, Vec<f32>>, Value, Saver) {
        (
            self.graph,
            self.trainable,
            self.initial,
            self.optimizer,
            Saver(self.saves),
        )
    }

    /// Write `trained` (the trainable params' values) at `out` as what was
    /// trained makes: a LoRA trainer's factors an adapter package over the
    /// package trained (`name` its name); an SAE a package of its own; else
    /// the package trained, the trained values in place of what it had.
    pub fn save(&self, trained: &HashMap<String, Vec<f32>>, out: &Path, name: &str) -> Result<()> {
        self.saves.save(trained, out, name)
    }
}

impl Saves {
    fn save(&self, trained: &HashMap<String, Vec<f32>>, out: &Path, name: &str) -> Result<()> {
        if !self.lora.is_empty() {
            self.adapter(trained, out, name)
        } else if self.sae.is_some() {
            self.sae_package(trained, out, name)
        } else {
            self.package(trained, out)
        }
    }

    /// Write the package trained, its trained params' values in place of
    /// what it had, at `out`: every function, the model block and its
    /// sidecars; params it references stay references (to the same files,
    /// by absolute path) but for those trained, which it now holds.
    fn package(&self, trained: &HashMap<String, Vec<f32>>, out: &Path) -> Result<()> {
        let pkg = Opened::open(&self.base)?;
        if pkg.meta.get("adapter").is_some() {
            bail!("an adapter package saves its training as another adapter");
        }
        let mut tensors: Vec<Tensor> = pkg
            .weights
            .held()
            .into_iter()
            .filter(|t| !trained.contains_key(&t.name))
            .collect();
        let mut names: Vec<&String> = trained.keys().collect();
        names.sort();
        for n in names {
            let dims = self
                .shapes
                .get(n)
                .cloned()
                .unwrap_or_else(|| vec![trained[n].len()]);
            tensors.push(Tensor::f32(n.clone(), dims, &trained[n]));
        }
        let mut model = pkg.meta.clone();
        if let Some(sources) = model.get_mut("weights").and_then(Value::as_array_mut) {
            for s in sources {
                if let Some(uri) = s.get("uri").and_then(Value::as_str) {
                    s["uri"] = Value::String(super::rlxp::source_uri(&pkg.path, uri));
                }
            }
        }
        let sidecars: Vec<(String, String, Vec<u8>)> = pkg
            .pack
            .sidecar_ids()
            .into_iter()
            .filter(|(id, _)| id != "model.json")
            .map(|(id, media)| {
                let bytes = pkg.file(&format!("sidecar:{id}"))?;
                Ok((id, media.unwrap_or_default(), bytes))
            })
            .collect::<Result<_>>()?;
        let sidecars: Vec<(&str, &str, Vec<u8>)> = sidecars
            .iter()
            .map(|(i, m, b)| (i.as_str(), m.as_str(), b.clone()))
            .collect();
        let mut module = rlx_ir::rlxmodel::Module::new(pkg.graphs.clone());
        module.dims = pkg.dims.clone();
        module.model = Some(model);
        super::rlxp::save_package_module(out, &module, &tensors, &sidecars)
    }

    /// Write the SAE trained as a package at `out`: its function `sae`
    /// (`h` `[rows, d]` → `features` `[rows, width]`, `reconstruction`
    /// `[rows, d]`, symbolic in the rows), its params, and the model
    /// block's `sae` (`base`, `tap`, `width`, `l1`).
    fn sae_package(
        &self,
        values: &HashMap<String, Vec<f32>>,
        out: &Path,
        name: &str,
    ) -> Result<()> {
        let sae = self
            .sae
            .as_ref()
            .context("only an SAE trainer saves an SAE")?;
        let (d, w) = (sae.d, sae.width);
        let at = |rows: usize| {
            let mut g = Graph::new("sae");
            let h = g.input("h", Shape::new(&[rows, d], DType::F32));
            let (z, r) = sae_apply(&mut g, h, d, w);
            g.node_mut(z).name = Some("features".into());
            g.node_mut(r).name = Some("reconstruction".into());
            g.set_outputs(vec![z, r]);
            g
        };
        const ROWS: u32 = 0;
        let g = rlx_ir::dynamic::generalize(&at(5), &at(7), &[(ROWS, 5, 7)])
            .map_err(anyhow::Error::msg)?;
        let dims = [vec![d, w], vec![1, w], vec![w, d], vec![1, d]];
        let tensors: Vec<Tensor> = SAE_PARAMS
            .iter()
            .zip(&dims)
            .map(|(p, s)| Tensor::f32(*p, s.clone(), &values[*p]))
            .collect();
        let base = std::fs::canonicalize(&self.base)
            .unwrap_or_else(|_| self.base.clone())
            .display()
            .to_string();
        let mut module = rlx_ir::rlxmodel::Module::new(vec![g]);
        module.dims = vec![("rows".into(), ROWS)];
        module.model = Some(serde_json::json!({
            "name": name,
            "task": "sparse-autoencoder",
            "sae": {"base": base, "tap": sae.tap, "width": w, "l1": sae.l1},
            "entries": {"pipeline": {
                "inputs": ["h"],
                "outputs": ["features", "reconstruction"],
                "stages": [{"function": "sae", "inputs": {"h": "h"},
                            "outputs": ["features", "reconstruction"]}],
            }},
        }));
        super::rlxp::save_package_module(out, &module, &tensors, &[])
    }

    /// Write the LoRA factors as an adapter package at `out` over the
    /// package trained (opened, it is that package with them merged).
    fn adapter(&self, values: &HashMap<String, Vec<f32>>, out: &Path, name: &str) -> Result<()> {
        if self.lora.is_empty() {
            bail!("only a LoRA trainer saves an adapter");
        }
        let r = self.rank;
        let mut updates = Vec::new();
        for l in &self.lora {
            let d = &values[&l.down];
            let u = &values[&l.up];
            // The matmul's update is D·U `[k, n]`; the param's, D·U or (D·U)ᵀ
            // = Uᵀ·Dᵀ: as L·R with L `[rows, r]`, R `[r, cols]`.
            let (lf, rf) = if l.transposed {
                (transpose(u, r, l.n), transpose(d, l.k, r))
            } else {
                (d.clone(), u.clone())
            };
            let (rows, cols) = if l.transposed { (l.n, l.k) } else { (l.k, l.n) };
            let (tensor, stored_t) = self.layout[&l.param].clone();
            // The adapter's delta is B·A in the tensor's stored layout.
            let (b, a) = if stored_t {
                (transpose(&rf, r, cols), transpose(&lf, rows, r))
            } else {
                (lf, rf)
            };
            updates.push(LoraUpdate {
                tensor,
                rank: r,
                a,
                b,
                scale: self.scale,
            });
        }
        let base = std::fs::canonicalize(&self.base)
            .unwrap_or_else(|_| self.base.clone())
            .display()
            .to_string();
        save_adapter(out, name, &base, &updates)
    }
}
