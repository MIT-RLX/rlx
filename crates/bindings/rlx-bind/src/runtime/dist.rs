// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distributed nodes: a [`Group`] is this process's place in a mesh of rlx
//! nodes — point-to-point messages, collectives, and the ship-graph protocol
//! of `rlx_runtime::dist` (a coordinator ships a stage or a training job, and
//! generic workers run it on their own hardware, resolving weights locally).
//!
//! Every call blocks until its peers take part: a binding runs each rank on its
//! own thread (or process, or machine). Specs and reports cross as JSON.

use std::sync::Arc;

use rlx_driver::ReduceKind;
use rlx_ir::Graph;
use rlx_runtime::dist::node::{self, NodeConfig, ProcessGroup, Topology};
use rlx_runtime::dist::{self as rdist, DataRef, StageSpec, TrainMetrics, TrainSpec, WeightRef};
use serde_json::{Value, json};

use super::opts::{Opts, type_name};
use crate::panics::catch;
use crate::{ApiError, ApiResult};

/// User message tags start here, clear of the ship-graph protocol's tags and
/// of the collectives' reserved range.
const USER_TAG_BASE: u32 = 1 << 16;

fn runtime(e: impl std::fmt::Display) -> ApiError {
    ApiError::runtime(e.to_string())
}

/// `world` free loopback addresses, for running every rank on one machine.
pub fn loopback_peers(world: u32) -> ApiResult<Vec<String>> {
    use std::net::{Ipv4Addr, TcpListener};
    // Hold every listener until all ports are chosen, so none repeats.
    let listeners = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(runtime))
        .collect::<ApiResult<Vec<_>>>()?;
    listeners
        .iter()
        .map(|l| {
            l.local_addr()
                .map(|a| format!("127.0.0.1:{}", a.port()))
                .map_err(runtime)
        })
        .collect()
}

fn strings(v: &Value, what: &str) -> ApiResult<Vec<String>> {
    match v {
        Value::Array(items) => items
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| ApiError::type_(format!("{what}: expected strings")))
            })
            .collect(),
        Value::String(s) => Ok(s.split(',').map(|p| p.trim().to_string()).collect()),
        other => Err(ApiError::type_(format!(
            "{what}: expected a list of strings, got {}",
            type_name(other)
        ))),
    }
}

fn reduce_kind(op: &str) -> ApiResult<ReduceKind> {
    match op.trim().to_ascii_lowercase().as_str() {
        "sum" => Ok(ReduceKind::Sum),
        "mean" | "avg" => Ok(ReduceKind::Mean),
        "max" => Ok(ReduceKind::Max),
        "min" => Ok(ReduceKind::Min),
        other => Err(ApiError::type_(format!(
            "unknown reduce op '{other}' (known: sum, mean, max, min)"
        ))),
    }
}

/// `name → uri` pairs from `{name: uri}` or `[{name, uri}]`.
fn weight_refs(v: Option<&Value>, packed: bool, what: &str) -> ApiResult<Vec<WeightRef>> {
    let Some(v) = v else {
        return Ok(Vec::new());
    };
    let item = |name: &str, uri: &Value| -> ApiResult<WeightRef> {
        let uri = uri
            .as_str()
            .ok_or_else(|| ApiError::type_(format!("{what} '{name}': the uri must be a string")))?;
        Ok(WeightRef {
            name: name.to_string(),
            uri: uri.to_string(),
            packed,
        })
    };
    match v {
        Value::Object(m) => m.iter().map(|(k, u)| item(k, u)).collect(),
        Value::Array(items) => items
            .iter()
            .map(|x| {
                let o = Opts::new(what, x)?;
                let name = o
                    .str("name")?
                    .ok_or_else(|| ApiError::type_(format!("{what}: each entry needs a name")))?;
                let uri = o
                    .get("uri")
                    .ok_or_else(|| ApiError::type_(format!("{what} '{name}': needs a uri")))?;
                let packed = o.bool("packed", packed)?;
                let mut w = item(name, uri)?;
                w.packed = packed;
                o.finish()?;
                Ok(w)
            })
            .collect(),
        other => Err(ApiError::type_(format!(
            "{what}: expected {{name: uri}} or [{{name, uri}}], got {}",
            type_name(other)
        ))),
    }
}

fn data_refs(v: Option<&Value>) -> ApiResult<Vec<DataRef>> {
    let Some(Value::Array(items)) = v else {
        return match v {
            None => Ok(Vec::new()),
            Some(other) => Err(ApiError::type_(format!(
                "train spec data: expected a list, got {}",
                type_name(other)
            ))),
        };
    };
    items
        .iter()
        .map(|x| {
            let o = Opts::new("train spec data", x)?;
            let need = |name: &str| {
                o.u64(name)?.map(|n| n as usize).ok_or_else(|| {
                    ApiError::type_(format!("train spec data: each entry needs '{name}'"))
                })
            };
            let input = o
                .str("input")?
                .ok_or_else(|| ApiError::type_("train spec data: each entry needs 'input'"))?
                .to_string();
            let uri = o
                .str("uri")?
                .ok_or_else(|| ApiError::type_("train spec data: each entry needs 'uri'"))?
                .to_string();
            let d = DataRef {
                input,
                uri,
                elem: need("elem")?,
                shard_start: need("shard_start")?,
                shard_len: need("shard_len")?,
            };
            o.finish()?;
            Ok(d)
        })
        .collect()
}

/// A [`TrainSpec`] for backward `graph` from its JSON form.
pub fn train_spec(graph: &Graph, spec: &Value) -> ApiResult<TrainSpec> {
    let o = Opts::new("train spec", spec)?;
    let params = weight_refs(o.get("params"), false, "train spec params")?;
    let data = data_refs(o.get("data"))?;
    let lr_per_epoch = match o.get("lr_per_epoch") {
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| ApiError::type_("train spec lr_per_epoch: expected numbers"))
            })
            .collect::<ApiResult<Vec<f32>>>()?,
        Some(other) => {
            return Err(ApiError::type_(format!(
                "train spec lr_per_epoch: expected a list of numbers, got {}",
                type_name(other)
            )));
        }
        None => {
            let lr = o.f32("lr", 0.01)?;
            vec![lr; o.usize("epochs", 1)?]
        }
    };
    let batch =
        o.u64("batch")?
            .ok_or_else(|| ApiError::type_("train spec: 'batch' is required"))? as usize;
    let s = TrainSpec {
        graph: graph.clone(),
        params,
        grad_start: o.usize("grad_start", 1)?,
        loss_index: o.usize("loss_index", 0)?,
        data,
        seed_input: Some(o.str("seed_input")?.unwrap_or("d_output").to_string()),
        momentum: o.f32("momentum", 0.0)?,
        lr_per_epoch,
        batch,
        device: o.str("device")?.unwrap_or("auto").to_string(),
        grad_group: o.u64("grad_group")?.unwrap_or(0),
        push_data: o.bool("push_data", false)?,
    };
    o.finish()?;
    Ok(s)
}

fn metrics_json(m: &TrainMetrics) -> Value {
    json!({
        "device": rlx_runtime::device_label(m.device),
        "lanes": m.lanes.iter().map(|d| rlx_runtime::device_label(*d)).collect::<Vec<_>>(),
        "samples": m.samples,
        "wall_s": m.wall_s,
        "compute_s": m.compute_s,
        "comm_s": m.comm_s,
        "first_loss": m.first_loss,
        "last_loss": m.last_loss,
    })
}

fn params_json(params: &[(String, Vec<f32>)]) -> Value {
    Value::Object(params.iter().map(|(n, v)| (n.clone(), json!(v))).collect())
}

/// This process's membership in a mesh.
pub struct Group {
    pub inner: Arc<ProcessGroup>,
}

impl Group {
    /// Join as `rank` of `world`. Options: `peers` (one `host:port` per rank,
    /// or the coordinator's alone for a star), `topology` (`"mesh"` |
    /// `"star"`), `device` (`"auto"`), `discover` (`{port, data_port, via}`,
    /// UDP rendezvous instead of `peers`), `heap_mb`.
    pub fn connect(rank: u32, world: u32, opts: &Value) -> ApiResult<Self> {
        if world == 0 || rank >= world {
            return Err(ApiError::range(format!(
                "rank {rank} is not in a world of {world}"
            )));
        }
        let o = Opts::new("node", opts)?;
        let mut cfg = NodeConfig::new(rank, world);
        if let Some(d) = o.str("device")? {
            cfg = cfg.device(d);
        }
        match o.str("topology")?.map(str::to_ascii_lowercase).as_deref() {
            None | Some("mesh") => cfg = cfg.topology(Topology::Mesh),
            Some("star") => cfg = cfg.topology(Topology::Star),
            Some(other) => {
                return Err(ApiError::type_(format!(
                    "node topology must be 'mesh' or 'star', got '{other}'"
                )));
            }
        }
        if let Some(mb) = o.u64("heap_mb")? {
            cfg = cfg.heap_bytes((mb as usize) << 20);
        }
        let peers = o.get("peers");
        let discover = o.get("discover");
        match (peers, discover) {
            (Some(_), Some(_)) => {
                return Err(ApiError::type_(
                    "node options: give 'peers' or 'discover', not both",
                ));
            }
            (Some(p), None) => {
                cfg = cfg
                    .peers(strings(p, "node peers")?)
                    .map_err(ApiError::type_)?;
            }
            (None, Some(d)) => {
                let d = Opts::new("node discover", d)?;
                let port = d.u64("port")?.unwrap_or(29600) as u16;
                let data_port = d.u64("data_port")?.unwrap_or(29500) as u16;
                cfg = cfg.discover(port, data_port);
                if let Some(via) = d.str("via")? {
                    cfg = cfg.discover_via(via);
                }
                d.finish()?;
            }
            (None, None) if world > 1 => {
                return Err(ApiError::type_(
                    "node options: a world of more than one needs 'peers' or 'discover'",
                ));
            }
            (None, None) => {
                cfg = cfg.peers(loopback_peers(1)?).map_err(ApiError::type_)?;
            }
        }
        o.finish()?;
        let inner = catch("connect", || cfg.connect())?.map_err(runtime)?;
        Ok(Self { inner })
    }

    /// Join with `RANK` / `WORLD` / `PEERS` / `TOPOLOGY` / `DEVICE` /
    /// `DISCOVER` from the environment.
    pub fn from_env() -> ApiResult<Self> {
        let cfg = NodeConfig::from_env().map_err(ApiError::type_)?;
        let inner = catch("connect", || cfg.connect())?.map_err(runtime)?;
        Ok(Self { inner })
    }

    pub fn rank(&self) -> u32 {
        self.inner.rank()
    }

    pub fn world(&self) -> u32 {
        self.inner.world_size()
    }

    pub fn is_leader(&self) -> bool {
        self.inner.is_leader()
    }

    pub fn barrier(&self) -> ApiResult<()> {
        self.inner.barrier().map_err(runtime)
    }

    fn tag(tag: u32) -> ApiResult<u32> {
        tag.checked_add(USER_TAG_BASE)
            .filter(|t| *t < rlx_driver::transport::TAG_RESERVED_BASE)
            .ok_or_else(|| ApiError::range(format!("message tag {tag} is too large")))
    }

    pub fn send(&self, to: u32, tag: u32, data: &[f32]) -> ApiResult<()> {
        self.inner
            .send_f32(to, Self::tag(tag)?, data)
            .map_err(runtime)
    }

    pub fn recv(&self, from: u32, tag: u32) -> ApiResult<Vec<f32>> {
        self.inner.recv_f32(from, Self::tag(tag)?).map_err(runtime)
    }

    pub fn all_reduce(&self, data: &[f32], op: &str) -> ApiResult<Vec<f32>> {
        let op = reduce_kind(op)?;
        let mut buf = data.to_vec();
        self.inner.all_reduce(&mut buf, op).map_err(runtime)?;
        Ok(buf)
    }

    pub fn all_gather(&self, data: &[f32]) -> ApiResult<Vec<f32>> {
        self.inner.all_gather(data).map_err(runtime)
    }

    pub fn broadcast(&self, root: u32, data: &[f32]) -> ApiResult<Vec<f32>> {
        let mut buf = data.to_vec();
        self.inner.broadcast(root, &mut buf).map_err(runtime)?;
        Ok(buf)
    }

    /// Coordinator: the capabilities every worker announced.
    pub fn collect_caps(&self) -> ApiResult<Value> {
        let caps = node::collect_caps(&self.inner).map_err(runtime)?;
        serde_json::to_value(caps).map_err(runtime)
    }

    /// Coordinator: ship a stage of `graph` to worker `rank`. Options:
    /// `input`, `output` (node names; default the graph's first input and
    /// `"output"`), `device` (`"auto"`), `weights` / `packed` (`{param: uri}`),
    /// `stage_id`.
    pub fn ship_stage(&self, rank: u32, graph: &Graph, opts: &Value) -> ApiResult<()> {
        let o = Opts::new("stage", opts)?;
        let input = match o.str("input")? {
            Some(s) => s.to_string(),
            None => graph
                .nodes()
                .iter()
                .find_map(|n| match &n.op {
                    rlx_ir::Op::Input { name } => Some(name.clone()),
                    _ => None,
                })
                .ok_or_else(|| ApiError::type_("stage: the graph has no input; pass 'input'"))?,
        };
        let output = o.str("output")?.unwrap_or("output").to_string();
        let device = o.str("device")?.unwrap_or("auto").to_string();
        let mut spec = StageSpec::new(graph.clone(), input, output, device);
        spec.weights = weight_refs(o.get("weights"), false, "stage weights")?;
        spec.weights
            .extend(weight_refs(o.get("packed"), true, "stage packed")?);
        if let Some(id) = o.str("stage_id")? {
            spec = spec.stage_id(id);
        }
        o.finish()?;
        rdist::ship_stage(&self.inner, rank, &spec).map_err(runtime)
    }

    pub fn send_activation(&self, to: u32, data: &[f32]) -> ApiResult<()> {
        rdist::send_activation(&self.inner, to, data).map_err(runtime)
    }

    pub fn recv_activation(&self, from: u32) -> ApiResult<Vec<f32>> {
        rdist::recv_activation(&self.inner, from).map_err(runtime)
    }

    /// Worker: announce, receive a stage, serve activations until the
    /// coordinator finishes (or `steps` of them).
    pub fn serve_worker(&self, opts: &Value) -> ApiResult<Value> {
        let o = Opts::new("worker", opts)?;
        let ctl = match o.u64("steps")? {
            Some(n) => node::NodeControl::steps(n as usize),
            None => node::NodeControl::unbounded(),
        };
        o.finish()?;
        let caps = node::NodeCaps {
            rank: self.rank(),
            role: node::NodeRole::Worker,
            devices: rlx_runtime::available_devices()
                .into_iter()
                .map(|d| d.as_arg().to_string())
                .collect(),
            platform: node::platform_tag().into(),
        };
        let r = catch("serve_worker", || {
            node::serve_worker_n(&self.inner, &caps, rdist::uri_resolver, &ctl)
        })?
        .map_err(runtime)?;
        Ok(json!({
            "rank": r.rank,
            "device": rlx_runtime::device_label(r.device),
            "platform": r.platform,
            "activations": r.activations,
        }))
    }

    /// Coordinator: ship a training job for backward `graph` to worker `rank`
    /// (and its data shard, when `push_data` is set).
    pub fn ship_train(&self, rank: u32, graph: &Graph, spec: &Value) -> ApiResult<()> {
        let spec = train_spec(graph, spec)?;
        rdist::ship_train(&self.inner, rank, &spec).map_err(runtime)?;
        if spec.push_data {
            rdist::push_shards(&self.inner, rank, &spec, rdist::uri_resolver).map_err(runtime)?;
        }
        Ok(())
    }

    /// Train this rank's shard in lockstep with the workers:
    /// `{metrics, params}`.
    pub fn run_train(&self, graph: &Graph, spec: &Value) -> ApiResult<Value> {
        let spec = train_spec(graph, spec)?;
        let world = self.world();
        let group = &self.inner;
        let (metrics, params) = catch("run_train", || {
            rdist::run_train(
                &spec,
                world,
                rdist::uri_resolver,
                |flat| node::mean_reduce(group, flat),
                false,
            )
        })?
        .map_err(runtime)?;
        Ok(json!({ "metrics": metrics_json(&metrics), "params": params_json(&params) }))
    }

    /// Worker: announce, receive a training job, train in lockstep.
    pub fn serve_trainer(&self, opts: &Value) -> ApiResult<Value> {
        let o = Opts::new("trainer", opts)?;
        let log = o.bool("log", false)?;
        o.finish()?;
        let r = catch("serve_trainer", || {
            node::serve_trainer_here(&self.inner, rdist::uri_resolver, log)
        })?
        .map_err(runtime)?;
        Ok(json!({
            "rank": r.rank,
            "platform": r.platform,
            "metrics": metrics_json(&r.metrics),
            "params": params_json(&r.params),
        }))
    }
}
