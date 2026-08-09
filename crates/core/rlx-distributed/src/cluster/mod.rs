// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Cluster orchestration** — the high-level DX over the pipeline primitives.
//! Turn a declarative [`ClusterConfig`] into a probed, HW-planned, monitored run:
//!
//! ```no_run
//! # use rlx_distributed::cluster::*;
//! # fn cost() -> ModelCost { unimplemented!() }
//! let mut cx = Cluster::from_path("cluster.toml")?;
//! cx.probe("/home/user/rlx-models/target/release/examples/dsv4_cluster")?; // ssh-probe every node
//! for a in cx.plan(cost())? { println!("{} -> layers {:?} on {}", a.addr, a.layers, a.device); }
//! // The model crate builds each stage; the coordinator drives + monitors:
//! let children = cx.launch("/home/user/.../dsv4_cluster")?;            // spawn workers
//! let run = cx.drive(vec![/* input NamedTensor */])?;                   // relay + time
//! println!("{}", run.table());
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! The model-specific bits (build a `Stage` for a layer range, load its weights)
//! stay in the model crate behind [`crate::Stage`] / [`crate::ParamSource`]; this
//! module only orchestrates.

pub mod caps;
pub mod config;
pub mod placement;
pub mod stats;

pub use caps::{DeviceInfo, NodeCaps, probe_local, probe_remote};
pub use config::{ClusterConfig, KvPolicy, NodeConfig, PlacementPolicy};
pub use placement::{Assignment, ModelCost, plan_placement};
pub use rlx_driver::{ProcessGroup, ReduceKind};
pub use stats::{ClusterRun, NodeReport, StageTiming};

use crate::graph::transport::run_pipeline_tcp_timed;
use crate::graph::{NamedTensor, Stage};
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Child;

/// Cross-node **all-reduce** over the cluster's [`ProcessGroup`] — the collective
/// complementary to the pipeline's sequential hidden-state relay. Model crates use
/// it for TENSOR parallelism (shard one layer across nodes, then `Sum` the partial
/// matmul outputs) or to aggregate per-node values (`Mean`/`Max`/`Min`). The
/// ProcessGroup rides the same rlx-driver transports (TCP / Thunderbolt / MLX).
pub fn all_reduce(pg: &ProcessGroup, buf: &mut [f32], kind: ReduceKind) -> Result<()> {
    pg.all_reduce(buf, kind)
        .map_err(|e| anyhow::anyhow!("all_reduce: {e:?}"))
}

/// A configured, probed, planned cluster.
pub struct Cluster {
    pub cfg: ClusterConfig,
    /// Probed hardware, one per node (same order as `cfg.nodes`). Empty until [`Cluster::probe`].
    pub caps: Vec<NodeCaps>,
    /// Layer→node plan. Empty until [`Cluster::plan`].
    pub plan: Vec<Assignment>,
}

impl Cluster {
    pub fn from_config(cfg: ClusterConfig) -> Self {
        Self {
            cfg,
            caps: Vec::new(),
            plan: Vec::new(),
        }
    }
    pub fn from_path(p: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::from_config(ClusterConfig::from_path(p)?))
    }

    /// Probe every node's hardware. `remote_bin` is the worker binary path on the
    /// remote hosts (its `--probe` mode self-reports); nodes without `ssh` are
    /// probed locally.
    pub fn probe(&mut self, remote_bin: &str) -> Result<&[NodeCaps]> {
        self.caps.clear();
        for n in &self.cfg.nodes {
            let caps = match &n.ssh {
                Some(host) => probe_remote(host, remote_bin, &n.addr, &n.ckpt_dir)
                    .with_context(|| format!("probe {host}"))?,
                None => probe_local(&n.addr, &n.ckpt_dir, true),
            };
            self.caps.push(caps);
        }
        Ok(&self.caps)
    }

    /// Plan placement from the model cost model + probed caps (probe first).
    pub fn plan(&mut self, model: ModelCost) -> Result<&[Assignment]> {
        anyhow::ensure!(!self.caps.is_empty(), "probe() before plan()");
        let nodes: Vec<_> = self
            .caps
            .iter()
            .cloned()
            .zip(self.cfg.nodes.iter().cloned())
            .collect();
        let reserve = (self.cfg.reserve_ram_gb * 1e9) as u64;
        self.plan = plan_placement(&model, &nodes, self.cfg.placement.policy, reserve)?;
        Ok(&self.plan)
    }

    /// The worker command-line for node `i` from its plan + config — so the model
    /// crate's worker binary honours device / precision / rng / kv / layers.
    pub fn worker_argv(&self, i: usize) -> Vec<String> {
        let a = &self.plan[i];
        let n = &self.cfg.nodes[i];
        let mut v = vec![
            "--role".into(),
            "worker".into(),
            "--index".into(),
            i.to_string(),
            "--layers".into(),
            format!("{}:{}", a.layers.start, a.layers.end),
            "--ckpt".into(),
            n.ckpt_dir.clone(),
            "--addr".into(),
            n.addr.clone(),
            "--seq".into(),
            self.cfg.seq.to_string(),
            "--device".into(),
            n.device.clone(),
            "--precision".into(),
            n.precision.clone(),
            "--kv".into(),
            format!("{:?}", n.kv_cache).to_lowercase(),
            "--rng".into(),
            self.cfg.seed_for(i).to_string(),
        ];
        if a.first {
            v.push("--first".into());
        }
        if a.last {
            v.push("--last".into());
        }
        v
    }

    /// Spawn a worker process per node (ssh for remote, local otherwise). The
    /// caller drives them via [`Cluster::drive`], then reaps the children.
    pub fn launch(&self, remote_bin: &str) -> Result<Vec<Child>> {
        anyhow::ensure!(!self.plan.is_empty(), "plan() before launch()");
        use std::process::Stdio;
        let mut kids = Vec::new();
        for (i, n) in self.cfg.nodes.iter().enumerate() {
            let argv = self.worker_argv(i);
            // Pipe stdout so the coordinator can await "serving on" + collect the
            // node's build report; leave stderr inherited for live error output.
            let child = match &n.ssh {
                Some(host) => std::process::Command::new("ssh")
                    .arg(host)
                    .arg(format!("{remote_bin} {}", argv.join(" ")))
                    .stdout(Stdio::piped())
                    .spawn()?,
                // Local node: this very binary, not the remote path. Its stderr is
                // otherwise a black hole (the coordinator's own stderr races it);
                // capture to a file when RLX_WORKER_ERR_DIR is set, for diagnosis.
                None => {
                    let mut c = std::process::Command::new(std::env::current_exe()?);
                    c.args(&argv).stdout(Stdio::piped());
                    if let Ok(dir) = std::env::var("RLX_WORKER_ERR_DIR")
                        && let Ok(f) = std::fs::File::create(format!("{dir}/local_worker_{i}.err"))
                    {
                        c.stderr(Stdio::from(f));
                    }
                    c.spawn()?
                }
            };
            kids.push(child);
        }
        Ok(kids)
    }

    /// Drive one forward through the planned stages, timing each, and collect the
    /// output. `inputs` seeds the first stage (e.g. `input_ids`). Boundary tensor
    /// name is `hidden`/`hidden_in` per the model's stage I/O convention.
    pub fn drive(&self, inputs: Vec<NamedTensor>) -> Result<ClusterRun> {
        let stages: Vec<Stage> = (0..self.plan.len()).map(|i| self.meta_stage(i)).collect();
        let addrs: Vec<String> = self.cfg.nodes.iter().map(|n| n.addr.clone()).collect();
        let (out, per_stage_ms) = run_pipeline_tcp_timed(&stages, &addrs, inputs)?;
        let timings = self
            .plan
            .iter()
            .zip(per_stage_ms)
            .map(|(a, ms)| StageTiming {
                addr: a.addr.clone(),
                layers: a.layers.clone(),
                device: a.device.clone(),
                build_ms: 0,
                forward_ms: ms,
                resident_bytes: a.est_bytes,
            })
            .collect::<Vec<_>>();
        let total: u64 = timings.iter().map(|t| t.forward_ms).sum();
        Ok(ClusterRun {
            timings,
            total_forward_ms: total,
            output: out.into_iter().next().map(|t| t.data).unwrap_or_default(),
        })
    }

    /// Coordinator-side boundary metadata for stage `i` (no weights, no graph):
    /// first stage takes `input_ids`, others take `hidden_in`; last emits
    /// `logits`, others `hidden_in`.
    fn meta_stage(&self, i: usize) -> Stage {
        let (first, last) = (self.plan[i].first, self.plan[i].last);
        Stage {
            index: i,
            graph: rlx_ir::Graph::new("meta"),
            inputs: vec![if first { "input_ids" } else { "hidden_in" }.into()],
            outputs: vec![if last { "logits" } else { "hidden_in" }.into()],
            output_shapes: vec![],
            params: vec![],
        }
    }
}
