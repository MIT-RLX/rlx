// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Cluster orchestration** — the high-level DX over the pipeline primitives.
//! Turn a declarative [`ClusterConfig`] into a probed, HW-planned, monitored run:
//!
//! ```no_run
//! # use rlx_distributed::cluster::*;
//! # fn tensor_index() -> Vec<TensorEntry> { unimplemented!() }
//! let bin = "/home/user/rlx-models/target/release/examples/dsv4_cluster";
//! let mut cx = Cluster::from_path("cluster.toml")?;
//!
//! // Discover machines, probe them, derive the cost from the checkpoint's
//! // tensor index, and plan — one call.
//! cx.autoplan(&tensor_index(), &CostOptions::default().with_experts(8, 288), bin)?;
//! print!("{}", cx.plan_table());
//!
//! // The model crate builds each stage; the coordinator drives + monitors:
//! let children = cx.launch(bin)?;                    // spawn workers
//! let run = cx.drive(vec![/* input NamedTensor */])?; // relay + time
//! println!("{}", run.table());
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! The three steps are still available separately ([`Cluster::discover`],
//! [`Cluster::probe`], [`Cluster::plan`]) when you want to inspect or override
//! something in between.
//!
//! The model-specific bits (build a `Stage` for a layer range, load its weights)
//! stay in the model crate behind [`crate::Stage`] / [`crate::ParamSource`]; this
//! module only orchestrates.

pub mod caps;
pub mod config;
pub mod cost;
pub mod discover;
pub mod placement;
pub mod stats;

pub use caps::{DeviceInfo, NodeCaps, probe_local, probe_remote};
pub use config::{
    ClusterConfig, DeviceList, KvPolicy, NodeConfig, NodeRole, Objective, Optimize,
    PlacementPolicy, PlacementSection, Precision, UnknownKv,
};
pub use cost::{
    BlockKv, CostOptions, KvDetail, KvMarkers, KvProfile, KvSource, Markers, ModelCost, TensorEntry,
};
pub use discover::{DiscoverSection, Discovered, discover_ssh, discover_subnet};
pub use placement::{Assignment, assign_standbys, plan_placement, plan_placement_with, resolve_kv};
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
    /// Layer→node plan. Empty until [`Cluster::plan`]. Standby stages follow the
    /// active ones and mirror their layer ranges.
    pub plan: Vec<Assignment>,
    /// Set when the precision ladder had to step down from the stored width.
    pub chosen_precision: Option<String>,
    /// KV profile the last plan resolved to.
    last_kv: Option<KvProfile>,
}

impl Cluster {
    pub fn from_config(cfg: ClusterConfig) -> Self {
        Self {
            cfg,
            caps: Vec::new(),
            plan: Vec::new(),
            chosen_precision: None,
            last_kv: None,
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

    /// Find machines from the `[discover]` section and append them to
    /// `cfg.nodes`. Returns `(added, failures)`; a host that will not probe is
    /// reported rather than fatal, so one dead machine does not stop the rest.
    ///
    /// Idempotent: nodes already present (by `addr`) are left alone, so
    /// hand-pinned entries survive and re-running adds nothing.
    pub fn discover(&mut self, remote_bin: &str) -> Result<(usize, Vec<(String, String)>)> {
        let d = self.cfg.discover.clone();
        if d.is_empty() {
            return Ok((0, Vec::new()));
        }
        let mut added = 0;
        let mut failed = Vec::new();
        if !d.ssh_hosts.is_empty() {
            let (found, f) = discover::discover_ssh(&d, remote_bin)?;
            added += discover::merge_nodes(&mut self.cfg.nodes, found);
            failed.extend(f);
        }
        if d.subnet.is_some() {
            let found = discover::discover_subnet(&d)?;
            added += discover::merge_nodes(&mut self.cfg.nodes, found);
        }
        if self.cfg.nodes.is_empty() {
            anyhow::bail!(
                "discovery found no usable nodes{}",
                if failed.is_empty() {
                    String::new()
                } else {
                    format!(
                        " ({} host(s) failed to probe: {})",
                        failed.len(),
                        failed
                            .iter()
                            .map(|(h, e)| format!("{h}: {e}"))
                            .collect::<Vec<_>>()
                            .join("; ")
                    )
                }
            );
        }
        Ok((added, failed))
    }

    /// **Discover → probe → cost → plan, in one call.**
    ///
    /// The DX this module exists for: point it at a checkpoint's tensor index
    /// and get an assignment back, without hand-writing node blocks or working
    /// out per-layer bytes. `remote_bin` is the worker binary on the remote
    /// hosts (its `--probe` mode self-reports).
    ///
    /// Discovery is skipped when `[discover]` is absent, so an all-manual config
    /// still works and this is just `probe` + `plan` with the cost derived for
    /// you.
    pub fn autoplan(
        &mut self,
        index: &[TensorEntry],
        opts: &CostOptions,
        remote_bin: &str,
    ) -> Result<&[Assignment]> {
        if !self.cfg.discover.is_empty() {
            let (added, failed) = self.discover(remote_bin)?;
            for (h, e) in &failed {
                eprintln!("discover: skipping {h}: {e}");
            }
            if added > 0 {
                eprintln!("discover: added {added} node(s)");
            }
        }
        self.probe(remote_bin)?;
        let cost = ModelCost::from_tensor_index(index, opts)?;
        eprintln!("model: {}", cost.summary());
        self.plan(cost)
    }

    /// **Retire a node**, moving its work off without a restart.
    ///
    /// Marks it `Draining` and, if a standby is mirroring it, promotes that
    /// standby in place — it already holds the same layers, so this is a
    /// pointer swap. With no standby the remaining active nodes are re-planned,
    /// which works but reloads weights, so a cluster you intend to drain should
    /// carry spares.
    ///
    /// Re-plans from the caps already probed; call [`Cluster::probe`] first if
    /// the hardware may have changed.
    pub fn retire(&mut self, addr: &str, model: ModelCost) -> Result<&[Assignment]> {
        use config::NodeRole;
        let i = self
            .cfg
            .nodes
            .iter()
            .position(|n| n.addr == addr)
            .with_context(|| format!("no node {addr} in this cluster"))?;
        anyhow::ensure!(
            self.cfg.nodes[i].role != NodeRole::Draining,
            "node {addr} is already draining"
        );
        self.cfg.nodes[i].role = NodeRole::Draining;

        // Promote a standby that was already mirroring it, if there is one.
        let promoted = self
            .cfg
            .nodes
            .iter_mut()
            .find(|n| n.role == NodeRole::Standby && n.standby_for.as_deref() == Some(addr))
            .map(|n| {
                n.role = NodeRole::Active;
                n.standby_for = None;
                n.addr.clone()
            });
        if let Some(p) = &promoted {
            eprintln!("retire {addr}: promoted standby {p} (weights already loaded)");
        } else {
            eprintln!("retire {addr}: no standby — re-planning across the remaining nodes");
        }
        self.plan(model)
    }

    /// Active nodes remaining after any draining/standby ones are excluded.
    pub fn active_nodes(&self) -> Vec<&NodeConfig> {
        self.cfg
            .nodes
            .iter()
            .filter(|n| n.role == config::NodeRole::Active)
            .collect()
    }

    /// The KV profile the last plan used, after the objective's override and
    /// unknown-handling are applied.
    pub fn kv_profile(&self) -> KvProfile {
        self.last_kv.clone().unwrap_or_default()
    }

    /// The plan as a table — what each machine ends up holding.
    ///
    /// Shows the disk column and the predicted per-stage time, because for an
    /// MoE those are what the assignment actually turned on; a plan that looks
    /// RAM-balanced can still be badly skewed by one node's slow disk.
    pub fn plan_table(&self) -> String {
        let mut out =
            String::from("node                 device    layers      resident      disk   stage\n");
        for a in &self.plan {
            out.push_str(&format!(
                "{:<20} {:<9} {:>3}..{:<3} {:>7.1}/{:<5.1}G {:>7.1}G {:>7.2}G {:>6.3}s{}\n",
                a.addr,
                a.device,
                a.layers.start,
                a.layers.end,
                a.est_bytes as f64 / 1e9,
                a.budget_bytes as f64 / 1e9,
                a.est_disk_bytes as f64 / 1e9,
                a.est_kv_bytes as f64 / 1e9,
                a.est_stage_secs,
                match (a.role, a.backs.as_deref()) {
                    (config::NodeRole::Standby, Some(b)) => format!(" standby<-{b}"),
                    (config::NodeRole::Standby, None) => " standby".into(),
                    (config::NodeRole::Draining, _) => " draining".into(),
                    _ if a.experts_paged => " paged".into(),
                    _ => String::new(),
                },
            ));
        }
        let active_layers: u64 = self
            .plan
            .iter()
            .filter(|a| a.role == config::NodeRole::Active)
            .map(|a| a.layers.len() as u64)
            .sum();
        if active_layers > 0 {
            out.push_str(&format!(
                "{}\n",
                self.kv_profile()
                    .summary(active_layers, self.cfg.placement.objective.context)
            ));
        }
        if let Some(slowest) = self
            .plan
            .iter()
            .filter(|a| a.role == config::NodeRole::Active)
            .max_by(|x, y| x.est_stage_secs.total_cmp(&y.est_stage_secs))
        {
            out.push_str(&format!(
                "critical path: {} at {:.3}s/token\n",
                slowest.addr, slowest.est_stage_secs
            ));
        }
        out
    }

    /// Plan placement from the model cost model + probed caps (probe first).
    pub fn plan(&mut self, model: ModelCost) -> Result<&[Assignment]> {
        anyhow::ensure!(!self.caps.is_empty(), "probe() before plan()");
        anyhow::ensure!(
            self.caps.len() == self.cfg.nodes.len(),
            "probed {} node(s) but the config has {} — re-probe after discovery",
            self.caps.len(),
            self.cfg.nodes.len()
        );
        let all: Vec<(NodeCaps, NodeConfig)> = self
            .caps
            .iter()
            .cloned()
            .zip(self.cfg.nodes.iter().cloned())
            .collect();
        // Only Active nodes carry a stage; standbys mirror one afterwards and
        // draining nodes are on their way out.
        let active: Vec<(NodeCaps, NodeConfig)> = all
            .iter()
            .filter(|(_, c)| c.role == config::NodeRole::Active)
            .cloned()
            .collect();
        anyhow::ensure!(
            !active.is_empty(),
            "no active nodes: {} standby, {} draining",
            all.iter()
                .filter(|(_, c)| c.role == config::NodeRole::Standby)
                .count(),
            all.iter()
                .filter(|(_, c)| c.role == config::NodeRole::Draining)
                .count()
        );
        let reserve = (self.cfg.reserve_ram_gb * 1e9) as u64;
        let obj = self.cfg.placement.objective.clone();
        let policy = self.cfg.placement.policy;

        // Walk the precision ladder: try the model as stored, then each step
        // down, and report which one fit rather than just "too big".
        let mut attempts: Vec<(Option<String>, ModelCost)> = vec![(None, model.clone())];
        for name in &obj.precision_ladder {
            match name.bits().and_then(|b| model.at_bits(b)) {
                Some(m) => attempts.push((Some(name.to_string()), m)),
                None => eprintln!("plan: ignoring precision `{name}` (no fixed width)"),
            }
        }
        let mut last_err = None;
        for (name, m) in attempts {
            match placement::plan_placement_with(&m, &active, policy, reserve, &obj) {
                Ok(mut p) => {
                    if let Some(n) = &name {
                        eprintln!(
                            "plan: does not fit as stored; using precision `{n}` \
                             ({:.1} GB)",
                            m.total_bytes() as f64 / 1e9
                        );
                        self.chosen_precision = Some(n.clone());
                    }
                    let kv = placement::resolve_kv(&m, &obj)?;
                    p.extend(placement::assign_standbys(
                        &m,
                        &all,
                        &p,
                        reserve,
                        &kv,
                        obj.context,
                    ));
                    self.last_kv = Some(kv);
                    self.plan = p;
                    return Ok(&self.plan);
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("placement failed")))
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
            n.device.to_string(),
            "--precision".into(),
            n.precision.to_string(),
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
                    if let Some(dir) = rlx_ir::env::var("RLX_WORKER_ERR_DIR")
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

#[cfg(test)]
mod tests {
    use super::*;
    use config::NodeRole;

    fn caps_for(addr: &str, ram_gb: f64) -> NodeCaps {
        NodeCaps {
            addr: addr.into(),
            os: "linux".into(),
            cores: 8,
            ram_total: (ram_gb * 1e9) as u64,
            ram_avail: (ram_gb * 1e9) as u64,
            disk_free: 1_000_000_000_000,
            devices: vec![],
            gflops: 100.0,
            io_mbps: 2000.0,
        }
    }
    fn node_for(addr: &str, role: NodeRole, backs: Option<&str>) -> NodeConfig {
        NodeConfig {
            addr: addr.into(),
            ssh: None,
            ckpt_dir: "/m".into(),
            device: "cpu".parse().unwrap(),
            precision: "bf16".parse().unwrap(),
            kv_cache: Default::default(),
            rng_seed: None,
            max_ram_gb: None,
            layers: None,
            role,
            standby_for: backs.map(String::from),
        }
    }
    fn dense(n_layers: usize, per_layer: u64) -> ModelCost {
        ModelCost {
            n_layers,
            per_layer_bytes: per_layer,
            per_layer_expert_bytes: 0,
            per_layer_expert_active_bytes: 0,
            embed_bytes: 0,
            head_bytes: 0,
            per_layer_flops: 1.0,
            kv: cost::KvProfile::declared(0),
            params: 0,
            hidden_size: 0,
        }
    }
    fn cluster(nodes: Vec<NodeConfig>) -> Cluster {
        let caps = nodes.iter().map(|n| caps_for(&n.addr, 32.0)).collect();
        let cfg = ClusterConfig {
            model: "m".into(),
            seq: 8,
            rng_seed: None,
            reserve_ram_gb: 1.0,
            placement: Default::default(),
            discover: Default::default(),
            nodes,
        };
        let mut c = Cluster::from_config(cfg);
        c.caps = caps;
        c
    }

    /// Standbys are planned but do not carry traffic: they mirror an active
    /// stage and are excluded from the critical path.
    #[test]
    fn plan_covers_actives_and_mirrors_standbys() {
        let mut cx = cluster(vec![
            node_for("a", NodeRole::Active, None),
            node_for("b", NodeRole::Active, None),
            node_for("spare", NodeRole::Standby, Some("b")),
        ]);
        let plan = cx.plan(dense(8, 1_000_000_000)).unwrap().to_vec();
        let active: Vec<_> = plan.iter().filter(|a| a.role == NodeRole::Active).collect();
        let standby: Vec<_> = plan
            .iter()
            .filter(|a| a.role == NodeRole::Standby)
            .collect();
        assert_eq!(active.len(), 2);
        assert_eq!(standby.len(), 1);
        // Active stages tile 0..8 exactly; the standby duplicates one of them.
        assert_eq!(active[0].layers.start, 0);
        assert_eq!(active.last().unwrap().layers.end, 8);
        let b = active.iter().find(|a| a.addr == "b").unwrap();
        assert_eq!(standby[0].layers, b.layers);
    }

    /// Retiring a node with a standby promotes it in place — the layer range
    /// does not move, so no weights are reloaded.
    #[test]
    fn retire_promotes_the_standby_without_moving_layers() {
        let mut cx = cluster(vec![
            node_for("a", NodeRole::Active, None),
            node_for("b", NodeRole::Active, None),
            node_for("spare", NodeRole::Standby, Some("b")),
        ]);
        let before = cx.plan(dense(8, 1_000_000_000)).unwrap().to_vec();
        let b_layers = before
            .iter()
            .find(|a| a.addr == "b")
            .unwrap()
            .layers
            .clone();

        let after = cx.retire("b", dense(8, 1_000_000_000)).unwrap().to_vec();
        // `b` is gone from the active set, `spare` is active and holds its range.
        assert!(
            !after
                .iter()
                .any(|a| a.addr == "b" && a.role == NodeRole::Active)
        );
        let promoted = after.iter().find(|a| a.addr == "spare").unwrap();
        assert_eq!(promoted.role, NodeRole::Active);
        assert_eq!(promoted.layers, b_layers, "promotion must not move layers");
        // Still a complete cover.
        let mut act: Vec<_> = after
            .iter()
            .filter(|a| a.role == NodeRole::Active)
            .collect();
        act.sort_by_key(|a| a.layers.start);
        assert_eq!(act[0].layers.start, 0);
        assert_eq!(act.last().unwrap().layers.end, 8);
    }

    /// Without a standby, retiring still works — it just re-plans, which is the
    /// slow path (weights reload) and is why spares exist.
    #[test]
    fn retire_without_a_standby_replans_the_rest() {
        let mut cx = cluster(vec![
            node_for("a", NodeRole::Active, None),
            node_for("b", NodeRole::Active, None),
        ]);
        cx.plan(dense(8, 1_000_000_000)).unwrap();
        let after = cx.retire("a", dense(8, 1_000_000_000)).unwrap().to_vec();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].addr, "b");
        assert_eq!(after[0].layers, 0..8);
    }

    #[test]
    fn retiring_the_last_active_node_is_refused() {
        let mut cx = cluster(vec![node_for("a", NodeRole::Active, None)]);
        cx.plan(dense(4, 1_000_000_000)).unwrap();
        let err = cx
            .retire("a", dense(4, 1_000_000_000))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no active nodes"), "{err}");
    }

    #[test]
    fn retiring_an_unknown_node_is_an_error() {
        let mut cx = cluster(vec![node_for("a", NodeRole::Active, None)]);
        cx.caps = vec![caps_for("a", 32.0)];
        let err = cx.retire("nope", dense(4, 1)).unwrap_err().to_string();
        assert!(err.contains("no node nope"), "{err}");
    }

    /// The ladder steps down until the model fits, and records what it chose.
    #[test]
    fn precision_ladder_steps_down_until_it_fits() {
        let mut cx = cluster(vec![node_for("a", NodeRole::Active, None)]);
        cx.cfg.placement.objective.precision_ladder = vec!["mxfp4".parse().unwrap()];
        // 8 layers × 4 GB = 32 GB at bf16; the node has 32 GB with 1 GB reserved.
        let mut model = dense(8, 4_000_000_000);
        model.params = model.total_bytes() / 2; // 16 bits per weight
        let plan = cx.plan(model).unwrap().to_vec();
        assert_eq!(plan[0].layers, 0..8);
        assert_eq!(cx.chosen_precision.as_deref(), Some("mxfp4"));
    }
}
