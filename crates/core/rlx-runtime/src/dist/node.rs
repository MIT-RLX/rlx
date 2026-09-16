// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Platform-neutral **node driver** — the piece a non-desktop shell links.
//!
//! Everything a node needs to join a mesh and serve work, with **no
//! environment variables, no process spawning, and no stdout contract**. That
//! matters because the existing node driver lives in an *example*
//! (`rlx-collectives/examples/dist_node.rs`), and an example cannot be linked
//! into an Android `.so`, an iOS framework, or an embedded host shim.
//!
//! ```no_run
//! # use rlx_runtime::dist::node::{NodeConfig, serve_worker};
//! # fn demo() -> Result<(), String> {
//! let cfg = NodeConfig::new(1, 2)
//!     .peers(["192.168.1.10:29500", "192.168.1.11:29501"])?
//!     .device("auto");
//! let group = cfg.connect().map_err(|e| e.to_string())?;
//! let report = serve_worker(&group, |_uri| Vec::new())?;
//! # let _ = report; Ok(()) }
//! ```
//!
//! # Roles
//!
//! * [`NodeRole::Worker`] — the general case: compile and run whatever stage
//!   the coordinator ships. Any host with a CPU qualifies (phone, SBC, laptop).
//! * [`NodeRole::FixedFunction`] — a **pre-synthesized** accelerator whose
//!   datapath is baked into a bitstream (an FPGA). It cannot compile a shipped
//!   graph, so it advertises the one stage it implements and refuses anything
//!   else. See [`FixedFunction`].

use super::{StageSpec, WeightCache, recv_activation, resolve_device, send_activation};
use crate::Device;
use rlx_driver::{Node, ReduceKind};

/// Re-exported so a shell can name what [`NodeConfig::connect`] hands back
/// without adding a dependency on the transport crate. The node API is meant
/// to be the whole surface a non-Rust shell needs — `.star()` / `.mesh()`
/// exist for the same reason.
pub use rlx_driver::{ProcessGroup, Topology};
use rlx_ir::DType;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Wire tag for the capability handshake. Kept clear of the ship-graph tags
/// (`40`/`41` in [`super::inference`]) and the reserved collective range.
const TAG_CAPS: u32 = 42;

// ── fixed-function (pre-synthesized) ranks ────────────────────────────────

/// A stage baked into hardware that cannot be reprogrammed at run time.
///
/// An FPGA bitstream implements exactly one datapath. Synthesis takes minutes,
/// so the mesh cannot ship it a graph the way it ships one to a CPU/GPU worker
/// — which is why `rlx-fpga` is a code generator and deliberately does not
/// implement the runtime `Backend` trait. Instead the board joins as a rank
/// that advertises *this* descriptor, and the coordinator only ever assigns it
/// the matching stage.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct FixedFunction {
    /// Identifier of the synthesized stage. The coordinator sets the same id on
    /// the [`StageSpec`] it ships; a mismatch is refused before any work moves.
    pub stage_id: String,
    /// Element count of the activation this datapath consumes.
    pub input_elems: usize,
    /// Element count it produces.
    pub output_elems: usize,
    /// Element type the datapath was synthesized for (INT8 datapaths are common
    /// on FPGA; the host still exchanges f32 on the wire and the board shim
    /// does the conversion).
    pub dtype: DType,
}

impl FixedFunction {
    pub fn new(
        stage_id: impl Into<String>,
        input_elems: usize,
        output_elems: usize,
        dtype: DType,
    ) -> Self {
        Self {
            stage_id: stage_id.into(),
            input_elems,
            output_elems,
            dtype,
        }
    }

    /// Whether this hardware can run `spec`. Checked by the **coordinator**
    /// before shipping, so a mismatch is a clear error at placement time rather
    /// than a hang or a wrong number at execution time.
    pub fn accepts(&self, spec: &StageSpec) -> Result<(), String> {
        match spec.stage_id.as_deref() {
            Some(id) if id == self.stage_id => Ok(()),
            Some(id) => Err(format!(
                "fixed-function rank implements stage [{}], cannot run [{id}]: \
                 an FPGA datapath is synthesized, not compiled",
                self.stage_id
            )),
            None => Err(format!(
                "fixed-function rank implements stage [{}], but the shipped \
                 StageSpec carries no stage_id — set StageSpec::stage_id to \
                 target pre-synthesized hardware",
                self.stage_id
            )),
        }
    }
}

/// The host-side driver for a piece of fixed-function hardware.
///
/// Implemented per board/link (PCIe, AXI on a Zynq PS, USB, UART). Keeping it a
/// trait means the mesh side is testable with [`LoopbackFixedFunction`] and no
/// hardware attached.
pub trait FixedFunctionDevice: Send {
    /// The stage this board implements — must match the advertised descriptor.
    fn stage_id(&self) -> &str;
    /// Push one activation to the board and read the result back.
    fn execute(&mut self, input: &[f32]) -> Result<Vec<f32>, String>;
}

/// A software stand-in for a synthesized board, so the fixed-function mesh path
/// can be exercised in tests and on a dev machine with no FPGA attached.
pub struct LoopbackFixedFunction {
    stage_id: String,
    f: Box<dyn FnMut(&[f32]) -> Vec<f32> + Send>,
}

impl LoopbackFixedFunction {
    pub fn new(
        stage_id: impl Into<String>,
        f: impl FnMut(&[f32]) -> Vec<f32> + Send + 'static,
    ) -> Self {
        Self {
            stage_id: stage_id.into(),
            f: Box::new(f),
        }
    }
}

impl FixedFunctionDevice for LoopbackFixedFunction {
    fn stage_id(&self) -> &str {
        &self.stage_id
    }
    fn execute(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        Ok((self.f)(input))
    }
}

// ── node identity ─────────────────────────────────────────────────────────

/// What kind of work a node accepts.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub enum NodeRole {
    /// Compiles and runs any stage shipped to it.
    #[default]
    Worker,
    /// Runs exactly one pre-synthesized stage.
    FixedFunction(FixedFunction),
}

/// What a node tells the coordinator about itself, so placement is driven by
/// what the hardware can actually do rather than by assumption.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeCaps {
    pub rank: u32,
    pub role: NodeRole,
    /// Backends live on this node, as `Device::as_arg` tokens (the wire stays
    /// stable even if the `Device` enum gains variants).
    pub devices: Vec<String>,
    /// Free-form platform tag (`"ios"`, `"android"`, `"linux"`, …) for logs and
    /// placement heuristics.
    pub platform: String,
}

impl NodeCaps {
    /// Whether this node can run `spec` at all.
    pub fn accepts(&self, spec: &StageSpec) -> Result<(), String> {
        match &self.role {
            NodeRole::Worker => Ok(()),
            NodeRole::FixedFunction(ff) => ff.accepts(spec),
        }
    }

    /// Whether this node can join a **training** run.
    ///
    /// A coordinator must ask before shipping a
    /// [`TrainSpec`](super::TrainSpec). Shipping one to a rank that cannot
    /// train does not fail loudly: the spec lands on a tag that rank never
    /// reads, and then the coordinator blocks forever on the first gradient
    /// all-reduce, because a barrier needs every rank to arrive.
    pub fn can_train(&self) -> Result<(), String> {
        match &self.role {
            NodeRole::Worker => Ok(()),
            NodeRole::FixedFunction(ff) => Err(format!(
                "rank {} implements the pre-synthesized stage [{}] and cannot \
                 train: a bitstream has no backward pass, and a rank that never \
                 reduces stalls every rank that does",
                self.rank, ff.stage_id
            )),
        }
    }
}

/// Compile-time platform tag. Not a capability — just an identifier, so a
/// coordinator log says *which* kind of node answered.
pub const fn platform_tag() -> &'static str {
    if cfg!(target_os = "ios") {
        "ios"
    } else if cfg!(target_os = "android") {
        "android"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_arch = "wasm32") {
        "wasm"
    } else {
        "unknown"
    }
}

// ── peer configuration ────────────────────────────────────────────────────

/// How this node finds its peers.
#[derive(Clone, Debug)]
pub enum PeerSource {
    /// Explicit `host:port` list, index = rank.
    Static(Vec<String>),
    /// UDP rendezvous. `via` unicasts the query to a known host instead of
    /// broadcasting — the path that works from a NAT'd or broadcast-restricted
    /// network (mobile hotspots, Docker, QEMU).
    Discover {
        disc_port: u16,
        data_base: u16,
        via: Option<String>,
    },
}

impl Default for PeerSource {
    fn default() -> Self {
        Self::Static(Vec::new())
    }
}

// ── node configuration ────────────────────────────────────────────────────

/// Programmatic node configuration.
///
/// Every field is settable in code. [`NodeConfig::from_env`] exists for parity
/// with the CLI workflow, but nothing here *requires* an environment — an iOS
/// app or an Android service has no meaningful `RANK`/`PEERS` to read.
#[derive(Clone, Debug)]
pub struct NodeConfig {
    pub rank: u32,
    pub world: u32,
    pub peers: PeerSource,
    pub topology: Topology,
    /// Device directive: `auto` (node's fastest) or a name (`cpu`, `metal`, …).
    pub device: String,
    pub heap_bytes: Option<usize>,
    pub role: NodeRole,
}

impl NodeConfig {
    pub fn new(rank: u32, world: u32) -> Self {
        Self {
            rank,
            world,
            peers: PeerSource::default(),
            topology: Topology::Mesh,
            device: "auto".into(),
            heap_bytes: None,
            role: NodeRole::Worker,
        }
    }

    /// Static peer list.
    ///
    /// Accepts either form:
    /// * `world` addresses, indexed by rank — a full mesh, where every rank
    ///   must be able to reach every other.
    /// * exactly one address — the **coordinator**, for a star. This is what a
    ///   phone can actually supply: a handset does not know its own reachable
    ///   address, and on most Wi-Fi it cannot accept inbound connections at
    ///   all, so it dials out and nothing dials it.
    pub fn peers<I, S>(mut self, addrs: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let v: Vec<String> = addrs.into_iter().map(Into::into).collect();
        if v.len() != self.world as usize && v.len() != 1 {
            return Err(format!(
                "peers: got {} address(es); expected {} (one per rank, for a \
                 mesh) or 1 (the coordinator, for a star)",
                v.len(),
                self.world
            ));
        }
        self.peers = PeerSource::Static(v);
        Ok(self)
    }

    /// Zero-config UDP peer discovery on a shared LAN.
    pub fn discover(mut self, disc_port: u16, data_base: u16) -> Self {
        self.peers = PeerSource::Discover {
            disc_port,
            data_base,
            via: None,
        };
        self
    }

    /// Unicast the discovery query to a known rendezvous host. Use this on
    /// networks where UDP broadcast does not reach every peer — which includes
    /// most carrier and guest Wi-Fi, and iOS without the multicast entitlement.
    pub fn discover_via(mut self, host: impl Into<String>) -> Self {
        if let PeerSource::Discover { via, .. } = &mut self.peers {
            *via = Some(host.into());
        } else {
            self.peers = PeerSource::Discover {
                disc_port: 29600,
                data_base: 29500,
                via: Some(host.into()),
            };
        }
        self
    }

    pub fn topology(mut self, t: Topology) -> Self {
        self.topology = t;
        self
    }

    /// Star topology: workers dial the coordinator, which never waits to hear
    /// them. This is the shape a phone needs — a handset on Wi-Fi often cannot
    /// accept inbound connections, and in a mesh every rank must reach every
    /// other.
    pub fn star(self) -> Self {
        self.topology(Topology::Star)
    }

    /// Full mesh: every rank connects to every other. The default.
    pub fn mesh(self) -> Self {
        self.topology(Topology::Mesh)
    }

    pub fn device(mut self, d: impl Into<String>) -> Self {
        self.device = d.into();
        self
    }

    pub fn heap_bytes(mut self, n: usize) -> Self {
        self.heap_bytes = Some(n);
        self
    }

    pub fn role(mut self, r: NodeRole) -> Self {
        self.role = r;
        self
    }

    /// Declare this node a pre-synthesized fixed-function rank.
    pub fn fixed_function(self, ff: FixedFunction) -> Self {
        self.role(NodeRole::FixedFunction(ff))
    }

    /// Read the same variables the CLI node driver uses (`RANK`, `WORLD`,
    /// `PEERS` / `DISCOVER`, `TOPOLOGY`, `HEAP_MB`, `DEVICE`). Provided for
    /// parity; prefer the builder on platforms without a shell environment.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_env() -> Result<Self, String> {
        let var = |k: &str| std::env::var(k).ok();
        let rank: u32 = var("RANK")
            .as_deref()
            .unwrap_or("0")
            .parse()
            .map_err(|_| "RANK must be an integer".to_string())?;
        let world: u32 = var("WORLD")
            .as_deref()
            .unwrap_or("1")
            .parse()
            .map_err(|_| "WORLD must be an integer".to_string())?;
        let mut cfg = NodeConfig::new(rank, world).device(var("DEVICE").unwrap_or("auto".into()));

        let star =
            var("TOPOLOGY").as_deref() == Some("star") || var("DIAL_OUT").is_some_and(|v| v != "0");
        cfg = cfg.topology(if star { Topology::Star } else { Topology::Mesh });

        if let Some(mb) = var("HEAP_MB").and_then(|v| v.parse::<usize>().ok()) {
            cfg = cfg.heap_bytes(mb << 20);
        }

        if var("DISCOVER").is_some_and(|v| v != "0") {
            let dp = var("DISC_PORT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(29600);
            let db = var("DATA_PORT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(29500);
            cfg = cfg.discover(dp, db);
            if let Some(h) = var("DISCOVER_HOST") {
                cfg = cfg.discover_via(h);
            }
        } else if let Some(peers) = var("PEERS") {
            let addrs: Vec<String> = peers.split(',').map(|s| s.trim().to_string()).collect();
            cfg = cfg.peers(addrs)?;
        }
        Ok(cfg)
    }

    /// Build the transport and join the mesh.
    pub fn connect(&self) -> std::io::Result<Arc<ProcessGroup>> {
        let mut node = Node::new(self.rank, self.world).topology(self.topology);
        if let Some(n) = self.heap_bytes {
            node = node.heap_bytes(n);
        }
        node = match &self.peers {
            PeerSource::Static(v) => node.peers(v.iter().map(String::as_str))?,
            PeerSource::Discover {
                disc_port,
                data_base,
                via,
            } => {
                let n = node.discover(*disc_port, *data_base);
                match via {
                    Some(h) => n.discover_via(h.clone()),
                    None => n,
                }
            }
        };
        node.connect()
    }

    /// This node's advertised capabilities.
    pub fn caps(&self) -> NodeCaps {
        NodeCaps {
            rank: self.rank,
            role: self.role.clone(),
            devices: crate::available_devices()
                .into_iter()
                .map(|d| d.as_arg().to_string())
                .collect(),
            platform: platform_tag().into(),
        }
    }
}

// ── capability handshake ──────────────────────────────────────────────────

/// Worker: announce capabilities to the coordinator (rank 0).
pub fn send_caps(group: &ProcessGroup, caps: &NodeCaps) -> Result<(), String> {
    let bytes = serde_json::to_vec(caps).map_err(|e| e.to_string())?;
    group
        .transport()
        .send_bytes(0, TAG_CAPS, &bytes)
        .map_err(|e| format!("send_caps: {e}"))
}

/// Coordinator: collect every worker's capabilities (ranks `1..world`).
///
/// Call this before placement so a stage is only ever assigned to hardware that
/// can run it — the difference between a clear error and a silent hang when a
/// fixed-function rank is in the mesh.
pub fn collect_caps(group: &ProcessGroup) -> Result<Vec<NodeCaps>, String> {
    let mut out = Vec::new();
    for r in 1..group.world_size() {
        let bytes = group
            .transport()
            .recv_bytes(r, TAG_CAPS)
            .map_err(|e| format!("collect_caps(rank {r}): {e}"))?;
        out.push(serde_json::from_slice(&bytes).map_err(|e| format!("NodeCaps(rank {r}): {e}"))?);
    }
    Ok(out)
}

// ── run control ───────────────────────────────────────────────────────────

/// Bounds a serving loop: a step budget, plus a stop flag another thread can
/// raise.
///
/// A mobile node needs both. An Android service or an iOS app gets backgrounded,
/// thermally throttled, or unplugged, and must leave the mesh without waiting
/// for the coordinator to finish the job.
///
/// **The flag is observed between activations, not during one.** A node parked
/// in `recv` stays parked until its peer sends or the link drops — stopping is
/// cooperative, not a cancellation.
#[derive(Clone)]
pub struct NodeControl {
    steps: Option<usize>,
    stop: Arc<AtomicBool>,
}

impl NodeControl {
    /// Serve until the coordinator finishes or [`NodeStopHandle::stop`] fires.
    pub fn unbounded() -> Self {
        Self {
            steps: None,
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Serve at most `n` activations.
    pub fn steps(n: usize) -> Self {
        Self {
            steps: Some(n),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A handle another thread can use to wind this node down.
    pub fn stop_handle(&self) -> NodeStopHandle {
        NodeStopHandle(self.stop.clone())
    }

    /// Whether the loop should run another activation.
    pub fn should_continue(&self, done: usize) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            return false;
        }
        self.steps.is_none_or(|limit| done < limit)
    }
}

impl Default for NodeControl {
    fn default() -> Self {
        Self::unbounded()
    }
}

/// Raises the stop flag on a [`NodeControl`] from another thread.
#[derive(Clone)]
pub struct NodeStopHandle(Arc<AtomicBool>);

impl NodeStopHandle {
    /// Ask the node to leave the mesh after its current activation.
    pub fn stop(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_stopped(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

// ── serving ───────────────────────────────────────────────────────────────

/// What a node did, returned instead of printed — a mobile shell has no stdout
/// to scrape.
#[derive(Clone, Debug)]
pub struct NodeReport {
    pub rank: u32,
    pub device: Device,
    pub platform: &'static str,
    /// Activations processed.
    pub activations: usize,
}

/// Serve as a general worker: announce caps, receive a stage, then run
/// activations until the peer closes the link.
///
/// `ctl` bounds the serving loop — a step budget and/or a cooperative stop
/// flag. The default ([`NodeControl::unbounded`]) serves until the transport
/// errors, which is how a node exits when the coordinator finishes.
pub fn serve_worker_n<F>(
    group: &ProcessGroup,
    caps: &NodeCaps,
    resolve: F,
    ctl: &NodeControl,
) -> Result<NodeReport, String>
where
    F: FnMut(&str) -> Vec<f32>,
{
    send_caps(group, caps)?;
    let rank = group.rank();
    let world = group.world_size();
    let mut stage = super::recv_stage(group, resolve)?;
    let next = if rank + 1 < world { rank + 1 } else { 0 };

    let mut n = 0usize;
    while ctl.should_continue(n) {
        let Ok(x) = recv_activation(group, rank - 1) else {
            break; // coordinator finished / link closed — a clean exit
        };
        let y = stage.run(&x);
        send_activation(group, next, &y)?;
        n += 1;
    }
    Ok(NodeReport {
        rank,
        device: stage.device,
        platform: platform_tag(),
        activations: n,
    })
}

/// [`serve_worker_n`] with default capabilities and no step bound.
pub fn serve_worker<F>(group: &ProcessGroup, resolve: F) -> Result<NodeReport, String>
where
    F: FnMut(&str) -> Vec<f32>,
{
    let caps = NodeCaps {
        rank: group.rank(),
        role: NodeRole::Worker,
        devices: crate::available_devices()
            .into_iter()
            .map(|d| d.as_arg().to_string())
            .collect(),
        platform: platform_tag().into(),
    };
    serve_worker_n(group, &caps, resolve, &NodeControl::unbounded())
}

/// Serve as a **fixed-function** rank: announce the one stage this hardware
/// implements, then stream activations through the board.
///
/// No graph is compiled and no `StageSpec` is consumed — the datapath is
/// already synthesized. The activation length is checked against the descriptor
/// on every step, so a mismatched feed is refused instead of being clocked into
/// the fabric.
pub fn serve_fixed_function_n(
    group: &ProcessGroup,
    ff: &FixedFunction,
    board: &mut dyn FixedFunctionDevice,
    ctl: &NodeControl,
) -> Result<NodeReport, String> {
    if board.stage_id() != ff.stage_id {
        return Err(format!(
            "board implements [{}] but rank advertises [{}]",
            board.stage_id(),
            ff.stage_id
        ));
    }
    let caps = NodeCaps {
        rank: group.rank(),
        role: NodeRole::FixedFunction(ff.clone()),
        devices: vec!["fpga".into()],
        platform: platform_tag().into(),
    };
    send_caps(group, &caps)?;

    let rank = group.rank();
    let world = group.world_size();
    let next = if rank + 1 < world { rank + 1 } else { 0 };

    let mut n = 0usize;
    while ctl.should_continue(n) {
        let Ok(x) = recv_activation(group, rank - 1) else {
            break;
        };
        if x.len() != ff.input_elems {
            return Err(format!(
                "fixed-function stage [{}] takes {} elems, got {}: the datapath \
                 is synthesized for one shape and cannot adapt",
                ff.stage_id,
                ff.input_elems,
                x.len()
            ));
        }
        let y = board.execute(&x)?;
        if y.len() != ff.output_elems {
            return Err(format!(
                "fixed-function stage [{}] declares {} output elems, board \
                 returned {}",
                ff.stage_id,
                ff.output_elems,
                y.len()
            ));
        }
        send_activation(group, next, &y)?;
        n += 1;
    }
    Ok(NodeReport {
        rank,
        device: Device::Cpu, // host lane; the datapath itself is off-CPU
        platform: platform_tag(),
        activations: n,
    })
}

// ── training ──────────────────────────────────────────────────────────────

/// What a node did in a training run.
#[derive(Clone, Debug)]
pub struct TrainReport {
    pub rank: u32,
    pub platform: &'static str,
    /// Metrics from the local training loop, including which backends ran.
    pub metrics: super::TrainMetrics,
    /// Final synced parameters, as `(name, values)`.
    pub params: Vec<(String, Vec<f32>)>,
}

/// Serve as a **training** worker: announce capabilities, receive a
/// [`TrainSpec`](super::TrainSpec) the coordinator ships, and run the
/// data-parallel loop against this node's own hardware.
///
/// No model code is baked in — the coordinator owns the graph, the worker just
/// executes it. Weights and data stay node-local; only the spec crosses the
/// wire (or the data shard, when the spec sets `push_data` because there is no
/// shared filesystem — a phone never has one).
///
/// The cross-rank gradient reduce runs on this group's
/// [`all_reduce`](ProcessGroup::all_reduce), so the node driver needs no
/// dependency on the in-graph collectives crate.
///
/// **Every rank must call this in lockstep.** The reduce is a barrier, so a
/// node that leaves early — a phone going to background — stalls the rest.
/// That is why there is no step budget here: a training run is not something a
/// node can drop out of halfway.
pub fn serve_trainer<F>(
    group: &ProcessGroup,
    caps: &NodeCaps,
    mut resolve: F,
    log: bool,
) -> Result<TrainReport, String>
where
    F: FnMut(&str) -> Vec<f32>,
{
    send_caps(group, caps)?;
    let mut spec = super::recv_train(group)?;

    // No shared filesystem: pull this rank's shard and rewrite the spec to the
    // files we just received.
    if spec.push_data {
        super::pull_shards(group, &mut spec, &std::env::temp_dir())?;
    }

    let world = group.world_size();
    let (metrics, params) = super::run_train(
        &spec,
        world,
        |uri| resolve(uri),
        |flat| mean_reduce(group, flat),
        log,
    )?;

    Ok(TrainReport {
        rank: group.rank(),
        platform: platform_tag(),
        metrics,
        params,
    })
}

/// Average a flat gradient buffer across every rank in `group`.
///
/// The reduce a data-parallel step needs, exposed because **every rank must use
/// the same one**: the coordinator trains alongside the workers, and a group
/// where one side sums while another averages scales the step by the world size
/// on some replicas and not others — which shows up as slow divergence, not as
/// an error.
///
/// Mean rather than Sum because [`run_train`](super::run_train) applies the
/// result directly as the gradient.
pub fn mean_reduce(group: &ProcessGroup, flat: &[f32]) -> Vec<f32> {
    let mut buf = flat.to_vec();
    match group.all_reduce(&mut buf, ReduceKind::Mean) {
        Ok(()) => buf,
        // `run_train`'s reduce closure has no error channel. Returning the
        // unreduced buffer would silently train on local gradients only, which
        // looks like convergence and is not; zeros stall the step visibly.
        Err(_) => vec![0.0; flat.len()],
    }
}

/// [`serve_trainer`] with capabilities derived from this node.
pub fn serve_trainer_here<F>(
    group: &ProcessGroup,
    resolve: F,
    log: bool,
) -> Result<TrainReport, String>
where
    F: FnMut(&str) -> Vec<f32>,
{
    let caps = NodeCaps {
        rank: group.rank(),
        role: NodeRole::Worker,
        devices: crate::available_devices()
            .into_iter()
            .map(|d| d.as_arg().to_string())
            .collect(),
        platform: platform_tag().into(),
    };
    serve_trainer(group, &caps, resolve, log)
}

/// Resolve this node's device directive to a concrete backend.
pub fn node_device(spec: &str) -> Device {
    resolve_device(spec)
}

/// Weight cache constructor re-exported for shells that resolve their own URIs.
pub fn weight_cache() -> WeightCache {
    WeightCache::new()
}
