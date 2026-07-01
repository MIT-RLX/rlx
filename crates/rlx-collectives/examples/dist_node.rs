// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Cross-machine distributed smoke test: one process per rank joins a
//! real [`NetTransport`] TCP mesh, then runs
//!
//!   * **tensor-parallel inference** — a K-sharded matmul whose partial
//!     products are summed by the *in-graph* `collective.all_reduce`,
//!     checked against the analytic full `x @ W` on every rank; and
//!   * **data-parallel training** — a tiny linear-regression step where
//!     each rank holds a *row shard* of the batch, computes its local
//!     gradient (forward pass through the RLX runtime), and the gradients
//!     are summed across ranks with `ProcessGroup::all_reduce`. The
//!     distributed weight trajectory is checked, step for step, against a
//!     single-process full-batch reference.
//!
//! Both phases drive the SAME mesh, so a green run proves the multi-node
//! transport end to end (the path `rlx-collectives`' README flags as
//! "multi-node hardware not exercised").
//!
//! Config is via env vars (so it runs identically on every node):
//!
//! ```text
//!   RANK=0 WORLD=2 PEERS=127.0.0.1:29500,127.0.0.1:29501 MODE=both \
//!     cargo run -q -p rlx-collectives --example dist_node
//! ```
//!
//!   RANK   this process's rank in 0..WORLD            (default 0)
//!   WORLD  number of ranks                            (default 1)
//!   PEERS  comma-separated host:port, one per rank    (default loopback pair)
//!   MODE   infer | train | both | bench | pipeline | topology | placement |
//!          multidev | coordinator | worker            (default both)
//!          coordinator/worker = ship-graph thin-executor model (rank 0 is
//!          the coordinator, ranks ≥1 are generic workers);
//!          placement = per-backend feasibility + run-the-graph-on-each demo;
//!          multidev  = split one forward into regions on different backends
//!                      (REGIONS=cpu,metal,cpu); WEIGHTS_DIR=<dir> makes the
//!                      coordinator emit file:// weight URIs instead of seed://
//!   DEVICE auto | cpu | metal | cuda | mlx | vulkan | … | `cuda,cpu`  (default cpu)
//!          `auto` = fastest available; a single device is forced (aborts if
//!          missing); a comma list is an allow-list in preference order
//!          (GPU-first, CPU fallback), honored against per-graph op feasibility
//!   WORKER_DEVICE  device spec the coordinator hands each worker (default auto)
//!   STEPS  training steps                             (default 200)
//!   ACCUM  micro-passes accumulated per gradient sync (default 1)
//!   LABEL  free-text node label for the banner        (default "rank<R>")
//!
//! With `DEVICE=metal` (Mac) / `DEVICE=cuda` (NVIDIA) the matmuls run on
//! that backend; the cross-rank reductions stay on the host (there is no
//! device-resident `collective.all_reduce` outside MLX yet), so the GPU
//! does the compute and the mesh does the communication — exactly the
//! split a real tensor/data-parallel deployment uses today.
//!
//! Across two machines the cleanest way to satisfy "every rank reaches
//! PEERS[higher]" without poking firewall holes is an SSH tunnel, so the
//! PEERS addresses stay on loopback at both ends.

use rlx_collectives::planner::{self, Link};
use rlx_collectives::{all_reduce as graph_all_reduce, register, register_group, unregister_group};
use rlx_driver::{DEFAULT_HEAP_BYTES, ProcessGroup, ReduceKind, TcpTransport};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{
    Device, DevicePolicy, GraphDevices, Session, device_label, fastest_device, full_name,
    is_available, parse_device, parse_device_list,
};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// In-graph collective group id this process registers its group under.
/// Only needs to be consistent *within* a process (the graph carries it
/// in op attrs and the kernel resolves it locally), so a constant is fine.
const GID: u64 = 0;

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn parse_peers(s: &str) -> Vec<SocketAddr> {
    s.split(',')
        .map(|h| {
            h.trim()
                .to_socket_addrs_first()
                .unwrap_or_else(|| panic!("bad peer address: {h:?}"))
        })
        .collect()
}

/// Resolve a single `host:port` to a `SocketAddr` (handles names too).
trait ToSocketAddrsFirst {
    fn to_socket_addrs_first(&self) -> Option<SocketAddr>;
}
impl ToSocketAddrsFirst for str {
    fn to_socket_addrs_first(&self) -> Option<SocketAddr> {
        use std::net::ToSocketAddrs;
        self.to_socket_addrs().ok()?.next()
    }
}

// ── deterministic global problems (same on every rank) ─────────────

// Inference: y = x @ W, with x [B,K], W [K,N].
const B: usize = 2;
const K: usize = 8;
const N: usize = 4;
fn x_full() -> Vec<f32> {
    (0..B * K).map(|i| (i as f32 * 0.1).sin()).collect()
}
fn w_full() -> Vec<f32> {
    (0..K * N).map(|i| (i as f32 * 0.07).cos()).collect()
}

// Training: linear regression, X [M,D], targets y = X @ w_true (noiseless).
const M: usize = 8; // batch rows (must divide by WORLD)
const D: usize = 4; // features
fn x_design() -> Vec<f32> {
    (0..M * D)
        .map(|i| (i as f32 * 0.37 + 1.0).sin() * 0.5)
        .collect()
}
fn w_true() -> Vec<f32> {
    (0..D).map(|d| (d as f32 * 0.5).cos() * 0.7).collect()
}
fn y_targets(x: &[f32], w: &[f32]) -> Vec<f32> {
    (0..M)
        .map(|i| (0..D).map(|d| x[i * D + d] * w[d]).sum())
        .collect()
}
const LR: f32 = 0.15;

fn main() {
    let rank: u32 = env("RANK", "0").parse().expect("RANK");
    let world: u32 = env("WORLD", "1").parse().expect("WORLD");
    let mode = env("MODE", "both");
    let steps: usize = env("STEPS", "200").parse().expect("STEPS");
    let accum: usize = env("ACCUM", "1").parse::<usize>().expect("ACCUM").max(1);
    let label = env("LABEL", &format!("rank{rank}"));
    let device = resolve_device(&env("DEVICE", "cpu"), &label);
    // DISCOVER=1 → find peers by UDP broadcast (zero-config); else PEERS env.
    let peers = if env("DISCOVER", "0") != "0" {
        let disc_port: u16 = env("DISC_PORT", "29600").parse().expect("DISC_PORT");
        let data_base: u16 = env("DATA_PORT", "29500").parse().expect("DATA_PORT");
        discover_peers(rank, world, disc_port, data_base, &label)
    } else {
        parse_peers(&env("PEERS", "127.0.0.1:29500,127.0.0.1:29501"))
    };
    assert_eq!(
        peers.len(),
        world as usize,
        "PEERS must list WORLD={world} addresses, got {}",
        peers.len()
    );

    eprintln!(
        "[{label}] rank {rank}/{world}  device={}  binding {}  peers={peers:?}  mode={mode}",
        device_label(device),
        peers[rank as usize]
    );
    print_inventory(&label);

    register(); // install in-graph collective op (idempotent)

    // Join the full-mesh TCP transport. rank r connects out to every
    // higher rank and accepts from every lower rank; a 2-rank mesh is a
    // single rank0->rank1 connection carrying both directions.
    let transport = TcpTransport::bind(rank, world, peers.clone(), DEFAULT_HEAP_BYTES)
        .expect("TcpTransport::bind — is the peer up / reachable?");
    let group = Arc::new(ProcessGroup::new(Arc::new(transport)));
    register_group(GID, group.clone());

    group.barrier().expect("initial barrier");
    eprintln!("[{label}] mesh connected ✓");

    let mut ok = true;

    if mode == "bench" {
        run_bench(&group, &label, parse_dtype(&env("DTYPE", "f32")));
        group.barrier().expect("post-bench barrier");
    }

    if mode == "pipeline" {
        ok &= run_pipeline(&group, &label, device);
        group.barrier().expect("post-pipeline barrier");
    }

    if mode == "topology" {
        run_topology(&group, &label, device);
        group.barrier().expect("post-topology barrier");
    }

    if mode == "placement" {
        run_placement(&group, &label, &env("DEVICE", "auto"));
        group.barrier().expect("post-placement barrier");
    }

    if mode == "multidev" {
        ok &= run_multidev(&group, &label, &env("REGIONS", ""));
        group.barrier().expect("post-multidev barrier");
    }

    // Coordinator/worker (thin-executor) model: the worker runs NO model code —
    // it executes a serialized graph the coordinator ships and resolves its own
    // weights from a manifest. Run rank 0 as `coordinator`, the rest as `worker`.
    if mode == "coordinator" {
        ok &= run_coordinator(&group, &label, device);
        group.barrier().expect("post-coordinator barrier");
    }
    if mode == "worker" {
        ok &= run_worker(&group, &label);
        group.barrier().expect("post-worker barrier");
    }

    if mode == "infer" || mode == "both" {
        ok &= run_inference(&group, &label, device);
        // The in-graph `collective.all_reduce` op (this crate's headline) only
        // ships a CPU kernel, so it's a GLOBAL opt-in (INGRAPH=1, all ranks on
        // cpu) — never gated on the *local* device, which would desync a
        // heterogeneous mesh (one rank entering the collective, others not).
        if env("INGRAPH", "0") != "0" {
            ok &= run_inference_ingraph(&group, &label);
        }
        group.barrier().expect("post-inference barrier");
    }

    if mode == "train" || mode == "both" {
        ok &= run_training(&group, &label, steps, device, accum);
        group.barrier().expect("post-training barrier");
    }

    unregister_group(GID);
    // Keep reader threads alive for any late peer until everyone is done.
    group.barrier().expect("final barrier");

    if ok {
        eprintln!("[{label}] ALL DISTRIBUTED CHECKS PASSED ✓");
    } else {
        eprintln!("[{label}] DISTRIBUTED CHECKS FAILED ✗");
        std::process::exit(1);
    }
}

/// The default-route local IP (picks the outbound interface; on the Mac
/// that's the wired GbE, on msi the WiFi address).
fn local_ip() -> IpAddr {
    let s = UdpSocket::bind("0.0.0.0:0").expect("udp bind");
    s.connect("8.8.8.8:80").expect("udp connect"); // no packet sent; selects iface
    s.local_addr().expect("local_addr").ip()
}

/// UDP-broadcast rendezvous: every node announces `rank -> ip:dataport` on
/// `disc_port` and listens until it has all `world` addresses, then returns
/// the peer list sorted by rank. This is the zero-config "devices find each
/// other" property (minimal form: ranks are assigned, addresses discovered)
/// — no hand-maintained PEERS list, no hardcoded IPs.
fn discover_peers(
    rank: u32,
    world: u32,
    disc_port: u16,
    data_base: u16,
    label: &str,
) -> Vec<SocketAddr> {
    use std::collections::BTreeMap;
    let my_addr = SocketAddr::new(local_ip(), data_base + rank as u16);
    let sock = UdpSocket::bind(("0.0.0.0", disc_port)).expect("discovery bind");
    sock.set_broadcast(true).ok();
    sock.set_read_timeout(Some(Duration::from_millis(150))).ok();
    let bcast = SocketAddr::new(IpAddr::V4(Ipv4Addr::BROADCAST), disc_port);
    let msg = format!("RLXDISC {rank} {my_addr}");

    let mut peers: BTreeMap<u32, SocketAddr> = BTreeMap::new();
    peers.insert(rank, my_addr);
    let mut buf = [0u8; 256];
    while (peers.len() as u32) < world {
        let _ = sock.send_to(msg.as_bytes(), bcast);
        if let Ok((n, _)) = sock.recv_from(&mut buf)
            && let Ok(s) = std::str::from_utf8(&buf[..n])
        {
            let mut it = s.split_whitespace();
            if it.next() == Some("RLXDISC")
                && let (Some(r), Some(a)) = (it.next(), it.next())
                && let (Ok(r), Ok(a)) = (r.parse::<u32>(), a.parse::<SocketAddr>())
            {
                peers.insert(r, a);
            }
        }
    }
    // Brief drain so late joiners still hear our announcement before we move on.
    for _ in 0..5 {
        let _ = sock.send_to(msg.as_bytes(), bcast);
    }
    let list: Vec<SocketAddr> = peers.into_values().collect();
    eprintln!("[{label}] discovered {world} peers via UDP: {list:?}");
    list
}

/// `(this rank's x-slice [B,kr], W-slice [kr,N], full reference y [B,N], kr)`
/// for the K-sharded tensor-parallel matmul.
fn infer_shards(rank: usize, world: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    assert_eq!(K % world, 0, "K={K} must divide WORLD={world}");
    let kr = K / world;
    let x = x_full();
    let w = w_full();

    let mut y_ref = vec![0f32; B * N];
    for b in 0..B {
        for j in 0..N {
            let mut s = 0.0f32;
            for k in 0..K {
                s += x[b * K + k] * w[k * N + j];
            }
            y_ref[b * N + j] = s;
        }
    }

    let k0 = rank * kr;
    let mut x_r = vec![0f32; B * kr];
    for b in 0..B {
        for i in 0..kr {
            x_r[b * kr + i] = x[b * K + k0 + i];
        }
    }
    let mut w_r = vec![0f32; kr * N];
    for i in 0..kr {
        for j in 0..N {
            w_r[i * N + j] = w[(k0 + i) * N + j];
        }
    }
    (x_r, w_r, y_ref, kr)
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── Coordinator / thin-worker model ───────────────────────────────────────
// One tag for the whole stage envelope; one for activations.
const TAG_SPEC: u32 = 20; // serialized StageSpec
const TAG_ACT: u32 = 22; // activation hand-off

/// A self-describing unit of work for a worker: the graph, its I/O node names,
/// where each weight comes from, and the placement directive. The worker reads
/// it all from here — nothing is implied by convention.
#[derive(serde::Serialize, serde::Deserialize)]
struct StageSpec {
    graph: Graph,
    input_name: String,
    output_name: String,
    params: Vec<ParamSource>,
    device: String, // device-policy spec: auto | cuda | cuda,cpu | …
}

/// Where a parameter's bytes come from; the worker resolves it itself.
#[derive(serde::Serialize, serde::Deserialize)]
struct ParamSource {
    name: String,
    uri: String, // seed://<n>?len=<k> | file://<path> | (http(s):// future)
}

/// Resolve a weight tensor from its source URI — the worker's own job, so
/// large weights stay node-local (HF / local file / object store in a real
/// deployment), never on the wire.
fn resolve_weights(uri: &str, label: &str) -> Vec<f32> {
    if let Some(rest) = uri.strip_prefix("seed://") {
        let (seed_s, len) = match rest.split_once('?') {
            Some((s, q)) => (
                s,
                q.strip_prefix("len=")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
            ),
            None => (rest, 0),
        };
        materialize_weights(seed_s.parse().unwrap_or(0), len)
    } else if let Some(path) = uri.strip_prefix("file://") {
        // "access local data": read a little-endian f32 tensor from disk.
        match std::fs::read(path) {
            Ok(b) => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Err(e) => {
                eprintln!("[{label}] weight load {path}: {e}");
                Vec::new()
            }
        }
    } else {
        eprintln!("[{label}] unknown weight URI scheme: {uri}");
        Vec::new()
    }
}

/// Deterministic stand-in weights — the `seed://` scheme.
fn materialize_weights(seed: u64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u64).wrapping_add(seed.wrapping_mul(1009))) as f32 * 0.013).cos() * 0.2)
        .collect()
}

/// One pipeline stage as a graph: `h [batch,d] @ W [d,d] -> [batch,d]`.
fn stage_graph(batch: usize, d: usize) -> Graph {
    let mut g = Graph::new("stage");
    let h = g.input("h", Shape::new(&[batch, d], DType::F32));
    let w = g.param("W", Shape::new(&[d, d], DType::F32));
    let mm = g.matmul(h, w, Shape::new(&[batch, d], DType::F32));
    g.set_outputs(vec![mm]);
    g
}

/// Coordinator (rank 0): owns the "model". It ships each worker a *serialized*
/// stage graph + a weight manifest, runs stage 0 locally, then drives the
/// activation through the worker chain and checks the result. The workers hold
/// no model code and load their own weights — only graphs (KB) and activations
/// cross the wire, never weights.
fn run_coordinator(group: &Arc<ProcessGroup>, label: &str, device: Device) -> bool {
    let n = group.world_size() as usize;
    let (batch, d) = (2usize, 8usize);
    let x: Vec<f32> = (0..batch * d).map(|i| (i as f32 * 0.1).sin()).collect();

    // Full-model reference (coordinator knows every stage's seed).
    let mut href = x.clone();
    for s in 0..n {
        href = matmul_host(&href, &materialize_weights(s as u64, d * d), batch, d, d);
    }

    // Ship each worker a self-describing StageSpec: the graph, its I/O names,
    // where to fetch each weight (URI, node-local), and the placement spec.
    // With WEIGHTS_DIR set, weights are written to disk and referenced by a
    // file:// URI (real local-data load); otherwise the seed:// stand-in.
    let worker_dev = std::env::var("WORKER_DEVICE").unwrap_or_else(|_| "auto".into());
    let weights_dir = std::env::var("WEIGHTS_DIR").ok();
    for r in 1..n {
        let uri = match &weights_dir {
            Some(dir) => {
                let path = format!("{dir}/stage_{r}.f32");
                let bytes: Vec<u8> = materialize_weights(r as u64, d * d)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                std::fs::write(&path, &bytes).expect("write weights file");
                format!("file://{path}")
            }
            None => format!("seed://{r}?len={}", d * d),
        };
        let spec = StageSpec {
            graph: stage_graph(batch, d),
            input_name: "h".into(),
            output_name: "y".into(),
            params: vec![ParamSource {
                name: "W".into(),
                uri,
            }],
            device: worker_dev.clone(),
        };
        let bytes = serde_json::to_vec(&spec).expect("serialize spec");
        group
            .transport()
            .send_bytes(r as u32, TAG_SPEC, &bytes)
            .expect("send spec");
    }
    eprintln!(
        "[{label}] COORDINATOR: shipped {} self-describing StageSpec(s) [graph + I/O names + weight URIs], placement='{worker_dev}'",
        n.saturating_sub(1)
    );

    // Stage 0 on the coordinator's own device.
    let mut c0 = Session::new(device).compile(stage_graph(batch, d));
    c0.set_param("W", &materialize_weights(0, d * d));
    let mut h = c0.run(&[("h", x.as_slice())]).into_iter().next().unwrap();

    // Drive activation through the chain (0 → 1 → … → n-1 → 0).
    if n > 1 {
        group.send_f32(1, TAG_ACT, &h).expect("send act");
        h = group.recv_f32(n as u32 - 1, TAG_ACT).expect("recv final");
    }

    let err = max_abs_err(&h, &href);
    let pass = err < 1e-4;
    eprintln!(
        "[{label}] COORDINATOR: pipeline output max_err={err:.2e} vs full-model reference  {}",
        if pass { "PASS ✓" } else { "FAIL ✗" }
    );
    pass
}

/// Worker (rank r > 0): a GENERIC executor. From a received `StageSpec` it
/// learns the graph, its I/O names, where to fetch each weight, and the
/// placement — nothing baked in by convention. It resolves its own weights,
/// honors the directed backend, runs the handed activation, and forwards it.
fn run_worker(group: &Arc<ProcessGroup>, label: &str) -> bool {
    let rank = group.rank() as usize;
    let n = group.world_size() as usize;

    // 1. Receive the self-describing spec — the only thing the worker knows.
    let bytes = group
        .transport()
        .recv_bytes(0, TAG_SPEC)
        .expect("recv spec");
    let spec: StageSpec = match serde_json::from_slice(&bytes) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[{label}] WORKER rank {rank}: StageSpec deserialize failed: {e}");
            return false;
        }
    };
    let nodes = spec.graph.len();

    // 2. Placement: honor spec.device vs this node's hardware + op feasibility.
    let (policy, hint) = parse_device_policy(&spec.device);
    let resolved = {
        let g2 = serde_json::from_slice::<StageSpec>(&bytes)
            .map(|s| s.graph)
            .expect("spec re-parse");
        GraphDevices::with_policy(g2, policy)
            .resolve(hint)
            .unwrap_or_else(|_| fastest_device())
    };
    let mut compiled = Session::new(resolved).compile(spec.graph);

    // 3. Resolve every weight from its source URI (node-local, off the wire).
    for p in &spec.params {
        compiled.set_param(&p.name, &resolve_weights(&p.uri, label));
    }

    // 4. Run with the spec's NAMED input; forward the result downstream.
    let input = group.recv_f32(rank as u32 - 1, TAG_ACT).expect("recv act");
    let out = compiled
        .run(&[(spec.input_name.as_str(), input.as_slice())])
        .into_iter()
        .next()
        .unwrap();
    let next = if rank + 1 < n { rank as u32 + 1 } else { 0 };
    group.send_f32(next, TAG_ACT, &out).expect("forward act");

    let sources: Vec<&str> = spec.params.iter().map(|p| p.uri.as_str()).collect();
    eprintln!(
        "[{label}] WORKER rank {rank}: ran {nodes}-node graph on {} (placement '{}') | in='{}' out='{}' | weights from [{}] → rank {next}",
        device_label(resolved),
        spec.device,
        spec.input_name,
        spec.output_name,
        sources.join(", ")
    );
    true
}

/// Plain host `[m,k] @ [k,n] -> [m,n]` for references.
fn matmul_host(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// Pipeline parallelism: the model is split into `world` sequential stages,
/// one per rank. Activations flow rank→rank (point-to-point `send`/`recv`,
/// the PP primitive), each stage a `relu(h @ W_stage)` on the local backend.
/// The last rank checks its output against the full single-node forward.
/// This is the comm-light parallelism (one hand-off per boundary) that suits
/// a slow link or a model too big for one device.
fn run_pipeline(group: &Arc<ProcessGroup>, label: &str, device: Device) -> bool {
    let rank = group.rank() as usize;
    let n = group.world_size() as usize;
    let (batch, d) = (2usize, 8usize);

    let x: Vec<f32> = (0..batch * d).map(|i| (i as f32 * 0.1).sin()).collect();
    let stage_w = |s: usize| -> Vec<f32> {
        (0..d * d)
            .map(|i| (((i + s * 7) as f32) * 0.03).cos() * 0.2)
            .collect()
    };
    let relu = |v: &mut [f32]| v.iter_mut().for_each(|x| *x = x.max(0.0));

    // Full forward reference (every rank can compute it deterministically).
    let mut href = x.clone();
    for s in 0..n {
        href = matmul_host(&href, &stage_w(s), batch, d, d);
        relu(&mut href);
    }

    // This rank's stage on the selected backend: input @ W_rank -> relu.
    let mut g = Graph::new("pp_stage");
    let hin = g.input("h", Shape::new(&[batch, d], DType::F32));
    let wp = g.param("W", Shape::new(&[d, d], DType::F32));
    let mm = g.matmul(hin, wp, Shape::new(&[batch, d], DType::F32));
    g.set_outputs(vec![mm]);
    let mut compiled = Session::new(device).compile(g);
    compiled.set_param("W", &stage_w(rank));

    const TAG: u32 = 11;
    let input = if rank == 0 {
        x.clone()
    } else {
        group.recv_f32(rank as u32 - 1, TAG).expect("pipeline recv")
    };
    let mut h = compiled
        .run(&[("h", input.as_slice())])
        .into_iter()
        .next()
        .unwrap();
    relu(&mut h);
    if rank + 1 < n {
        group
            .send_f32(rank as u32 + 1, TAG, &h)
            .expect("pipeline send");
    }

    let pass = if rank == n - 1 {
        max_abs_err(&h, &href) < 1e-4
    } else {
        true // intermediate stages can't see the final output
    };
    eprintln!(
        "[{label}] PIPELINE stage {rank}/{n} ({} fwd){}",
        device_label(device),
        if rank == n - 1 {
            format!(
                ", final output max_err={:.2e}  {}",
                max_abs_err(&h, &href),
                if pass { "PASS ✓" } else { "FAIL ✗" }
            )
        } else {
            " → handed off ✓".to_string()
        }
    );
    pass
}

/// Tensor-parallel matmul, **GPU-friendly split**: each rank runs its
/// partial `x_r @ W_r` on the selected backend (Metal / CUDA / CPU), then
/// the partials are summed across ranks on the host. Result must equal
/// the full `x @ W`. This is the path that actually exercises rlx-metal /
/// rlx-cuda in a distributed setting.
fn run_inference(group: &Arc<ProcessGroup>, label: &str, device: Device) -> bool {
    let rank = group.rank() as usize;
    let world = group.world_size() as usize;
    let (x_r, w_r, y_ref, kr) = infer_shards(rank, world);

    let mut g = Graph::new("tp_mm_dev");
    let xin = g.input("x", Shape::new(&[B, kr], DType::F32));
    let wp = g.param("W", Shape::new(&[kr, N], DType::F32));
    let mm = g.matmul(xin, wp, Shape::new(&[B, N], DType::F32));
    g.set_outputs(vec![mm]);

    let mut compiled = Session::new(device).compile(g);
    compiled.set_param("W", &w_r);
    let mut partial = compiled
        .run(&[("x", x_r.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // Sum the per-rank partials over the mesh (host collective).
    group
        .all_reduce(&mut partial, ReduceKind::Sum)
        .expect("inference all_reduce");

    let max_err = max_abs_err(&partial, &y_ref);
    let pass = partial.len() == B * N && max_err < 1e-4;
    eprintln!(
        "[{label}] INFERENCE (tensor-parallel {world}-way matmul, {} compute + host all-reduce): \
         max_err={max_err:.2e}  {}",
        device_label(device),
        if pass { "PASS ✓" } else { "FAIL ✗" }
    );
    pass
}

/// Same matmul, but the cross-rank sum runs *inside the compiled graph*
/// via this crate's `collective.all_reduce` op (CPU kernel only).
fn run_inference_ingraph(group: &Arc<ProcessGroup>, label: &str) -> bool {
    let rank = group.rank() as usize;
    let world = group.world_size() as usize;
    let (x_r, w_r, y_ref, kr) = infer_shards(rank, world);

    let mut g = Graph::new("tp_mm_ingraph");
    let xin = g.input("x", Shape::new(&[B, kr], DType::F32));
    let wp = g.param("W", Shape::new(&[kr, N], DType::F32));
    let mm = g.matmul(xin, wp, Shape::new(&[B, N], DType::F32));
    let out = graph_all_reduce(&mut g, mm, GID);
    g.set_outputs(vec![out]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    compiled.set_param("W", &w_r);
    let y = compiled
        .run(&[("x", x_r.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    let max_err = max_abs_err(&y, &y_ref);
    let pass = y.len() == B * N && max_err < 1e-4;
    eprintln!(
        "[{label}] INFERENCE (in-graph collective.all_reduce, cpu): max_err={max_err:.2e}  {}",
        if pass { "PASS ✓" } else { "FAIL ✗" }
    );
    pass
}

/// Data-parallel SGD on a linear-regression MSE loss. Each rank holds a
/// disjoint row shard; per-rank gradients are summed across ranks
/// (gather-to-root all-reduce over the mesh) to form the exact
/// full-batch gradient. The distributed weight trajectory is compared,
/// step for step, to a single-process full-batch reference.
fn run_training(
    group: &Arc<ProcessGroup>,
    label: &str,
    steps: usize,
    device: Device,
    accum: usize,
) -> bool {
    let rank = group.rank() as usize;
    let world = group.world_size() as usize;
    assert_eq!(M % world, 0, "M={M} must divide WORLD={world}");
    let m_local = M / world;

    let x = x_design();
    let w_t = w_true();
    let y = y_targets(&x, &w_t);

    // This rank's row shard of X and y.
    let r0 = rank * m_local;
    let x_sh: Vec<f32> = x[r0 * D..(r0 + m_local) * D].to_vec();
    let y_sh: Vec<f32> = y[r0..r0 + m_local].to_vec();

    // Forward pass pred = X_shard @ w  through the RLX runtime. w is the
    // param we re-set each step; X_shard is the (constant) input.
    let mut g = Graph::new("dp_forward");
    let xin = g.input("X", Shape::new(&[m_local, D], DType::F32));
    let wp = g.param("w", Shape::new(&[D, 1], DType::F32));
    let pred = g.matmul(xin, wp, Shape::new(&[m_local, 1], DType::F32));
    g.set_outputs(vec![pred]);
    let mut compiled = Session::new(device).compile(g);

    // Identical init on every rank.
    let mut w = vec![0.0f32; D];
    // Single-process full-batch reference trajectory (computed locally).
    let mut w_ref = vec![0.0f32; D];

    let mut first_loss = f32::NAN;
    let mut last_loss = f32::NAN;
    let mut max_w_dev = 0.0f32; // distributed vs reference
    let mut comm_calls = 0usize; // gradient all-reduces actually issued

    for step in 0..steps {
        // ---- distributed step: accumulate `accum` micro-passes, sync once ----
        // The shard is static, so averaging `accum` passes reproduces the
        // accum=1 trajectory exactly — the point is to amortize ONE network
        // sync over `accum` forward/backward passes (lower comm:compute).
        let mut grad = vec![0.0f32; D];
        let mut local_sse = 0.0f32;
        for _ in 0..accum {
            compiled.set_param("w", &w);
            let pred = compiled
                .run(&[("X", x_sh.as_slice())])
                .into_iter()
                .next()
                .unwrap();
            // gradient contribution scaled by 2/M so the SUM across ranks is
            // the full-batch gradient (2/M) Xᵀ(Xw−y).
            for i in 0..m_local {
                let r = pred[i] - y_sh[i];
                local_sse += r * r;
                for d in 0..D {
                    grad[d] += (2.0 / M as f32) * x_sh[i * D + d] * r;
                }
            }
        }
        let inv = 1.0 / accum as f32;
        for g in grad.iter_mut() {
            *g *= inv;
        }
        local_sse *= inv;

        // Fire the gradient all-reduce NON-BLOCKING and overlap it with the
        // single-process reference step (real local compute) before joining.
        let pending = group.spawn_all_reduce(grad, ReduceKind::Sum);
        comm_calls += 1;

        // ---- single-process full-batch reference step (runs concurrently) ----
        let mut grad_ref = vec![0.0f32; D];
        for i in 0..M {
            let mut p = 0.0f32;
            for d in 0..D {
                p += x[i * D + d] * w_ref[d];
            }
            let r = p - y[i];
            for d in 0..D {
                grad_ref[d] += (2.0 / M as f32) * x[i * D + d] * r;
            }
        }
        for d in 0..D {
            w_ref[d] -= LR * grad_ref[d];
        }

        // join the overlapped collective -> exact full-batch gradient
        let grad = pending.join().expect("overlapped all_reduce");
        for d in 0..D {
            w[d] -= LR * grad[d];
        }

        // global loss only at the ends (logging); no per-step loss sync.
        if step == 0 || step == steps - 1 {
            let mut loss = vec![local_sse];
            group
                .all_reduce(&mut loss, ReduceKind::Sum)
                .expect("loss all_reduce");
            let mse = loss[0] / M as f32;
            if step == 0 {
                first_loss = mse;
            }
            last_loss = mse;
        }

        let dev = w
            .iter()
            .zip(&w_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        max_w_dev = max_w_dev.max(dev);
    }

    // recovered weights vs ground truth
    let recover_err = w
        .iter()
        .zip(&w_t)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let converged = last_loss <= first_loss; // loss did not diverge
    let matches_ref = max_w_dev < 5e-3; // distributed == single-node
    let pass = converged && matches_ref;
    let passes = steps * accum;
    eprintln!(
        "[{label}] TRAINING  (data-parallel {world}-way, {} fwd, {steps} steps ×{accum} accum): \
         loss {first_loss:.4e} -> {last_loss:.4e}  |  ‖w-w*‖∞={recover_err:.2e}  \
         dist-vs-singlenode max dev={max_w_dev:.2e}  |  comm: {comm_calls} grad-syncs \
         over {passes} fwd passes (overlapped)  {}",
        device_label(device),
        if pass { "PASS ✓" } else { "FAIL ✗" }
    );
    pass
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Parse a reduce dtype for the collective bench.
fn parse_dtype(s: &str) -> DType {
    match s.trim().to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "half" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        "f64" | "fp64" | "double" => DType::F64,
        _ => DType::F32,
    }
}

/// Measure the cost of the cross-rank collective itself, so the
/// compute-vs-communication crossover can be computed from real numbers.
/// Reports a latency floor (barrier RTT) and `all_reduce` time vs payload.
fn run_bench(group: &Arc<ProcessGroup>, label: &str, dtype: DType) {
    let rank = group.rank();
    let leader = rank == 0;
    let esz = dtype.size_bytes();

    // ── latency floor: barrier round-trips ──
    for _ in 0..20 {
        group.barrier().unwrap();
    } // warm up
    let mut bar = Vec::new();
    for _ in 0..200 {
        let t = Instant::now();
        group.barrier().unwrap();
        bar.push(t.elapsed().as_secs_f64());
    }
    let alpha_s = median(bar.clone());
    if leader {
        let mn = bar.iter().cloned().fold(f64::MAX, f64::min) * 1e3;
        eprintln!(
            "[{label}] barrier RTT: median {:.2} ms, min {mn:.2} ms (latency floor α)",
            alpha_s * 1e3
        );
    }

    // ── all_reduce time vs payload size (for the chosen reduce dtype) ──
    if leader {
        eprintln!(
            "[{label}]   all_reduce dtype={:?} ({esz} B/elem) — payload is dtype-sized; \
             compare per-op across dtypes at equal element count",
            dtype
        );
        eprintln!("[{label}]   payload      iters   median/op     link GB/s (2·bytes/t)");
    }
    let sizes = [
        256usize, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304,
    ];
    let mut last_per = 0.0f64;
    let mut last_bytes = 0.0f64;
    for &elems in &sizes {
        let bytes = elems * esz;
        // identical iteration schedule on both ranks keeps them in lockstep
        let iters = (2_000_000 / elems).clamp(5, 300);
        group.barrier().unwrap();
        let t = Instant::now();
        if dtype == DType::F32 {
            // native f32 fast path (the production gradient/activation path)
            let mut data = vec![1.0f32; elems];
            for _ in 0..iters {
                group.all_reduce(&mut data, ReduceKind::Sum).unwrap();
            }
        } else {
            // native-dtype wire: fp16/bf16 move `esz`-sized elements
            let mut data = vec![0u8; bytes];
            for _ in 0..iters {
                group
                    .all_reduce_typed(&mut data, dtype, ReduceKind::Sum)
                    .unwrap();
            }
        }
        let per = t.elapsed().as_secs_f64() / iters as f64;
        last_per = per;
        last_bytes = bytes as f64;
        if leader {
            // 2-rank ring moves ~2·bytes across the wire per op (L each way).
            let gbps = 2.0 * bytes as f64 / per / 1e9;
            let sz = if bytes >= 1 << 20 {
                format!("{} MiB", bytes >> 20)
            } else {
                format!("{} KiB", bytes >> 10)
            };
            eprintln!(
                "[{label}]   {sz:>8}   {iters:>6}   {:>8.3} ms   {gbps:>6.2}",
                per * 1e3
            );
        }
    }

    if leader {
        // Fit the link from the largest message (β = effective bandwidth) and
        // the barrier (α = latency floor), then feed it to the topology
        // planner — closing the loop from measurement to decision.
        let beta = 2.0 * last_bytes / last_per; // bytes/s actually on the wire
        let link = Link {
            bandwidth_bytes_per_s: beta,
            latency_s: alpha_s,
        };
        let r = 15e12f64; // reference sustained GPU throughput (FLOP/s)
        eprintln!(
            "[{label}] measured link: α={:.2} ms, β={:.1} MB/s  →  machine ratio ≈ {:.1e} FLOP/byte",
            alpha_s * 1e3,
            beta / 1e6,
            r / beta
        );
        let w = planner::Workload {
            params: 600_000_000,
            tokens_per_step: 65_536,
            d_model: 1024,
            n_layers: 28,
            bytes_per_param: 16,
        };
        let gpu = planner::Device {
            flops_per_s: r,
            mem_bytes: 16u64 << 30,
        };
        let plan = planner::recommend(&[gpu, gpu], link, w);
        eprintln!(
            "[{label}] planner (0.6B model, 64k tok/step, 2× this link): {:?} {:.2}× — {}",
            plan.strategy, plan.speedup, plan.rationale
        );
    }
}

/// Resolve the `DEVICE` env value to a [`Device`]. `auto` picks this node's
/// fastest available accelerator (so a heterogeneous mesh needs zero per-node
/// config); otherwise the spec is parsed and checked. Refuses to run if the
/// requested backend isn't available (so a GPU run that silently fell back to
/// CPU can't masquerade as a pass).
fn resolve_device(spec: &str, label: &str) -> Device {
    if spec.eq_ignore_ascii_case("auto") {
        let d = fastest_device();
        eprintln!(
            "[{label}] DEVICE=auto → {} ({})",
            device_label(d),
            full_name(d)
        );
        return d; // fastest_device() only returns available devices
    }
    // A single device is strict (a requested GPU that's missing aborts, so a
    // run can't silently fall back to CPU and masquerade as a GPU pass).
    if let Ok(device) = parse_device(spec) {
        if !is_available(device) {
            eprintln!(
                "[{label}] DEVICE={spec} ({device:?}) is not available on this host \
                 (backend not compiled in or no hardware). Aborting so the result is honest."
            );
            std::process::exit(2);
        }
        return device;
    }
    // A list like `cuda,cpu` is a placement preference — take the first
    // available; the per-graph policy (see `parse_device_policy`) does the
    // op-feasibility-aware version.
    if let Ok(list) = parse_device_list(spec)
        && let Some(d) = list.into_iter().find(|&d| is_available(d))
    {
        return d;
    }
    eprintln!("[{label}] DEVICE={spec:?} unusable here; falling back to fastest");
    fastest_device()
}

/// `MODE=multidev`: split one logical forward into regions and place each on a
/// different backend (e.g. `REGIONS=cpu,metal,cpu`), passing activations
/// between regions. Verifies against a single-device reference. This is
/// per-region heterogeneous execution of one model on one node — the
/// region boundary is where the cross-device (host) transfer happens, so a
/// region can be as fine as a single op.
///
/// (rlx has no in-IR per-op device tag and `RLX_DEVICE_CHAIN` is a *whole-graph*
/// fallback chain, so this graph-partition + per-region placement is the
/// buildable form; a no-host-transfer version would need device-tagged nodes
/// and a multi-device executor.)
fn run_multidev(group: &Arc<ProcessGroup>, label: &str, spec: &str) -> bool {
    let _ = group;
    let (batch, d) = (2usize, 8usize);

    // Region → device. Default: cpu → fastest-accelerator → cpu.
    let region_devs: Vec<Device> = if spec.trim().is_empty() {
        vec![Device::Cpu, fastest_device(), Device::Cpu]
    } else {
        spec.split(',')
            .filter_map(|s| {
                let want = parse_device(s.trim()).ok()?;
                if is_available(want) {
                    Some(want)
                } else {
                    eprintln!(
                        "[{label}] region device '{}' unavailable here → {}",
                        s.trim(),
                        device_label(fastest_device())
                    );
                    Some(fastest_device())
                }
            })
            .collect()
    };
    let k = region_devs.len().max(1);

    let x: Vec<f32> = (0..batch * d).map(|i| (i as f32 * 0.1).sin()).collect();
    // Single-device reference: the whole chain end to end.
    let mut href = x.clone();
    for s in 0..k {
        href = matmul_host(&href, &materialize_weights(s as u64, d * d), batch, d, d);
    }

    // Per-region execution: each region's subgraph on its assigned backend; the
    // activation transfers host-side at the region boundary.
    let mut h = x.clone();
    let mut placement = Vec::with_capacity(k);
    for (s, &dev) in region_devs.iter().enumerate() {
        let mut c = Session::new(dev).compile(stage_graph(batch, d));
        c.set_param("W", &materialize_weights(s as u64, d * d));
        h = c.run(&[("h", h.as_slice())]).into_iter().next().unwrap();
        placement.push(device_label(dev));
    }

    let err = max_abs_err(&h, &href);
    let pass = err < 1e-4;
    eprintln!(
        "[{label}] MULTIDEV: {k} regions placed [{}] in one forward → output max_err={err:.2e} vs single-device ref  {}",
        placement.join(" → "),
        if pass { "PASS ✓" } else { "FAIL ✗" }
    );
    pass
}

/// Parse a device spec into a **placement policy** + a hint. `auto` allows any
/// backend (cost model picks the fastest); a single device forces it; a
/// comma list `cuda,cpu` is an allow-list in preference order (GPU-first, CPU
/// fallback). The policy is intersected with per-graph op feasibility, so an
/// unsupported op transparently routes to the next allowed backend.
fn parse_device_policy(spec: &str) -> (DevicePolicy, Option<Device>) {
    let s = spec.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("auto") {
        return (DevicePolicy::all(), None);
    }
    match parse_device_list(s) {
        Ok(list) if !list.is_empty() => {
            // Allow-list + preference order; no hard hint, so the policy picks
            // the most-preferred *available* backend and falls back down the
            // list (e.g. cuda→cpu) instead of erroring when the top choice is
            // missing.
            (DevicePolicy::only(list.clone()).with_prefer(list), None)
        }
        _ => (DevicePolicy::all(), None),
    }
}

/// `MODE=placement`: for one stage graph, report each backend's feasibility
/// (available / supports-the-graph / blocker), execute the SAME graph on every
/// backend that can run it (parity-checked), and show what the `DEVICE` policy
/// resolves to. This is graph-execution specialization made visible — the same
/// IR runs on cpu / metal / cuda / … on whatever node it lands.
fn run_placement(group: &Arc<ProcessGroup>, label: &str, spec: &str) {
    let _ = group;
    let (batch, d) = (2usize, 8usize);
    let (policy, hint) = parse_device_policy(spec);
    let gd = GraphDevices::with_policy(stage_graph(batch, d), policy);
    let report = gd.report();

    eprintln!("[{label}] PLACEMENT (policy '{spec}') — per-backend feasibility:",);
    for c in &report {
        let mark = if c.recommended { "★" } else { " " };
        let blocker = c
            .blocker
            .as_deref()
            .map(|b| format!("  ⟂ {b}"))
            .unwrap_or_default();
        eprintln!(
            "[{label}]   {mark} {:<8} available={:<5} supports_graph={}{}",
            c.label, c.available, c.supports_graph, blocker
        );
    }

    // Run the SAME graph on every available+supported backend; parity vs host.
    let w = materialize_weights(3, d * d);
    let x: Vec<f32> = (0..batch * d).map(|i| (i as f32 * 0.1).sin()).collect();
    let reference = matmul_host(&x, &w, batch, d, d);
    let mut ran = 0usize;
    for c in &report {
        if c.available && c.supports_graph {
            let mut compiled = Session::new(c.device).compile(stage_graph(batch, d));
            compiled.set_param("W", &w);
            let out = compiled
                .run(&[("h", x.as_slice())])
                .into_iter()
                .next()
                .unwrap();
            let err = max_abs_err(&out, &reference);
            eprintln!(
                "[{label}]   ↳ executed on {:<8} max_err={err:.2e}  {}",
                c.label,
                if err < 1e-4 { "✓" } else { "✗" }
            );
            ran += 1;
        }
    }
    let resolved = gd.resolve(hint).unwrap_or_else(|_| {
        // Fall back *within* the policy (first feasible), not to the global fastest.
        report
            .iter()
            .find(|c| c.available && c.supports_graph)
            .map(|c| c.device)
            .unwrap_or(Device::Cpu)
    });
    eprintln!(
        "[{label}] policy '{spec}' resolved to {} (graph ran on {ran} backend(s))",
        device_label(resolved)
    );
}

/// Every `Device` the runtime knows, for the local-capability probe.
const ALL_DEVICES: [Device; 11] = [
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Ane,
    Device::Cuda,
    Device::Rocm,
    Device::OneApi,
    Device::Gpu,
    Device::Vulkan,
    Device::Tpu,
    Device::Hexagon,
];

/// Print which backends this node can actually run — the per-node slice of a
/// heterogeneous mesh's hardware inventory.
fn print_inventory(label: &str) {
    let avail: Vec<&str> = ALL_DEVICES
        .iter()
        .filter(|&&d| is_available(d))
        .map(|&d| device_label(d))
        .collect();
    eprintln!("[{label}] local backends: [{}]", avail.join(", "));
}

/// Stable small code per device so a rank's chosen backend survives the f32
/// all-gather used to assemble the mesh topology.
fn device_code(d: Device) -> f32 {
    ALL_DEVICES.iter().position(|&x| x == d).unwrap_or(99) as f32
}
fn code_label(code: f32) -> &'static str {
    ALL_DEVICES
        .get(code as usize)
        .map(|&d| device_label(d))
        .unwrap_or("?")
}

/// Measure this node's actual matmul throughput (GFLOP/s) on `device`, so the
/// planner can reason about a heterogeneous mesh from real numbers rather than
/// assuming uniform hardware.
fn device_throughput(device: Device) -> f64 {
    let n = 1024usize;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 97) as f32) * 0.01).collect();
    let mut g = Graph::new("gflops_probe");
    let ain = g.input("a", Shape::new(&[n, n], DType::F32));
    let bp = g.param("b", Shape::new(&[n, n], DType::F32));
    let mm = g.matmul(ain, bp, Shape::new(&[n, n], DType::F32));
    g.set_outputs(vec![mm]);
    let mut c = Session::new(device).compile(g);
    c.set_param("b", &a);
    let _ = c.run(&[("a", a.as_slice())]); // warm up (compile/upload)
    let iters = 5;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = c.run(&[("a", a.as_slice())]);
    }
    let per = t.elapsed().as_secs_f64() / iters as f64;
    2.0 * (n as f64).powi(3) / per / 1e9
}

/// `MODE=topology`: each rank measures its own device throughput and the
/// mesh exchanges (device, GFLOP/s) so the leader prints the heterogeneous
/// topology and a capacity-weighted, link-aware plan. This is the flexibility
/// payoff — CPU, GPU and accelerators of different speeds sharing one job,
/// with work split proportional to each node's measured capability.
fn run_topology(group: &Arc<ProcessGroup>, label: &str, device: Device) {
    let gflops = device_throughput(device);
    eprintln!(
        "[{label}] device {} measured ≈{:.0} GFLOP/s (fp32 1024³ matmul)",
        device_label(device),
        gflops
    );

    // Exchange (device-code, GFLOP/s) with every rank.
    let mine = vec![device_code(device), gflops as f32];
    let all = group.all_gather(&mine).expect("topology all_gather");

    // Measure the link too (α from barriers, β from one ~1 MiB all-reduce).
    for _ in 0..10 {
        group.barrier().unwrap();
    }
    let mut bar = Vec::new();
    for _ in 0..50 {
        let t = Instant::now();
        group.barrier().unwrap();
        bar.push(t.elapsed().as_secs_f64());
    }
    let alpha_s = median(bar);
    let elems = 262_144usize; // 1 MiB
    let mut buf = vec![1.0f32; elems];
    group.barrier().unwrap();
    let t = Instant::now();
    for _ in 0..5 {
        group.all_reduce(&mut buf, ReduceKind::Sum).unwrap();
    }
    let per = t.elapsed().as_secs_f64() / 5.0;
    let beta = 2.0 * (elems * 4) as f64 / per;

    if group.is_leader() {
        let n = group.world_size() as usize;
        let mut devices = Vec::with_capacity(n);
        eprintln!("[{label}] ── heterogeneous mesh topology ──");
        for r in 0..n {
            let dl = code_label(all[r * 2]);
            let gf = all[r * 2 + 1] as f64;
            eprintln!("[{label}]   rank {r}: {dl:<7} ≈{gf:>6.0} GFLOP/s");
            devices.push(planner::Device {
                flops_per_s: gf * 1e9,
                mem_bytes: 8u64 << 30,
            });
        }
        let weights = planner::capacity_weights(&devices);
        let wpct: Vec<String> = weights
            .iter()
            .map(|w| format!("{:.0}%", w * 100.0))
            .collect();
        eprintln!(
            "[{label}]   link: α={:.2} ms, β={:.1} MB/s  |  capacity-weighted split: [{}]",
            alpha_s * 1e3,
            beta / 1e6,
            wpct.join(", ")
        );
        let link = Link {
            bandwidth_bytes_per_s: beta,
            latency_s: alpha_s,
        };
        let w = planner::Workload {
            params: 600_000_000,
            tokens_per_step: 65_536,
            d_model: 1024,
            n_layers: 28,
            bytes_per_param: 16,
        };
        let plan = planner::recommend(&devices, link, w);
        eprintln!(
            "[{label}]   plan (0.6B, 64k tok/step): {:?} {:.2}× — {}",
            plan.strategy, plan.speedup, plan.rationale
        );
    }
}
