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

//! The simplest way to run a distributed job: one TOML file + one launch per
//! node. Reads `rlx-dist.toml` (see the sibling file), builds a [`Node`] with
//! the configured discovery (static peers, mDNS, or rendezvous), joins the
//! network, and runs either data-parallel **training** or tensor-parallel
//! **inference**.
//!
//! ```text
//!   # same file on every machine; each node gets its own RANK
//!   RANK=0 dist_job rlx-dist.toml
//!   RANK=1 dist_job rlx-dist.toml
//! ```

use rlx_driver::{Node, ProcessGroup, ReduceKind, Topology};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, device_label, fastest_device, is_available, parse_device};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
struct Config {
    job: Job,
    net: Net,
}
#[derive(Deserialize)]
struct Job {
    task: String,
    world: u32,
    #[serde(default = "default_steps")]
    steps: usize,
}
#[derive(Deserialize)]
struct Net {
    #[serde(default = "default_device")]
    device: String,
    #[serde(default = "default_topology")]
    topology: String,
    discovery: String,
    #[serde(default)]
    peers: Vec<String>,
    #[serde(default)]
    coordinator: Option<String>,
    #[serde(default = "default_data_port")]
    data_port: u16,
    #[serde(default = "default_disc_port")]
    disc_port: u16,
}
fn default_steps() -> usize {
    200
}
fn default_device() -> String {
    "auto".into()
}
fn default_topology() -> String {
    "mesh".into()
}
fn default_data_port() -> u16 {
    29500
}
fn default_disc_port() -> u16 {
    29600
}

fn main() {
    // Config path from argv, else ./rlx-dist.toml. RANK from the environment.
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "rlx-dist.toml".into());
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(2);
    });
    let cfg: Config = toml::from_str(&text).unwrap_or_else(|e| {
        eprintln!("bad config {path}: {e}");
        std::process::exit(2);
    });
    let rank: u32 = std::env::var("RANK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let world = cfg.job.world;
    let device = resolve_device(&cfg.net.device);

    // ── 1. Build the node from config ────────────────────────────────────────
    // Discovery picks how peers find each other; mesh vs. star is the wire shape.
    let node = build_node(rank, world, &cfg.net);
    eprintln!(
        "[rank {rank}/{world}] task={} discovery={} topology={} device={}",
        cfg.job.task,
        cfg.net.discovery,
        node_topology(&cfg.net),
        device_label(device)
    );

    // ── 2. Join the network — one call ───────────────────────────────────────
    let group = node.connect().unwrap_or_else(|e| {
        eprintln!("[rank {rank}] connect failed: {e}");
        std::process::exit(1);
    });
    group.barrier().expect("initial barrier");
    eprintln!("[rank {rank}] joined the {world}-node network ✓");

    // ── 3. Run the job ───────────────────────────────────────────────────────
    match cfg.job.task.as_str() {
        "train" => train(&group, device, cfg.job.steps),
        "infer" => infer(&group, device),
        other => {
            eprintln!("unknown task '{other}' (use \"train\" or \"infer\")");
            std::process::exit(2);
        }
    }
    group.barrier().ok();
}

fn node_topology(net: &Net) -> &str {
    // mDNS and rendezvous are coordinator-centric → always star.
    if net.discovery == "static" {
        &net.topology
    } else {
        "star"
    }
}

/// Catch a placeholder address the user forgot to edit (e.g. the docs'
/// `AAA.BBB.CCC.DDD`): dotted groups that are all runs of one ASCII-uppercase
/// letter. Real IPs (digits) and hostnames (lowercase) never match.
fn is_placeholder(addr: &str) -> bool {
    let host = addr.rsplit_once(':').map_or(addr, |(h, _)| h);
    let groups: Vec<&str> = host.split('.').collect();
    // Each group is a run of one repeated ASCII-uppercase letter (AAA, BBB, …).
    groups.len() >= 2
        && groups.iter().all(|g| {
            let mut cs = g.chars();
            matches!(cs.next(), Some(first) if first.is_ascii_uppercase() && cs.all(|c| c == first))
        })
}

fn config_error(msg: String) -> ! {
    eprintln!("rlx-dist.toml: {msg}");
    std::process::exit(2);
}

fn build_node(rank: u32, world: u32, net: &Net) -> Node {
    match net.discovery.as_str() {
        "static" => {
            if net.peers.is_empty() {
                config_error(
                    "discovery = \"static\" requires [net] peers = [\"host:port\", …]".into(),
                );
            }
            if let Some(p) = net.peers.iter().find(|p| is_placeholder(p)) {
                config_error(format!(
                    "[net] peers still has the placeholder '{p}' — set real host:port values (one per rank)"
                ));
            }
            let topo = if net.topology == "star" {
                Topology::Star
            } else {
                Topology::Mesh
            };
            Node::new(rank, world)
                .topology(topo)
                .peers(net.peers.iter().map(String::as_str))
                .unwrap_or_else(|e| config_error(format!("[net] peers has a bad address: {e}")))
        }
        // Zero-config LAN discovery: coordinator advertises, workers browse.
        "mdns" => Node::new(rank, world)
            .topology(Topology::Star)
            .discover(net.disc_port, net.data_port),
        // Cross-network: workers unicast a query to `coordinator` and learn the
        // data port (LAN IP / host.docker.internal / Tailscale MagicDNS name).
        "rendezvous" => {
            let mut n = Node::new(rank, world)
                .topology(Topology::Star)
                .discover(net.disc_port, net.data_port);
            if rank != 0 {
                let host = net.coordinator.clone().unwrap_or_else(|| {
                    config_error(
                        "discovery = \"rendezvous\" requires [net] coordinator = \"<host>\"".into(),
                    )
                });
                if is_placeholder(&host) {
                    config_error(format!(
                        "[net] coordinator is still the placeholder '{host}' — set your \
                         coordinator's real host (LAN IP, host.docker.internal, or a Tailscale name)"
                    ));
                }
                n = n.discover_via(host);
            }
            n
        }
        other => {
            eprintln!("unknown discovery '{other}' (static | mdns | rendezvous)");
            std::process::exit(2);
        }
    }
}

fn resolve_device(spec: &str) -> Device {
    if spec.eq_ignore_ascii_case("auto") {
        return fastest_device();
    }
    match parse_device(spec) {
        Ok(d) if is_available(d) => d,
        _ => fastest_device(),
    }
}

// ── the model: a tiny linear layer y = x @ w ────────────────────────────────
const D: usize = 4; // features
const N: usize = 8; // samples per node

/// `y = x @ w`, x `[N, D]` input, w `[D, 1]` param. Compiled once, run each step.
fn linear_graph() -> Graph {
    let mut g = Graph::new("linear");
    let x = g.input("x", Shape::new(&[N, D], DType::F32));
    let w = g.param("w", Shape::new(&[D, 1], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[N, 1], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// This node's data shard: distinct `x` per rank, targets from a shared true `w`.
fn data_shard(rank: u32) -> (Vec<f32>, Vec<f32>) {
    let w_true = [0.5f32, -0.3, 0.8, 0.1];
    let mut x = vec![0f32; N * D];
    let mut y = vec![0f32; N];
    for i in 0..N {
        let seed = (rank as usize * N + i) as f32;
        let mut acc = 0f32;
        for j in 0..D {
            let v = ((seed * 0.7 + j as f32).sin()) * 0.5;
            x[i * D + j] = v;
            acc += v * w_true[j];
        }
        y[i] = acc; // noiseless target
    }
    (x, y)
}

/// Data-parallel SGD: each node forwards on its shard, computes the local MSE
/// gradient, and the gradients are **averaged across all nodes** each step
/// (`all_reduce(Mean)`) — the whole of distributed data-parallel training.
fn train(group: &Arc<ProcessGroup>, device: Device, steps: usize) {
    let (rank, world) = (group.rank(), group.world_size());
    let (x, target) = data_shard(rank);
    let mut compiled = Session::new(device).compile(linear_graph());
    let mut w = vec![0f32; D];
    let lr = 0.2f32;

    for _ in 0..steps {
        compiled.set_param("w", &w);
        let y = compiled
            .run(&[("x", x.as_slice())])
            .into_iter()
            .next()
            .unwrap();
        // grad of (1/N)·Σ(y-t)²  w.r.t. w  =  (2/N)·Xᵀ(y - t)
        let mut grad = vec![0f32; D];
        for i in 0..N {
            let e = y[i] - target[i];
            for j in 0..D {
                grad[j] += 2.0 * e * x[i * D + j] / N as f32;
            }
        }
        group
            .all_reduce(&mut grad, ReduceKind::Mean)
            .expect("all_reduce grad");
        for j in 0..D {
            w[j] -= lr * grad[j];
        }
    }

    // Final loss on the local shard (same globally-averaged w on every node).
    compiled.set_param("w", &w);
    let y = compiled
        .run(&[("x", x.as_slice())])
        .into_iter()
        .next()
        .unwrap();
    let loss: f32 = (0..N).map(|i| (y[i] - target[i]).powi(2)).sum::<f32>() / N as f32;
    if rank == 0 {
        eprintln!(
            "[rank 0] TRAINED {steps} steps across {world} node(s) on {} → w={:?}, loss={loss:.3e}  {}",
            device_label(device),
            w.iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>(),
            if loss < 1e-3 { "✓" } else { "" }
        );
    }
}

/// Tensor-parallel inference: the contraction dim `D` is sharded across nodes,
/// each computes a partial `x_shard @ w_shard`, and `all_reduce(Sum)` combines
/// them into the full result — the classic TP collective.
fn infer(group: &Arc<ProcessGroup>, device: Device) {
    let (rank, world) = (group.rank() as usize, group.world_size() as usize);
    assert!(
        D.is_multiple_of(world),
        "D={D} must divide by world={world} for this demo"
    );
    let dr = D / world; // this node's slice of the contraction dim

    // Shared full inputs/weights (every node agrees); each keeps only its slice.
    let (x_full, _) = data_shard(0);
    let w_full = [0.5f32, -0.3, 0.8, 0.1];
    let mut x_shard = vec![0f32; N * dr];
    for i in 0..N {
        for j in 0..dr {
            x_shard[i * dr + j] = x_full[i * D + (rank * dr + j)];
        }
    }
    let w_shard: Vec<f32> = (0..dr).map(|j| w_full[rank * dr + j]).collect();

    // y_partial = x_shard[N,dr] @ w_shard[dr,1]  on this node's device.
    let mut g = Graph::new("tp");
    let x = g.input("x", Shape::new(&[N, dr], DType::F32));
    let w = g.param("w", Shape::new(&[dr, 1], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[N, 1], DType::F32));
    g.set_outputs(vec![y]);
    let mut compiled = Session::new(device).compile(g);
    compiled.set_param("w", &w_shard);
    let mut y_partial = compiled
        .run(&[("x", x_shard.as_slice())])
        .into_iter()
        .next()
        .unwrap();

    // Sum the per-node partials → full y = x_full @ w_full.
    group
        .all_reduce(&mut y_partial, ReduceKind::Sum)
        .expect("all_reduce y");

    if rank == 0 {
        // Reference on rank 0 to confirm the distributed result.
        let y_ref: Vec<f32> = (0..N)
            .map(|i| (0..D).map(|j| x_full[i * D + j] * w_full[j]).sum())
            .collect();
        let err = y_partial
            .iter()
            .zip(&y_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "[rank 0] INFERRED y[{N}] tensor-parallel across {world} node(s) on {} → max_err={err:.2e}  {}",
            device_label(device),
            if err < 1e-4 { "✓" } else { "✗" }
        );
    }
}
