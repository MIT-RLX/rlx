// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Desktop coordinator for the iOS / Android node demos.
//!
//! Ships a stage to each phone rank, streams activations around the ring, and
//! checks what comes back against a CPU reference — so "it connected" and "it
//! computed the right thing" are separate, visible results.
//!
//! ```text
//!   coordinator (rank 0) → phone (rank 1) → … → phone (rank N-1) → coordinator
//! ```
//!
//! Each worker runs the *same* stage, so with two phones the activation is
//! transformed twice. That is what makes the demo a pipeline rather than a
//! round trip.
//!
//! # Running
//!
//! Zero-config on a shared LAN (the phone finds this process by UDP):
//!
//! ```sh
//! cargo run -p rlx-ffi --example node_coordinator -- --world 2
//! ```
//!
//! Explicit peers — use this when the phone cannot receive UDP broadcast
//! (iOS without the multicast entitlement, guest Wi-Fi, most hotspots):
//!
//! ```sh
//! cargo run -p rlx-ffi --example node_coordinator -- \
//!     --world 2 --peers 192.168.1.10:29500,192.168.1.11:29500
//! ```
//!
//! The first peer is this machine; the rest are the phones, in rank order.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::dist::node::{NodeConfig, collect_caps, mean_reduce, serve_trainer_here};
use rlx_runtime::dist::{
    self, DataRef, StageSpec, TrainSpec, WeightRef, recv_activation, send_activation, ship_stage,
    ship_train,
};

/// Activation width. Small enough to eyeball, wide enough to be a real vector.
const N: usize = 64;

/// The stage every worker runs: `gelu(x * 3 + 1)`.
///
/// Deliberately parameter-free. A real deployment ships weight *URIs* that each
/// worker resolves node-locally (GGUF / safetensors — weights never cross the
/// wire), but that needs the model staged on the handset. Constants travel
/// inside the graph, so this demo runs on a phone with nothing installed.
fn stage_graph() -> Graph {
    let mut g = Graph::new("mobile_demo_stage");
    let shape = Shape::new(&[1, N], DType::F32);
    let x = g.input("x", shape.clone());
    // Scalar-shaped constants, broadcast by the binary op. `Op::Constant`
    // carries FULL data (`numel * 4` bytes) — a scalar literal given a wide
    // shape fills element 0 and leaves the rest zero.
    let three = rlx_ir::rf::const_f32(&mut g, 3.0, rlx_ir::rf::scalar_f32());
    let one = rlx_ir::rf::const_f32(&mut g, 1.0, rlx_ir::rf::scalar_f32());
    let scaled = g.binary(BinaryOp::Mul, x, three, shape.clone());
    let shifted = g.binary(BinaryOp::Add, scaled, one, shape.clone());
    let y = g.activation(Activation::Gelu, shifted, shape);
    g.set_outputs(vec![y]);
    g
}

/// Host reference for one worker hop, so a wrong answer is distinguishable
/// from a dropped connection.
///
/// Computed in f64: this is the oracle the device result is judged against, so
/// its own error should sit well below the f32 difference being measured.
fn reference_hop(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| {
            let t = f64::from(*v) * 3.0 + 1.0;
            // Exact GELU (erf form), matching the CPU kernel's definition.
            (0.5 * t * (1.0 + erf(t / std::f64::consts::SQRT_2))) as f32
        })
        .collect()
}

/// Abramowitz–Stegun 7.1.26 (|error| < 1.5e-7) — ample for a tolerance check
/// against an f32 pipeline.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Reserve `n` loopback ports by binding and immediately releasing them. A
/// race is possible in principle; in practice the ranks bind microseconds later.
fn free_ports(n: usize) -> Result<Vec<u16>, String> {
    use std::net::{Ipv4Addr, TcpListener};
    let ls: Vec<TcpListener> = (0..n)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    ls.iter()
        .map(|l| l.local_addr().map(|a| a.port()).map_err(|e| e.to_string()))
        .collect()
}

/// Width of the demo regression problem, and how many samples each rank owns.
const TRAIN_D: usize = 3;
const TRAIN_PER_RANK: usize = 16;
const TRAIN_BATCH: usize = 4;

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn write_f32(dir: &std::path::Path, name: &str, vals: &[f32]) -> String {
    let path = dir.join(name);
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, &bytes).expect("write demo data");
    format!("file://{}", path.display())
}

/// `loss = mean((X·w - y)^2)` over this rank's shard, with its backward graph.
///
/// Data is written to `dir` and referenced by URI: weights and data are
/// resolved **on the node**, so only the spec crosses the wire. On a real
/// handset there is no shared filesystem, which is what `push_data` is for —
/// see `--train` notes in the help text.
fn train_spec(dir: &std::path::Path, shard: usize, world: usize, push_data: bool) -> TrainSpec {
    use rlx_ir::op::ReduceOp;
    let m = TRAIN_PER_RANK * world;
    let f = DType::F32;
    let mut g = Graph::new("demo_lr");
    let x = g.input("X", Shape::new(&[TRAIN_BATCH, TRAIN_D], f));
    let w = g.param("w", Shape::new(&[TRAIN_D, 1], f));
    let y = g.input("y", Shape::new(&[TRAIN_BATCH, 1], f));
    let pred = g.matmul(x, w, Shape::new(&[TRAIN_BATCH, 1], f));
    let diff = g.binary(BinaryOp::Sub, pred, y, Shape::new(&[TRAIN_BATCH, 1], f));
    let sq = g.binary(BinaryOp::Mul, diff, diff, Shape::new(&[TRAIN_BATCH, 1], f));
    let loss = g.add_node(
        rlx_ir::op::Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![sq],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);
    let bwd = rlx_autodiff::grad_with_loss(&g, &[w]);

    let w_true = seeded(TRAIN_D, 7);
    let x_data = seeded(m * TRAIN_D, 3);
    let y_data: Vec<f32> = (0..m)
        .map(|i| {
            (0..TRAIN_D)
                .map(|k| x_data[i * TRAIN_D + k] * w_true[k])
                .sum()
        })
        .collect();

    TrainSpec {
        graph: bwd,
        params: vec![WeightRef {
            name: "w".into(),
            // Identical init on every rank — data-parallel assumes replicas
            // start from the same point.
            uri: write_f32(dir, "w0.bin", &[0.0f32; TRAIN_D]),
            packed: false,
        }],
        grad_start: 1,
        loss_index: 0,
        data: vec![
            DataRef {
                input: "X".into(),
                uri: write_f32(dir, "X.bin", &x_data),
                elem: TRAIN_D,
                shard_start: shard * TRAIN_PER_RANK,
                shard_len: TRAIN_PER_RANK,
            },
            DataRef {
                input: "y".into(),
                uri: write_f32(dir, "y.bin", &y_data),
                elem: 1,
                shard_start: shard * TRAIN_PER_RANK,
                shard_len: TRAIN_PER_RANK,
            },
        ],
        seed_input: Some("d_output".into()),
        momentum: 0.0,
        lr_per_epoch: vec![0.3; 60],
        batch: TRAIN_BATCH,
        device: "auto".into(),
        grad_group: 0,
        // A handset has no shared filesystem, so the URIs above — paths on the
        // coordinator's disk — mean nothing to it. `push_data` sends the shard
        // over the wire instead and the worker rewrites its `DataRef`s to the
        // files it received. Only loopback can get away with `false`.
        push_data,
    }
}

struct Args {
    world: u32,
    peers: Vec<String>,
    steps: usize,
    train: bool,
    /// Run the worker ranks in this process on loopback, so the desktop side
    /// can be verified before a handset is involved.
    self_test: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut world = 2u32;
    let mut peers = Vec::new();
    let mut steps = 4usize;
    let mut self_test = false;
    let mut train = false;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String, String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--world" => {
                world = need(i)?.parse().map_err(|_| "--world must be an integer")?;
                i += 2;
            }
            "--peers" => {
                peers = need(i)?.split(',').map(|s| s.trim().to_string()).collect();
                i += 2;
            }
            "--steps" => {
                steps = need(i)?.parse().map_err(|_| "--steps must be an integer")?;
                i += 2;
            }
            "--train" => {
                train = true;
                i += 1;
            }
            "--self-test" => {
                self_test = true;
                i += 1;
            }
            "-h" | "--help" => {
                println!(
                    "node_coordinator [--world N] [--peers a:p,b:p,…] [--steps K]\n\
                     \n\
                     --world  ranks including this one (default 2 = one phone)\n\
                     --peers  explicit host:port per rank; omit for UDP discovery\n\
                     --steps  activations to stream (default 4)\n\
                     --self-test  run the workers here on loopback (no phone)\n\
                     --train      data-parallel training instead of inference"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if world < 2 {
        return Err("--world must be at least 2 (this process plus one node)".into());
    }
    // Star needs only the coordinator's own address; a mesh needs one per rank.
    if !peers.is_empty() && peers.len() != world as usize && peers.len() != 1 {
        return Err(format!(
            "--peers has {} entries; expected {world} (one per rank) or 1 (this \
             coordinator's listen address, for a star)",
            peers.len()
        ));
    }
    Ok(Args {
        world,
        peers,
        steps,
        train,
        self_test,
    })
}

/// Data-parallel training across the mesh: ship each worker its own shard,
/// then train this rank's shard in lockstep with them.
///
/// Every rank must reduce the same number of times — the gradient all-reduce
/// is a barrier — so the coordinator trains too rather than just waiting.
fn run_training(
    group: &rlx_runtime::dist::node::ProcessGroup,
    caps: &[rlx_runtime::dist::node::NodeCaps],
    args: &Args,
    workers: Vec<std::thread::JoinHandle<Option<Vec<f32>>>>,
) {
    let dir = std::env::temp_dir().join(format!("rlx_coord_train_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("error: temp dir: {e}");
        std::process::exit(1);
    }
    let world = args.world as usize;
    // Loopback shares this process's filesystem; a real node does not.
    let push = !args.self_test;
    for c in caps {
        // Ask before shipping. A rank that cannot train never reaches the
        // gradient barrier, and the coordinator would wait on it forever
        // rather than report anything.
        if let Err(e) = c.can_train() {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        let spec = train_spec(&dir, c.rank as usize, world, push);
        if let Err(e) = ship_train(group, c.rank, &spec) {
            eprintln!("error: ship_train(rank {}): {e}", c.rank);
            std::process::exit(1);
        }
        if push && let Err(e) = dist::push_shards(group, c.rank, &spec, dist::uri_resolver) {
            eprintln!("error: push_shards(rank {}): {e}", c.rank);
            std::process::exit(1);
        }
    }
    println!(
        "training job shipped to {} worker(s){}\n",
        caps.len(),
        if push {
            " (data pushed over the wire)"
        } else {
            ""
        }
    );

    let spec = train_spec(&dir, 0, world, false);
    let out = dist::run_train(
        &spec,
        args.world,
        dist::uri_resolver,
        |flat| mean_reduce(group, flat),
        false,
    );
    let (metrics, params) = match out {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: run_train: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "rank 0 trained on {} — {} samples, loss {:.5} -> {:.5} (compute {:.2}s, comm {:.2}s)",
        metrics.device.name(),
        metrics.samples,
        metrics.first_loss,
        metrics.last_loss,
        metrics.compute_s,
        metrics.comm_s
    );
    let mine = params
        .into_iter()
        .next()
        .map(|(_, v)| v)
        .unwrap_or_default();
    println!("rank 0 weights: {:?}", round4(&mine));

    // Loopback self-test: the worker threads hand their weights back, so the
    // run can check the replicas actually agree. Against a real handset there
    // is no return path for parameters — read its status line instead.
    let mut disagreed = false;
    for (i, w) in workers.into_iter().enumerate() {
        match w.join() {
            Ok(Some(theirs)) => {
                let err = mine
                    .iter()
                    .zip(&theirs)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "rank {} weights: {:?}  max|Δ| = {err:.3e}",
                    i + 1,
                    round4(&theirs)
                );
                // Each rank trains a different shard. Identical weights are
                // only possible if the gradient all-reduce ran; without it they
                // drift apart while each still converges on its own data.
                if err != 0.0 {
                    disagreed = true;
                }
            }
            Ok(None) => {}
            Err(_) => {
                eprintln!("error: a worker thread panicked");
                std::process::exit(1);
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();

    if metrics.last_loss >= metrics.first_loss {
        eprintln!("\nFAIL: loss did not fall");
        std::process::exit(1);
    }
    if disagreed {
        eprintln!("\nFAIL: replicas disagree — the gradient all-reduce did not happen");
        std::process::exit(1);
    }
    println!("\nPASS — the mesh trained together and the replicas agree.");
}

fn round4(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| (x * 1e4).round() / 1e4).collect()
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    // Loopback self-test: bind the whole world here and run the worker ranks on
    // threads. Same node code path the phone runs — only the transport is local.
    let mut args = args;
    let mut workers = Vec::new();
    if args.self_test {
        let ports = match free_ports(args.world as usize) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        args.peers = ports.iter().map(|p| format!("127.0.0.1:{p}")).collect();
        println!("self-test: {} loopback rank(s)", args.world - 1);
        // A loopback mesh needs every rank's address, unlike the star path.
        assert_eq!(args.peers.len(), args.world as usize);
        for rank in 1..args.world {
            let peers = args.peers.clone();
            let training = args.train;
            workers.push(std::thread::spawn(move || {
                let cfg = NodeConfig::new(rank, peers.len() as u32)
                    .mesh()
                    .device("cpu")
                    .peers(peers)
                    .expect("peers");
                let group = cfg.connect().expect("worker connect");
                if training {
                    let r = serve_trainer_here(&group, dist::uri_resolver, false)
                        .expect("worker training");
                    // Hand the trained weights back so the run can check that
                    // the replicas agree — the whole point of the all-reduce.
                    return r.params.into_iter().next().map(|(_, v)| v);
                }
                // Ends when the coordinator drops the link.
                let _ = rlx_runtime::dist::node::serve_worker(&group, |_| Vec::new());
                None
            }));
        }
    }

    // Star: workers dial in. A handset on Wi-Fi usually cannot accept inbound
    // connections, so a full mesh would not form. Loopback self-test uses a
    // plain mesh instead — every rank is reachable here.
    let base = NodeConfig::new(0, args.world).device("auto");
    let base = if args.self_test {
        base.mesh()
    } else {
        base.star()
    };
    let cfg = if args.peers.is_empty() {
        println!("waiting for {} node(s) to discover us…", args.world - 1);
        base.discover(29600, 29500)
    } else {
        println!(
            "waiting for {} node(s) at {:?}…",
            args.world - 1,
            args.peers
        );
        match base.peers(args.peers.clone()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        }
    };

    let group = match cfg.connect() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: connect failed: {e}");
            std::process::exit(1);
        }
    };
    println!("mesh up: world {}", args.world);

    // Who joined, and what can they actually run?
    let caps = match collect_caps(&group) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    for c in &caps {
        println!(
            "  rank {} — {} [{}] devices: {}",
            c.rank,
            c.platform,
            match &c.role {
                rlx_runtime::dist::node::NodeRole::Worker => "worker".to_string(),
                rlx_runtime::dist::node::NodeRole::FixedFunction(f) =>
                    format!("fixed-function: {}", f.stage_id),
            },
            c.devices.join(", ")
        );
    }

    if args.train {
        run_training(&group, &caps, &args, workers);
        return;
    }

    // Ship the same stage to every worker.
    let spec = StageSpec::new(stage_graph(), "x", "y", "auto");
    for c in &caps {
        if let Err(e) = c.accepts(&spec) {
            eprintln!("error: rank {} cannot run this stage: {e}", c.rank);
            std::process::exit(1);
        }
        if let Err(e) = ship_stage(&group, c.rank, &spec) {
            eprintln!("error: ship_stage(rank {}): {e}", c.rank);
            std::process::exit(1);
        }
    }
    println!("stage shipped to {} worker(s)\n", caps.len());

    let hops = args.world as usize - 1;
    let last = args.world - 1;
    let mut worst: f32 = 0.0;

    for step in 0..args.steps {
        // A different input each step, so a cached or echoed reply is visible.
        let x: Vec<f32> = (0..N)
            .map(|i| ((i + step * N) as f32 * 0.07).sin())
            .collect();

        if let Err(e) = send_activation(&group, 1, &x) {
            eprintln!("error: send: {e}");
            std::process::exit(1);
        }
        let got = match recv_activation(&group, last) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: recv: {e}");
                std::process::exit(1);
            }
        };

        // The same stage runs once per worker, so apply the reference `hops` times.
        let mut want = x.clone();
        for _ in 0..hops {
            want = reference_hop(&want);
        }

        if got.len() != want.len() {
            eprintln!(
                "error: step {step}: got {} elems, want {}",
                got.len(),
                want.len()
            );
            std::process::exit(1);
        }
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(err);
        println!(
            "step {step}: max|got-ref| = {err:.3e}   first 4: {:?}",
            got.iter()
                .take(4)
                .map(|v| (v * 1e4).round() / 1e4)
                .collect::<Vec<_>>()
        );
    }

    // The reference erf is an approximation (~1.5e-7 absolute), and the value
    // is chained once per hop, so scale the bound with the pipeline depth.
    let tol = 2e-5 * hops as f32;
    println!("\nworst error across {} step(s): {worst:.3e}", args.steps);
    if worst > tol {
        eprintln!("FAIL: exceeds tolerance {tol:.1e}");
        std::process::exit(1);
    }
    // Drop the group so the loopback workers' `recv` fails and they wind down.
    // Deliberately NOT joined: a rank parked in `recv` has no deadline, so
    // joining here would hang the demo after it had already produced its
    // verdict. Process exit reaps them.
    drop(group);
    drop(workers);
    println!("PASS — the phone(s) computed the stage correctly.");
}
