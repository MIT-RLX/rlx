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
//!          multidev | federated | coordinator | worker   (default both)
//!          federated = bounded-staleness federated averaging (LATE_MS makes
//!                      this rank a straggler; FED_DEADLINE_MS = patience);
//!                      DTYPE=i8 exercises the int8/edge collective in bench
//!   DIAL_OUT 1 = NAT-friendly star (workers dial the coordinator=peers[0],
//!          no inbound port) instead of the full-mesh TCP default
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
use rlx_driver::{Node, ProcessGroup, ReduceKind};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::dist; // reusable ship-graph worker/coordinator API
use rlx_runtime::{
    Device, DevicePolicy, GraphDevices, Session, device_label, fastest_device, full_name,
    is_available, parse_device, parse_device_list,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// In-graph collective group id this process registers its group under.
/// Only needs to be consistent *within* a process (the graph carries it
/// in op attrs and the kernel resolves it locally), so a constant is fine.
const GID: u64 = 0;

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
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
    // Node config (RANK/WORLD/PEERS or DISCOVER/TOPOLOGY) — one call.
    let node = Node::from_env().unwrap_or_else(|e| {
        eprintln!("[dist_node] bad node config: {e}");
        std::process::exit(2);
    });
    let (rank, world) = (node.rank(), node.world());
    let mode = env("MODE", "both");
    let steps: usize = env("STEPS", "200").parse().expect("STEPS");
    let accum: usize = env("ACCUM", "1").parse::<usize>().expect("ACCUM").max(1);
    let label = env("LABEL", &format!("rank{rank}"));
    let device = resolve_device(&env("DEVICE", "cpu"), &label);
    eprintln!(
        "[{label}] rank {rank}/{world}  device={}  mode={mode}",
        device_label(device)
    );
    print_inventory(&label);

    register(); // install in-graph collective op (idempotent)

    // Join the mesh — the builder handles mesh vs. star (DIAL_OUT), peer
    // discovery, and Arc wrapping.
    let group = node.connect().unwrap_or_else(|e| {
        eprintln!("[{label}] connect failed: {e} — is the peer up / reachable?");
        std::process::exit(1);
    });
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

    if mode == "federated" {
        ok &= run_federated(&group, &label);
        group.barrier().expect("post-federated barrier");
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

    // Distributed FFT: rank 0 scatters signal slices to the workers, each runs
    // an `Op::Fft` graph on its own device, rank 0 gathers + verifies vs a DFT
    // reference. Coordinator-centric, so it runs over a dial-out star (workers
    // dial in — the shape a Docker/QEMU node behind NAT needs).
    if mode == "fft" {
        ok &= if group.rank() == 0 {
            run_fft_coordinator(&group, &label, device)
        } else {
            run_fft_worker(&group, &label)
        };
        group.barrier().expect("post-fft barrier");
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
// The envelope, wire tags, ship/serve helpers now live in the reusable
// `rlx_runtime::dist` module — the SAME worker binary runs any model. This
// example only supplies a weight resolver (below) and the model graph.

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

/// Write a 1-tensor F32 safetensors file — the coordinator's stand-in for a
/// real model shard the worker loads via `safetensors://` (a real deployment
/// would already have the model's `.safetensors`/`.gguf` on each node).
fn write_safetensors(path: &str, name: &str, vals: &[f32]) {
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let header = format!(
        r#"{{"{name}":{{"dtype":"F32","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        vals.len(),
        data.len()
    );
    let mut buf = (header.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&data);
    std::fs::write(path, buf).expect("write safetensors");
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

/// Quantized stage: the weight is Q8_0-packed `U8`; `dequant_matmul` decodes it
/// at matmul time (native quant memory). GGUF matmul is BT — `out = h @ Wᵀ`.
fn stage_graph_q8(batch: usize, d: usize, packed_len: usize) -> Graph {
    let mut g = Graph::new("stage_q8");
    let h = g.input("h", Shape::new(&[batch, d], DType::F32));
    let w = g.param("W", Shape::new(&[packed_len], DType::U8));
    let mm = g.dequant_matmul_packed(
        h,
        w,
        rlx_ir::quant::QuantScheme::GgufQ8_0,
        Shape::new(&[batch, d], DType::F32),
    );
    g.set_outputs(vec![mm]);
    g
}

/// BT matmul reference (`out[i,j] = Σ_c a[i,c]·w[j,c]`, `w` is `[n,k]`), matching
/// the GGUF `DequantMatMul` layout used by [`stage_graph_q8`].
fn bt_matmul_host(a: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for c in 0..k {
                s += a[i * k + c] * w[j * k + c];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// Quantize `vals` to Q8_0 and write a 1-tensor GGUF the worker loads via
/// `gguf://…#name` (packed) — the coordinator's stand-in for a real GGUF shard.
fn write_gguf_q8(path: &str, name: &str, vals: &[f32]) {
    let packed = rlx_gguf::quantize(vals, rlx_gguf::GgmlType::Q8_0).expect("quantize q8_0");
    let mut w = rlx_gguf::GgufWriter::new();
    w.add_tensor_bytes(name, vec![vals.len()], rlx_gguf::GgmlType::Q8_0, packed)
        .expect("add gguf tensor");
    w.write_to_path(path).expect("write gguf");
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

    // Placement + weight format. WEIGHTS_FMT=q8 → Q8_0-packed weights run
    // through a DequantMatMul graph (native quant memory); the worker is
    // unchanged — it just runs the quantized graph it's shipped.
    let worker_dev = std::env::var("WORKER_DEVICE").unwrap_or_else(|_| "auto".into());
    let weights_dir = std::env::var("WEIGHTS_DIR").ok();
    let fmt = std::env::var("WEIGHTS_FMT").unwrap_or_else(|_| "seed".into());
    let quant = fmt == "q8";

    // Full-model reference over the SAME weights the stages use: dequantized
    // Q8_0 + BT matmul in quant mode, else f32 NN matmul.
    let mut href = x.clone();
    for s in 0..n {
        let wf = materialize_weights(s as u64, d * d);
        href = if quant {
            let packed = rlx_gguf::quantize(&wf, rlx_gguf::GgmlType::Q8_0).unwrap();
            let wdeq = rlx_gguf::dequant_q8_0(&packed, d * d).unwrap();
            bt_matmul_host(&href, &wdeq, batch, d, d)
        } else {
            matmul_host(&href, &wf, batch, d, d)
        };
    }

    // Ship each worker a self-describing StageSpec (graph + I/O names + weight
    // URI + placement); the worker fetches its weight node-local from the URI.
    //   q8                                    → gguf:// (packed, DequantMatMul)
    //   WEIGHTS_DIR + WEIGHTS_FMT=safetensors → safetensors:// (real load)
    //   WEIGHTS_DIR (default)                 → file:// (raw f32)
    //   neither                               → seed:// (materialize)
    for r in 1..n {
        let wf = materialize_weights(r as u64, d * d);
        let spec = if quant {
            let packed = rlx_gguf::quantize(&wf, rlx_gguf::GgmlType::Q8_0).unwrap();
            let dir = weights_dir
                .clone()
                .unwrap_or_else(|| std::env::temp_dir().to_string_lossy().into_owned());
            let path = format!("{dir}/stage_{r}.gguf");
            write_gguf_q8(&path, "W", &wf);
            dist::StageSpec::new(
                stage_graph_q8(batch, d, packed.len()),
                "h",
                "y",
                worker_dev.clone(),
            )
            .weight_packed("W", format!("gguf://{path}#W"))
        } else {
            let uri = match (&weights_dir, fmt.as_str()) {
                (Some(dir), "safetensors") => {
                    let path = format!("{dir}/stage_{r}.safetensors");
                    write_safetensors(&path, "W", &wf);
                    format!("safetensors://{path}#W")
                }
                (Some(dir), _) => {
                    let path = format!("{dir}/stage_{r}.f32");
                    let bytes: Vec<u8> = wf.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(&path, &bytes).expect("write weights file");
                    format!("file://{path}")
                }
                (None, _) => format!("seed://{r}?len={}", d * d),
            };
            dist::StageSpec::new(stage_graph(batch, d), "h", "y", worker_dev.clone())
                .weight("W", uri)
        };
        dist::ship_stage(group, r as u32, &spec).expect("ship stage");
    }
    eprintln!(
        "[{label}] COORDINATOR: shipped {} StageSpec(s) via rlx_runtime::dist ({}), placement='{worker_dev}'",
        n.saturating_sub(1),
        if quant {
            "Q8_0 packed → DequantMatMul"
        } else {
            fmt.as_str()
        }
    );

    // Stage 0 on the coordinator's own device (same graph kind as the workers).
    let mut h = if quant {
        let wf = materialize_weights(0, d * d);
        let packed = rlx_gguf::quantize(&wf, rlx_gguf::GgmlType::Q8_0).unwrap();
        let mut c0 = Session::new(device).compile(stage_graph_q8(batch, d, packed.len()));
        c0.set_param_typed("W", &packed, DType::U8);
        c0.run(&[("h", x.as_slice())]).into_iter().next().unwrap()
    } else {
        let mut c0 = Session::new(device).compile(stage_graph(batch, d));
        c0.set_param("W", &materialize_weights(0, d * d));
        c0.run(&[("h", x.as_slice())]).into_iter().next().unwrap()
    };

    // Drive activation through the chain (0 → 1 → … → n-1 → 0).
    if n > 1 {
        dist::send_activation(group, 1, &h).expect("send act");
        h = dist::recv_activation(group, n as u32 - 1).expect("recv final");
    }

    let err = max_abs_err(&h, &href);
    // Q8_0 is lossy, but the reference uses the SAME dequantized weights, so the
    // residual is just f32 accumulation-order rounding.
    let tol = if quant { 1e-2 } else { 1e-4 };
    let pass = err < tol;
    eprintln!(
        "[{label}] COORDINATOR: pipeline output max_err={err:.2e} vs reference (tol {tol:.0e})  {}",
        if pass { "PASS ✓" } else { "FAIL ✗" }
    );
    pass
}

/// Worker (rank r > 0): a GENERIC executor built on `rlx_runtime::dist` —
/// receives a stage, resolves its own weights (here via `resolve_weights`),
/// runs the handed activation, forwards it. The SAME code runs any model; a
/// real deployment plugs a GGUF/safetensors loader into the resolver closure.
fn run_worker(group: &Arc<ProcessGroup>, label: &str) -> bool {
    let (rank, n) = (group.rank(), group.world_size());
    // recv_stage's built-in cache handles gguf:// / safetensors:// / file://
    // (parse-once); this fallback only supplies the demo's seed:// scheme.
    let mut stage = match dist::recv_stage(group, |uri| resolve_weights(uri, label)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[{label}] WORKER rank {rank}: {e}");
            return false;
        }
    };
    let input = dist::recv_activation(group, rank - 1).expect("recv act");
    let out = stage.run(&input);
    let next = if rank + 1 < n { rank + 1 } else { 0 };
    dist::send_activation(group, next, &out).expect("forward act");
    eprintln!(
        "[{label}] WORKER rank {rank}: ran {}-node graph on {} → forwarded to rank {next}",
        stage.nodes,
        device_label(stage.device)
    );
    true
}

// ── Distributed FFT ─────────────────────────────────────────────────────────

/// FFT stage graph: input/output `[rows, 2*nfft]` — re plane then im plane, the
/// `Op::Fft` 2N-real convention. Forward, unnormalized. No weights.
fn fft_graph(rows: usize, nfft: usize) -> Graph {
    let mut g = Graph::new("fft");
    let x = g.input("x", Shape::new(&[rows, 2 * nfft], DType::F32));
    let y = g.fft(x, false);
    g.set_outputs(vec![y]);
    g
}

/// `rows` deterministic real signals of length `nfft`, laid out `[rows, 2*nfft]`
/// (re = signal, im = 0). Seeded per rank so each node FFTs a distinct batch.
fn fft_signal(seed: u64, rows: usize, nfft: usize) -> Vec<f32> {
    let mut v = vec![0f32; rows * 2 * nfft];
    for r in 0..rows {
        for i in 0..nfft {
            let (t, s) = (i as f32, seed as f32);
            v[r * 2 * nfft + i] = (t * (0.3 + 0.1 * s) + r as f32).sin() + 0.5 * (t * 0.75).cos();
        }
    }
    v
}

/// Naive DFT reference for a `[rows, 2*nfft]` batch (unnormalized forward).
fn fft_reference(x: &[f32], rows: usize, nfft: usize) -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    let mut out = vec![0f32; rows * 2 * nfft];
    for r in 0..rows {
        let base = r * 2 * nfft;
        for k in 0..nfft {
            let (mut re, mut im) = (0f32, 0f32);
            for nn in 0..nfft {
                let ang = -tau * (k * nn) as f32 / nfft as f32;
                let (xr, xi) = (x[base + nn], x[base + nfft + nn]);
                let (c, sn) = (ang.cos(), ang.sin());
                re += xr * c - xi * sn;
                im += xr * sn + xi * c;
            }
            out[base + k] = re;
            out[base + nfft + k] = im;
        }
    }
    out
}

fn check_fft(y: &[f32], x: &[f32], rows: usize, nfft: usize, who: &str) -> bool {
    let yref = fft_reference(x, rows, nfft);
    let err = max_abs_err(y, &yref);
    let scale = yref.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    let ok = err < 5e-3 * scale;
    eprintln!(
        "    FFT {who}: max_err={err:.2e} (scale {scale:.2e})  {}",
        if ok { "✓" } else { "✗" }
    );
    ok
}

/// FFT coordinator (rank 0): scatter signal batches → run its own on `device` →
/// gather each worker's spectrum → verify all vs the DFT reference.
fn run_fft_coordinator(group: &Arc<ProcessGroup>, label: &str, device: Device) -> bool {
    let n = group.world_size();
    let (rows, nfft) = (4usize, 16usize);
    let worker_dev = std::env::var("WORKER_DEVICE").unwrap_or_else(|_| "cpu".into());

    for r in 1..n {
        let spec = dist::StageSpec::new(fft_graph(rows, nfft), "x", "y", worker_dev.clone());
        dist::ship_stage(group, r, &spec).expect("ship fft stage");
        dist::send_activation(group, r, &fft_signal(r as u64, rows, nfft)).expect("scatter");
    }
    eprintln!(
        "[{label}] FFT COORDINATOR: shipped Op::Fft graph + scattered {rows}×{nfft} signals to {} worker(s), worker_dev='{worker_dev}'",
        n - 1
    );

    let mut c0 = Session::new(device).compile(fft_graph(rows, nfft));
    let x0 = fft_signal(0, rows, nfft);
    let y0 = c0.run(&[("x", x0.as_slice())]).into_iter().next().unwrap();
    let mut ok = check_fft(
        &y0,
        &x0,
        rows,
        nfft,
        &format!("rank0 [{}]", device_label(device)),
    );

    for r in 1..n {
        let y = dist::recv_activation(group, r).expect("gather");
        let x = fft_signal(r as u64, rows, nfft);
        ok &= check_fft(&y, &x, rows, nfft, &format!("rank{r}"));
    }
    eprintln!(
        "[{label}] FFT COORDINATOR: {} across {n} node(s)",
        if ok {
            "ALL SPECTRA MATCH DFT REFERENCE ✓"
        } else {
            "MISMATCH ✗"
        }
    );
    ok
}

/// FFT worker (rank r): receive the graph (no weights) + its signal batch, run
/// the FFT on its device, send the spectrum back to the coordinator.
fn run_fft_worker(group: &Arc<ProcessGroup>, label: &str) -> bool {
    let mut stage = match dist::recv_stage(group, |_uri| Vec::new()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[{label}] FFT worker: {e}");
            return false;
        }
    };
    let x = dist::recv_activation(group, 0).expect("recv signal");
    let y = stage.run(&x);
    dist::send_activation(group, 0, &y).expect("send spectrum");
    eprintln!(
        "[{label}] FFT WORKER rank {}: ran Op::Fft on {} ({} values) → coordinator",
        group.rank(),
        device_label(stage.device),
        y.len()
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
        let mut grad_ref = [0.0f32; D];
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
        "i8" | "int8" => DType::I8,
        "u8" | "uint8" => DType::U8,
        _ => DType::F32,
    }
}

/// `MODE=federated`: bounded-staleness federated averaging across the mesh.
/// Each rank contributes a vector; the coordinator averages those that arrive
/// within the deadline and drops the rest (so a slow/offline edge client can't
/// stall the round). `LATE_MS` makes this rank contribute late (to simulate a
/// straggler); `FED_DEADLINE_MS` sets the coordinator's patience.
fn run_federated(group: &Arc<ProcessGroup>, label: &str) -> bool {
    let n = group.world_size();
    let d = 8usize;
    let late_ms: u64 = env("LATE_MS", "0").parse().unwrap_or(0);
    let deadline = Duration::from_millis(env("FED_DEADLINE_MS", "200").parse().unwrap_or(200));

    let mut data = vec![group.rank() as f32 + 1.0; d]; // client update = rank+1
    if late_ms > 0 {
        std::thread::sleep(Duration::from_millis(late_ms));
    }
    let present = group
        .federated_average(&mut data, deadline)
        .expect("federated_average");

    if group.is_leader() {
        eprintln!(
            "[{label}] FEDERATED: aggregated {present}/{n} clients within {}ms → avg[0]={:.3}",
            deadline.as_millis(),
            data[0]
        );
    } else {
        eprintln!(
            "[{label}] FEDERATED: client rank {} contributed (late={late_ms}ms)",
            group.rank()
        );
    }
    true // dropout is expected behavior, not a failure
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
