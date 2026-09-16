// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A three-node cluster on one machine, end to end.
//!
//! Runs the whole path against real hardware and a real checkpoint index:
//! probe → derive cost → plan → retire a node → execute a pipeline-parallel
//! forward pass across three ranks over loopback TCP, and check the answer
//! matches running every layer in one process.
//!
//! The three "nodes" are three loopback ranks on this machine, so the plan's
//! hardware numbers are all the same box — that is the point: it exercises the
//! planner and the transport without needing three machines. Placement on a
//! genuinely heterogeneous cluster is covered by the unit tests, which feed
//! synthetic `NodeCaps`.
//!
//! ```text
//! cargo run --release -p rlx-distributed --example three_node_local
//! cargo run --release -p rlx-distributed --example three_node_local -- <manifest.tsv>
//! ```
//!
//! With a manifest (`name<TAB>dims<TAB>ggml_type`, as produced by
//! `rlx-models`' `scripts/glm5next_subset.py`) the cost model is derived from a
//! real checkpoint; without one it uses a GLM-5.3-Flash-shaped index built in.

use anyhow::Result;
use rlx_distributed::cluster::*;
use rlx_distributed::{
    BlockInput, BlockOutput, BlockRole, BlockRunner, PipelineCoordinator, block_role,
    pipeline_layer_range,
};
use rlx_driver::{NetTransport, ProcessGroup};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::ops::Range;
use std::sync::Arc;
use std::thread;

const HIDDEN: usize = 4096;
const KV_LORA: usize = 512;
const KDA_HEADS: usize = 64;
const KDA_HEAD_DIM: usize = 128;
const N_LAYERS: usize = 45;
const EXPERTS: usize = 288;
const TOP_K: usize = 8;

/// A GLM-5.3-Flash-shaped tensor index: 34 KDA layers, 11 latent-attention
/// layers, MoE from layer 3, with realistic packed byte counts (UD-IQ1_S).
fn builtin_index() -> Vec<TensorEntry> {
    let mut v = vec![
        TensorEntry::new("token_embd.weight", vec![HIDDEN, 154_880], 356_843_520),
        TensorEntry::new("output.weight", vec![HIDDEN, 154_880], 356_843_520),
        TensorEntry::new("output_norm.weight", vec![HIDDEN], 16_384),
    ];
    for i in 0..N_LAYERS {
        let b = format!("blk.{i}");
        v.push(TensorEntry::new(
            format!("{b}.attn_norm.weight"),
            vec![HIDDEN],
            16_384,
        ));
        v.push(TensorEntry::new(
            format!("{b}.ffn_norm.weight"),
            vec![HIDDEN],
            16_384,
        ));
        if i % 4 == 3 {
            // Latent attention + DSA indexer.
            v.push(TensorEntry::new(
                format!("{b}.attn_kv_a_mqa.weight"),
                vec![HIDDEN, KV_LORA],
                8_912_896,
            ));
            v.push(TensorEntry::new(
                format!("{b}.attn_q_b.weight"),
                vec![1536, 16_384],
                26_738_688,
            ));
            v.push(TensorEntry::new(
                format!("{b}.attn_output.weight"),
                vec![16_384, HIDDEN],
                46_137_344,
            ));
        } else {
            // KDA: recurrent state, no per-token cache.
            v.push(TensorEntry::new(format!("{b}.ssm_a"), vec![KDA_HEADS], 256));
            v.push(TensorEntry::new(
                format!("{b}.ssm_norm.weight"),
                vec![KDA_HEAD_DIM],
                512,
            ));
            v.push(TensorEntry::new(
                format!("{b}.attn_q.weight"),
                vec![HIDDEN, 8192],
                23_068_672,
            ));
            v.push(TensorEntry::new(
                format!("{b}.attn_output.weight"),
                vec![8192, HIDDEN],
                23_068_672,
            ));
        }
        if i >= 3 {
            // Routed experts: the bulk of the model.
            for bank in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                v.push(TensorEntry::new(
                    format!("{b}.{bank}.weight"),
                    vec![EXPERTS, 2048, HIDDEN],
                    622_854_144,
                ));
            }
        } else {
            v.push(TensorEntry::new(
                format!("{b}.ffn_down.weight"),
                vec![12_288, HIDDEN],
                41_287_680,
            ));
        }
    }
    v
}

/// Parse a `name<TAB>d,d,d<TAB>TYPE` manifest into a tensor index.
fn manifest_index(path: &str) -> Result<Vec<TensorEntry>> {
    const BLOCK: &[(&str, usize, usize)] = &[
        ("F32", 1, 4),
        ("F16", 1, 2),
        ("BF16", 1, 2),
        ("Q8_0", 32, 34),
        ("Q5_K", 256, 176),
        ("Q6_K", 256, 210),
        ("Q4_K", 256, 144),
        ("Q3_K", 256, 110),
        ("Q2_K", 256, 84),
        ("IQ1_S", 256, 50),
        ("IQ1_M", 256, 56),
        ("IQ2_XXS", 256, 66),
        ("IQ3_XXS", 256, 98),
        ("IQ4_XS", 256, 136),
    ];
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut f = line.split('\t');
        let (name, dims, ty) = match (f.next(), f.next(), f.next()) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => continue,
        };
        let dims: Vec<usize> = dims.split(',').filter_map(|d| d.parse().ok()).collect();
        let elems: usize = dims.iter().product();
        let Some((_, blk, sz)) = BLOCK.iter().find(|(n, _, _)| *n == ty) else {
            anyhow::bail!("{name}: unknown ggml type {ty}");
        };
        out.push(TensorEntry::new(name, dims, (elems / blk * sz) as u64));
    }
    Ok(out)
}

/// Where the probe measures disk from.
///
/// This is not cosmetic: `bench_io_mbps` times a read of the largest file in
/// `ckpt_dir` precisely so the number describes the medium the routed experts
/// will page from. Pointing it at `/tmp` measures whatever junk is largest
/// there — or falls through to a write probe, whose throughput on APFS swings
/// several-fold run to run and swings the plan with it.
fn ckpt_dir() -> String {
    for d in [
        "/tmp/rlx-weights/glm5next",
        "/Volumes/FOUR/rlx-weights",
        "/tmp",
    ] {
        if std::path::Path::new(d).is_dir() {
            return d.into();
        }
    }
    "/tmp".into()
}

fn node(addr: &str, role: NodeRole, backs: Option<&str>) -> NodeConfig {
    NodeConfig {
        addr: addr.into(),
        ssh: None, // local: probed in-process, no SSH
        ckpt_dir: ckpt_dir(),
        device: DeviceList::default(),
        precision: Precision::default(),
        kv_cache: KvPolicy::Host,
        rng_seed: None,
        max_ram_gb: None,
        layers: None,
        role,
        standby_for: backs.map(String::from),
    }
}

// ─────────────────────── real 3-rank execution ───────────────────────

/// A stage that runs actual arithmetic per layer, so the distributed answer can
/// be checked against a single-process one.
///
/// `h[i] += (layer + 1) * scale` per layer — order-independent per element but
/// order-*dependent* in aggregate, so a dropped, doubled or misordered stage
/// changes the result.
struct SumStage {
    role: BlockRole,
    layers: Range<usize>,
    hidden: usize,
}

fn serial_reference(n_layers: usize, hidden: usize, token: u32) -> Vec<f32> {
    let mut h: Vec<f32> = (0..hidden).map(|i| token as f32 + i as f32).collect();
    for g in 0..n_layers {
        for x in h.iter_mut() {
            *x += (g + 1) as f32;
        }
    }
    h
}

impl BlockRunner for SumStage {
    fn role(&self) -> BlockRole {
        self.role
    }
    fn run(&mut self, input: BlockInput<'_>) -> Result<BlockOutput> {
        let mut h: Vec<f32> = match input {
            BlockInput::Tokens(ids) => {
                let t = *ids.last().unwrap() as f32;
                (0..self.hidden).map(|i| t + i as f32).collect()
            }
            BlockInput::Hidden(hv) => hv.to_vec(),
        };
        for g in self.layers.clone() {
            for x in h.iter_mut() {
                *x += (g + 1) as f32;
            }
        }
        Ok(match self.role {
            BlockRole::Last | BlockRole::Single => BlockOutput::Logits(h),
            _ => BlockOutput::Hidden(h),
        })
    }
}

/// Run `world` loopback ranks, each owning one stage.
///
/// Roles and layer ranges come from [`block_role`] / [`pipeline_layer_range`]
/// rather than being derived here: the pipeline assigns blocks in **reverse**
/// (rank 0 is the `Last` block), and hand-rolling that mapping gets it backwards.
fn run_pipeline(num_layers: usize, world: u32, hidden: usize, token: u32) -> Vec<f32> {
    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    println!(
        "  ranks listening on {}",
        addrs
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    // `forward_step`'s callback must yield a token id, so the full hidden vector
    // is captured on the side to compare against the serial run.
    let out = Arc::new(std::sync::Mutex::new(Vec::new()));

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let addrs = addrs.clone();
            let out = out.clone();
            thread::spawn(move || {
                let t = NetTransport::from_listener(rank as u32, world, listener, addrs, 1 << 20)
                    .expect("transport");
                let group = ProcessGroup::new(Arc::new(t));
                let rank = rank as u32;
                let role = block_role(rank, world);
                let mut stage = SumStage {
                    role,
                    layers: pipeline_layer_range(num_layers, rank, world),
                    hidden,
                };
                let coord = PipelineCoordinator::new(group);
                let captured = out.clone();
                coord
                    .forward_step(&mut stage, &[token], move |l| {
                        *captured.lock().unwrap() = l.to_vec();
                        l[0].round() as u32
                    })
                    .expect("forward");
                coord.barrier().expect("barrier");
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    std::sync::Mutex::into_inner(
        std::sync::Arc::into_inner(out).expect("every rank thread has joined"),
    )
    .unwrap_or_else(|e| e.into_inner())
}

fn main() -> Result<()> {
    let index = match std::env::args().nth(1) {
        Some(p) => {
            println!("== tensor index from {p} ==");
            manifest_index(&p)?
        }
        None => {
            println!("== tensor index: built-in GLM-5.3-Flash shape ==");
            builtin_index()
        }
    };

    // ── 1. three local nodes; the third is a hot spare for the second ──
    let cfg = ClusterConfig {
        model: "glm-5.3-flash".into(),
        seq: 8,
        rng_seed: None,
        reserve_ram_gb: 4.0,
        placement: PlacementSection {
            policy: PlacementPolicy::Auto,
            objective: Objective {
                optimize: Optimize::Throughput,
                context: 8192,
                ..Default::default()
            },
        },
        discover: Default::default(),
        nodes: vec![
            node("127.0.0.1:9100", NodeRole::Active, None),
            node("127.0.0.1:9101", NodeRole::Active, None),
            node("127.0.0.1:9102", NodeRole::Standby, Some("127.0.0.1:9101")),
        ],
    };
    let mut cx = Cluster::from_config(cfg);

    // ── 2. probe this machine, once per node ──
    println!("\n== probe ==");
    for caps in cx.probe("")? {
        println!("  {} | {}", caps.addr, caps.summary());
        for d in &caps.devices {
            let mem = if d.mem_bytes > 0 {
                format!("{:.1} GB", d.mem_bytes as f64 / 1e9)
            } else {
                "unified".into()
            };
            if d.available {
                println!(
                    "      {:<8} {:<28} {:>7.1} GFLOP/s {mem}",
                    d.device, d.name, d.gflops
                );
            } else {
                // Present on the box, absent from the build. Saying so beats
                // printing 0.0 GFLOP/s, which reads as a failed benchmark.
                println!(
                    "      {:<8} {:<28} {:>7} {mem}  (not built in)",
                    d.device, d.name, "—"
                );
            }
        }
        println!("      disk {:.0} MB/s read", caps.io_mbps);
    }

    // ── 3. cost from the checkpoint, then plan ──
    let opts = CostOptions::default().with_experts(TOP_K, EXPERTS);
    let cost = ModelCost::from_tensor_index(&index, &opts)?;
    println!("\n== model ==\n  {}", cost.summary());
    println!("  hidden {} | {}", cost.hidden_size, {
        let kv = cost.kv.clone();
        kv.summary(cost.n_layers as u64, 8192)
    });

    println!("\n== plan (policy = auto, context = 8192) ==");
    cx.plan(cost.clone())?;
    print!("{}", cx.plan_table());

    // ── 4. retire the node the spare is backing ──
    println!("\n== retire 127.0.0.1:9101 ==");
    cx.retire("127.0.0.1:9101", cost)?;
    print!("{}", cx.plan_table());

    // ── 5. actually run three ranks over loopback TCP ──
    println!("\n== pipeline forward, 3 ranks over loopback ==");
    let hidden = 8usize;
    let n_layers = 12usize;
    let world = 3u32;
    println!(
        "  stages: {}",
        (0..world)
            .map(|r| {
                let g = pipeline_layer_range(n_layers, r, world);
                format!("rank{r}={:?} {}..{}", block_role(r, world), g.start, g.end)
            })
            .collect::<Vec<_>>()
            .join("  ")
    );
    let got = run_pipeline(n_layers, world, hidden, 7);
    let want = serial_reference(n_layers, hidden, 7);
    println!("  distributed: {got:?}");
    println!("  single-proc: {want:?}");
    anyhow::ensure!(got == want, "distributed result differs from serial");
    println!("  ✓ three ranks agree with one process");
    Ok(())
}
