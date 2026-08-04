// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **rlx-opscope** — a data-pattern recording harness.
//!
//! The idea: run existing op graphs with varied synthetic (and, later, real)
//! data and record cheap *sketches* of every tensor flowing through the ops we
//! care about — matmuls first. Offline we mine those sketches for exploitable
//! structure (sparsity, per-channel outliers, quantization headroom, sequence
//! structure) that justifies a specialized kernel.
//!
//! The load-bearing trick (see the crate README) is that we do **not** tap the
//! executor. We *rewrite the graph*: for each op-site we append reduction /
//! histogram nodes on its inputs and output and mark them as extra graph
//! outputs. The stats are then computed by the backend's own kernels — so this
//! works identically on CPU, Metal, CUDA, … and costs nothing when not applied.
//!
//! This module provides:
//! - [`inject_matmul_stats`] — the stat-injection graph pass (matmul MVP).
//! - [`StatConfig`] / [`StatSpec`] — what to record and how to label it.
//! - [`Dist`] / [`gen`] — synthetic distribution generators.
//! - [`Recorder`] — a dependency-free tidy-CSV sink.

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Philox4x32, Shape};
use std::collections::HashMap;
use std::io::{self, Write};

pub mod dataflow;
pub mod demo;
pub mod guard;
// Decomposition/actuation kernels + the per-layer allocator are opt-in — they're
// heavy, experimental optimization surface, kept out of the default build so the
// base harness stays lean (see the `decompose`/`allocate` features).
#[cfg(feature = "decompose")]
pub mod kernels;
#[cfg(feature = "allocate")]
pub mod layers;
pub mod live;
pub mod motifs;
pub mod online;
pub mod optimize;
pub mod parity;
#[cfg(feature = "parquet")]
pub mod parquet_sink;
pub mod probe;
pub mod shapes;
#[cfg(feature = "decompose")]
pub mod svd;
pub mod timing;

// ─────────────────────────── stat injection ───────────────────────────

/// Which sketches to append per tapped tensor.
#[derive(Clone, Copy, Debug)]
pub struct StatConfig {
    /// Value-histogram bin count.
    pub hist_bins: usize,
    /// Histogram lower / upper edge, used only when `hist_normalize` is off.
    pub hist_min: f32,
    pub hist_max: f32,
    /// Scale-normalize the histogram: bin `x / (max|x| + eps)` over `[-1, 1]`
    /// so the *shape* is comparable regardless of tensor scale. Fixes fixed-range
    /// over-reporting "spiky" on tightly-scaled weights. One extra pass in-graph.
    pub hist_normalize: bool,
    /// Per-last-axis `max|x|` — the per-channel outlier signal (quant headroom).
    pub per_channel: bool,
    /// Per-row (reduce the last axis) sum-of-squares — the sequence/position
    /// energy profile (sinks, decay, per-token sparsity).
    pub per_position: bool,
    /// Adjacent-row diff energy along axis 0 — within-call sequence coherence
    /// (adjacent rows/tokens similar → delta-compute exploit). 2-D tensors only.
    pub adjacency: bool,
}

impl Default for StatConfig {
    fn default() -> Self {
        Self {
            hist_bins: 32,
            hist_min: -6.0,
            hist_max: 6.0,
            hist_normalize: true,
            per_channel: true,
            per_position: true,
            adjacency: true,
        }
    }
}

/// One appended output: where it is in `graph.outputs`, and what it means.
#[derive(Clone, Debug)]
pub struct StatSpec {
    /// Index into the compiled graph's output list (== index into `run()`'s
    /// returned `Vec<Vec<f32>>`).
    pub out_idx: usize,
    /// Op-site label (node name, or `matmul#<id>`).
    pub site: String,
    /// Tensor role at the site: `"lhs"`, `"rhs"`, or `"out"`.
    pub role: &'static str,
    /// Sketch kind: `min`/`max`/`mean`/`l1`/`sumsq`/`nnz`/`hist`/
    /// `chan_maxabs`/`pos_sumsq`.
    pub stat: &'static str,
    /// Element count of this sketch (1 for scalars, `bins`/`C`/`rows` for
    /// vectors).
    pub len: usize,
    /// Element count of the *source* tensor (for density / normalization).
    pub numel: usize,
    /// FLOPs of the op at this site (`2·M·K·N` for a matmul; 0 for non-matmul
    /// taps). Lets the opportunity miner weight recurrence by cost.
    pub flops: u64,
}

/// Write a sidecar mapping each op-site to its FLOPs (deduplicated), consumed by
/// `opscope-optimize` to weight temporal-recurrence savings by op cost.
pub fn write_site_costs(path: &str, specs: &[StatSpec]) -> io::Result<()> {
    let mut seen: HashMap<String, u64> = HashMap::new();
    for s in specs {
        seen.entry(s.site.clone()).or_insert(s.flops);
    }
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "site,flops")?;
    for (site, flops) in seen {
        writeln!(f, "{},{}", site.replace(',', "_"), flops)?;
    }
    f.flush()
}

/// Reduce every axis of `t` to a single f32 (`[1]`).
fn reduce_scalar(g: &mut Graph, t: NodeId, op: ReduceOp) -> NodeId {
    let rank = g.shape(t).rank();
    g.reduce(
        t,
        op,
        (0..rank).collect(),
        false,
        Shape::new(&[1], DType::F32),
    )
}

/// Build the sketch nodes for one tensor `t`; returns `(node, stat_name)`.
fn build_tensor_stats(g: &mut Graph, t: NodeId, cfg: &StatConfig) -> Vec<(NodeId, &'static str)> {
    let tshape = g.shape(t).clone();
    let rank = tshape.rank();
    let mut out: Vec<(NodeId, &'static str)> = Vec::new();

    out.push((reduce_scalar(g, t, ReduceOp::Min), "min"));
    out.push((reduce_scalar(g, t, ReduceOp::Max), "max"));
    out.push((reduce_scalar(g, t, ReduceOp::Mean), "mean"));

    // L1 = Σ|x|
    let abs = g.activation(Activation::Abs, t, tshape.clone());
    out.push((reduce_scalar(g, abs, ReduceOp::Sum), "l1"));

    // Σx² (→ L2 norm in the miner via sqrt)
    let sq = g.add_node(Op::Binary(BinaryOp::Mul), vec![t, t], tshape.clone());
    out.push((reduce_scalar(g, sq, ReduceOp::Sum), "sumsq"));

    // nnz (→ density in the miner via / numel): count where x != 0
    let zero = g.full(&[1], 0.0, DType::F32);
    let ne = g.add_node(
        Op::Compare(CmpOp::Ne),
        vec![t, zero],
        tshape.clone().with_dtype(DType::Bool),
    );
    let ne_f = g.add_node(Op::Cast { to: DType::F32 }, vec![ne], tshape.clone());
    out.push((reduce_scalar(g, ne_f, ReduceOp::Sum), "nnz"));

    // Kurtosis = E[(x-μ)⁴]/E[(x-μ)²]² — a heavy-tail / outlier proxy: high means
    // the mass is concentrated in rare extreme values (gaussian ≈ 3), so the
    // tensor is HARD to quantize (a few outliers dominate the range). The flowing
    // data's real quant-difficulty signal, per op — beyond what static shapes show.
    let one = Shape::new(&[1], DType::F32);
    let mean1 = reduce_scalar(g, t, ReduceOp::Mean);
    let cen = g.add_node(Op::Binary(BinaryOp::Sub), vec![t, mean1], tshape.clone()); // x-μ (bcast)
    let c2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![cen, cen], tshape.clone());
    let m2 = reduce_scalar(g, c2, ReduceOp::Mean);
    let c4 = g.add_node(Op::Binary(BinaryOp::Mul), vec![c2, c2], tshape.clone());
    let m4 = reduce_scalar(g, c4, ReduceOp::Mean);
    let m2sq = g.add_node(Op::Binary(BinaryOp::Mul), vec![m2, m2], one.clone());
    let eps4 = g.full(&[1], 1e-20, DType::F32);
    let kden = g.add_node(Op::Binary(BinaryOp::Add), vec![m2sq, eps4], one.clone());
    let kurt = g.add_node(Op::Binary(BinaryOp::Div), vec![m4, kden], one.clone());
    out.push((kurt, "kurtosis"));

    // Value histogram. When normalized, bin `x / (max|x| + eps)` over [-1,1] so
    // the distribution *shape* is scale-invariant (a tight gaussian and a wide
    // one read the same; only genuinely spiky/quantized data reads spiky).
    // `hist_bins == 0` skips the histogram entirely — it's the node-heaviest sketch
    // (bins × Compare+Reduce), so a caller tapping MANY sites (whole-model inject)
    // that doesn't need the distribution can drop it to keep the injected graph
    // small enough to compile.
    if cfg.hist_bins > 0 {
        let hist_node = if cfg.hist_normalize {
            let absn = g.activation(Activation::Abs, t, tshape.clone());
            let scale = reduce_scalar(g, absn, ReduceOp::Max); // [1] = max|x|
            let eps = g.full(&[1], 1e-12, DType::F32);
            let denom = g.add_node(
                Op::Binary(BinaryOp::Add),
                vec![scale, eps],
                Shape::new(&[1], DType::F32),
            );
            let normed = g.add_node(Op::Binary(BinaryOp::Div), vec![t, denom], tshape.clone());
            g.histogram(normed, cfg.hist_bins, -1.0, 1.0)
        } else {
            g.histogram(t, cfg.hist_bins, cfg.hist_min, cfg.hist_max)
        };
        out.push((hist_node, "hist"));
    }

    // Element-wise fingerprint: gather `k` spread elements. Unlike the moment/
    // histogram sketches (which capture the *distribution*), this captures the
    // actual *values*, so the optimizer can tell an exact temporal repeat
    // (memoize) from mere distributional stationarity, and measure real per-step
    // element drift (delta-compute).
    let numel = tshape.num_elements().unwrap_or(0);
    let k = 16usize.min(numel);
    if k > 0 {
        let flat = g.reshape_(t, vec![numel as i64]);
        let idx: Vec<i64> = (0..k).map(|j| (j * numel / k) as i64).collect();
        let data: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let idx_node = g.add_node(Op::Constant { data }, vec![], Shape::new(&[k], DType::I64));
        let sig = g.gather_(flat, idx_node, 0);
        out.push((sig, "elemsig"));
    }

    if rank >= 2 && cfg.per_channel {
        // max|x| per last-axis channel: reduce every axis but the last.
        let absc = g.activation(Activation::Abs, t, tshape.clone());
        let c = tshape.dim(rank - 1).unwrap_static();
        let chan = g.reduce(
            absc,
            ReduceOp::Max,
            (0..rank - 1).collect(),
            false,
            Shape::new(&[c], DType::F32),
        );
        // Outlier SEVERITY as a scalar: peak channel / mean channel (of max|·|).
        // High ⇒ a few channels carry the mass — the runtime driver of quant loss
        // (int4 clips them; per-token int8 lets them crush the rest → W8A8). This
        // is exactly the AWQ/SmoothQuant saliency signal, recorded per op.
        let cmax = reduce_scalar(g, chan, ReduceOp::Max);
        let cmean = reduce_scalar(g, chan, ReduceOp::Mean);
        let epsc = g.full(&[1], 1e-20, DType::F32);
        let cden = g.add_node(
            Op::Binary(BinaryOp::Add),
            vec![cmean, epsc],
            Shape::new(&[1], DType::F32),
        );
        let cout = g.add_node(
            Op::Binary(BinaryOp::Div),
            vec![cmax, cden],
            Shape::new(&[1], DType::F32),
        );
        out.push((chan, "chan_maxabs"));
        out.push((cout, "chan_outlier"));
    }

    if rank >= 2 && cfg.per_position {
        // Σx² per row: reduce the last axis only → [rows].
        let sqp = g.add_node(Op::Binary(BinaryOp::Mul), vec![t, t], tshape.clone());
        let c = tshape.dim(rank - 1).unwrap_static();
        let rows = tshape.num_elements().unwrap_or(c) / c;
        let pos = g.reduce(
            sqp,
            ReduceOp::Sum,
            vec![rank - 1],
            false,
            Shape::new(&[rows], DType::F32),
        );
        out.push((pos, "pos_sumsq"));
    }

    if rank == 2 && cfg.adjacency {
        // Σ(row_t − row_{t-1})² per adjacent pair along axis 0. Small values
        // (relative to per-row energy) mean adjacent rows cohere → the matmul
        // could compute only the delta from the previous row.
        let rows = tshape.dim(0).unwrap_static();
        let d = tshape.dim(1).unwrap_static();
        if rows >= 2 {
            let pair = Shape::new(&[rows - 1, d], DType::F32);
            let prev = g.narrow_(t, 0, 0, rows - 1);
            let next = g.narrow_(t, 0, 1, rows - 1);
            let diff = g.add_node(Op::Binary(BinaryOp::Sub), vec![next, prev], pair.clone());
            let sq = g.add_node(Op::Binary(BinaryOp::Mul), vec![diff, diff], pair);
            let adj = g.reduce(
                sq,
                ReduceOp::Sum,
                vec![1],
                false,
                Shape::new(&[rows - 1], DType::F32),
            );
            out.push((adj, "adj_sumsq"));
        }
    }

    out
}

/// Rewrite `graph` so that every `Op::MatMul` / `Op::DequantMatMul` site's `lhs`,
/// `rhs`, `out` tensors get sketch outputs appended. The original outputs stay
/// first (same indices), so downstream code that reads output 0 is unaffected.
/// Returns the rewritten graph and the manifest describing each appended output.
pub fn inject_matmul_stats(graph: &Graph, cfg: &StatConfig) -> (Graph, Vec<StatSpec>) {
    inject_matmul_stats_filtered(graph, cfg, &|_lhs_numel, _out_numel| true)
}

/// [`inject_matmul_stats`] but only taps sites for which `keep(lhs_numel,
/// out_numel)` is true. The injected graph's compile cost grows superlinearly with
/// the number of tapped sites, so whole-model injection (100+ matmuls) can become
/// intractable; a `keep` predicate lets a caller tap just the tensors it reports
/// (e.g. residual stream + query by their element counts), keeping the graph small
/// enough to compile at depth.
pub fn inject_matmul_stats_filtered(
    graph: &Graph,
    cfg: &StatConfig,
    keep: &dyn Fn(usize, usize) -> bool,
) -> (Graph, Vec<StatSpec>) {
    let mut g = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    // (site, lhs, rhs, out) in the *new* graph's ids.
    let mut matmuls: Vec<(String, NodeId, NodeId, NodeId)> = Vec::new();
    // DequantMatMul sites: (site, lhs=activation, out). The rhs is PACKED quant
    // bytes (u8), not an f32 tensor, so we tap only the f32 activation + output —
    // captures the activation dataflow of quantized projections (MXFP4/MXFP8/int*).
    let mut dq_matmuls: Vec<(String, NodeId, NodeId)> = Vec::new();

    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = g.add_node(node.op.clone(), inputs.clone(), node.shape.clone());
        id_map.insert(node.id, new_id);
        if matches!(node.op, Op::MatMul) {
            // Always suffix the node id — flow-lowered graphs give every matmul
            // the same name ("mir"), which would otherwise collapse all sites.
            let site = match &node.name {
                Some(n) => format!("{n}#{}", node.id.0),
                None => format!("matmul#{}", node.id.0),
            };
            matmuls.push((site, inputs[0], inputs[1], new_id));
        }
        if matches!(node.op, Op::DequantMatMul { .. }) {
            let site = match &node.name {
                Some(n) => format!("{n}#{}", node.id.0),
                None => format!("dqmm#{}", node.id.0),
            };
            dq_matmuls.push((site, inputs[0], new_id));
        }
    }

    let mut outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
    let mut specs: Vec<StatSpec> = Vec::new();

    for (site, lhs, rhs, out_t) in matmuls {
        // 2·M·K·N: lhs [M,K], out [.,N].
        let ls = g.shape(lhs).clone();
        let os = g.shape(out_t).clone();
        if !keep(
            ls.num_elements().unwrap_or(0),
            os.num_elements().unwrap_or(0),
        ) {
            continue;
        }
        let m = ls.dim(0).unwrap_static();
        let kk = ls.dim(ls.rank() - 1).unwrap_static();
        let n = os.dim(os.rank() - 1).unwrap_static();
        let flops = 2 * m as u64 * kk as u64 * n as u64;
        for (role, t) in [("lhs", lhs), ("rhs", rhs), ("out", out_t)] {
            let numel = g.shape(t).num_elements().unwrap_or(0);
            for (node, stat) in build_tensor_stats(&mut g, t, cfg) {
                let len = g.shape(node).num_elements().unwrap_or(1);
                specs.push(StatSpec {
                    out_idx: outputs.len(),
                    site: site.clone(),
                    role,
                    stat,
                    len,
                    numel,
                    flops,
                });
                outputs.push(node);
            }
        }
    }

    // DequantMatMul taps: activation (lhs) + output only (rhs is packed bytes).
    for (site, lhs, out_t) in dq_matmuls {
        let ls = g.shape(lhs).clone();
        let os = g.shape(out_t).clone();
        if !keep(
            ls.num_elements().unwrap_or(0),
            os.num_elements().unwrap_or(0),
        ) {
            continue;
        }
        let m = ls.dim(0).unwrap_static();
        let kk = ls.dim(ls.rank() - 1).unwrap_static();
        let n = os.dim(os.rank() - 1).unwrap_static();
        let flops = 2 * m as u64 * kk as u64 * n as u64;
        for (role, t) in [("lhs", lhs), ("out", out_t)] {
            let numel = g.shape(t).num_elements().unwrap_or(0);
            for (node, stat) in build_tensor_stats(&mut g, t, cfg) {
                let len = g.shape(node).num_elements().unwrap_or(1);
                specs.push(StatSpec {
                    out_idx: outputs.len(),
                    site: site.clone(),
                    role,
                    stat,
                    len,
                    numel,
                    flops,
                });
                outputs.push(node);
            }
        }
    }

    g.set_outputs(outputs);
    (g, specs)
}

/// Build a **symmetric int8 fake-quant** subgraph on activation `a` (`[…, k]`):
/// `round(a·127/amax)·amax/127`, clamped to `[-127,127]`. `per_channel=false`
/// scopes `amax` per-token (reduce the last/feature axis) — hardware-friendly but
/// wrecked by a single outlier channel; `per_channel=true` scopes it per-channel
/// (reduce the token axes) — each outlier channel gets its own scale, the ceiling
/// SmoothQuant approximates. Returns the dequantized-back node (same shape as `a`).
/// The activation half of W8A8 — inserted before a matmul so the forward pass sees
/// quantized activations. `Op::Round` is the STE primitive rlx-ir notes for this.
fn fakequant(g: &mut Graph, a: NodeId, per_channel: bool) -> NodeId {
    let ashape = g.shape(a).clone();
    let rank = ashape.rank();
    if rank == 0 {
        return a;
    }
    let last = rank - 1;
    let dims: Vec<usize> = (0..rank).map(|i| ashape.dim(i).unwrap_static()).collect();
    // per-token: keep the last (feature) axis, reduce it → scale per row.
    // per-channel: reduce the token axes (all but last) → scale per feature channel.
    let (axes, mut rdims): (Vec<usize>, Vec<usize>) = if per_channel {
        let mut rd = dims.clone();
        (0..last).for_each(|i| rd[i] = 1);
        ((0..last).collect(), rd)
    } else {
        let mut rd = dims.clone();
        rd[last] = 1;
        (vec![last], rd)
    };
    if axes.is_empty() {
        rdims = dims.clone();
    }
    let rshape = Shape::new(&rdims, DType::F32);
    let scalar = |g: &mut Graph, v: f32| {
        g.add_node(
            Op::Constant {
                data: v.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        )
    };

    let abs = g.add_node(Op::Activation(Activation::Abs), vec![a], ashape.clone());
    let amax = g.reduce(abs, ReduceOp::Max, axes, true, rshape.clone());
    let inv = g.add_node(
        Op::Activation(Activation::Recip),
        vec![amax],
        rshape.clone(),
    ); // 1/amax
    let norm = g.add_node(Op::Binary(BinaryOp::Mul), vec![a, inv], ashape.clone()); // a/amax ∈ [-1,1] (bcast)
    let c127 = scalar(g, 127.0);
    let up = g.add_node(Op::Binary(BinaryOp::Mul), vec![norm, c127], ashape.clone());
    let r = g.add_node(Op::Activation(Activation::Round), vec![up], ashape.clone());
    let rc = g.add_node(
        Op::Clamp {
            min: -127.0,
            max: 127.0,
        },
        vec![r],
        ashape.clone(),
    );
    let c_inv = scalar(g, 1.0 / 127.0);
    let back = g.add_node(Op::Binary(BinaryOp::Mul), vec![rc, c_inv], ashape.clone()); // /127
    g.add_node(Op::Binary(BinaryOp::Mul), vec![back, amax], ashape.clone()) // * amax (bcast)
}

/// Rewrite `graph` so every `Op::MatMul`'s **activation input** (`input[0]`) is
/// per-token int8 fake-quantized — the missing half of an end-to-end **W8A8**
/// simulation (combine with int8 weights in the WeightMap). In a fused-attention
/// qwen graph the `MatMul` nodes are exactly the linear projections (Q/K/V/O/
/// gate/up/down + lm_head) — attention's internal QK/AV live inside `Op::Attention`
/// — so this quantizes precisely the linear-layer activations, as real W8A8 does.
/// Outputs are preserved (same indices); compile with `skip_fusion` (the inserted
/// nodes would otherwise trip SwiGLU fusion). `per_channel` picks the scale scope
/// (see [`fakequant`]) — `false` = per-token (deployable W8A8), `true` = per-channel
/// (the outlier-robust ceiling).
pub fn inject_activation_fakequant(graph: &Graph, per_channel: bool) -> Graph {
    let mut g = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if matches!(node.op, Op::MatMul) && !inputs.is_empty() {
            let fq = fakequant(&mut g, inputs[0], per_channel);
            let mut mm = inputs.clone();
            mm[0] = fq;
            g.add_node(Op::MatMul, mm, node.shape.clone())
        } else {
            g.add_node(node.op.clone(), inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }
    let outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
    g.set_outputs(outputs);
    g
}

/// One residual-`Add` tap: the recorded Σx² of each input, plus the block index.
pub struct ResidualSpec {
    /// Index into the run outputs for `‖input_a‖²` and `‖input_b‖²`.
    pub a_idx: usize,
    pub b_idx: usize,
    /// Execution order of this residual add (0,1 = layer0 attn/mlp, …).
    pub order: usize,
}

/// Tap every residual `Add(residual, block_out)` — record Σx² of both inputs so
/// the analysis can compute `‖block-delta‖/‖residual‖`, the **block-influence**
/// signal (ShortGPT/Gromov): a sublayer whose delta ≈ 0 barely changes the
/// residual stream ⇒ it's near-identity ⇒ a **layer-skip** candidate. Compile
/// with `skip_fusion` so the residual adds aren't folded into FusedResidualRmsNorm
/// before the tap sees them. Outputs are preserved (same indices).
pub fn inject_residual_stats(graph: &Graph) -> (Graph, Vec<ResidualSpec>) {
    let mut g = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut adds: Vec<(NodeId, NodeId)> = Vec::new();
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = g.add_node(node.op.clone(), inputs.clone(), node.shape.clone());
        id_map.insert(node.id, new_id);
        if matches!(node.op, Op::Binary(BinaryOp::Add)) && inputs.len() == 2 {
            adds.push((inputs[0], inputs[1]));
        }
    }
    let mut outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
    let mut specs = Vec::new();
    for (order, (a, b)) in adds.into_iter().enumerate() {
        let ash = g.shape(a).clone();
        let bsh = g.shape(b).clone();
        let sqa = g.add_node(Op::Binary(BinaryOp::Mul), vec![a, a], ash);
        let ssa = reduce_scalar(&mut g, sqa, ReduceOp::Sum);
        let sqb = g.add_node(Op::Binary(BinaryOp::Mul), vec![b, b], bsh);
        let ssb = reduce_scalar(&mut g, sqb, ReduceOp::Sum);
        specs.push(ResidualSpec {
            a_idx: outputs.len(),
            b_idx: outputs.len() + 1,
            order,
        });
        outputs.push(ssa);
        outputs.push(ssb);
    }
    g.set_outputs(outputs);
    (g, specs)
}

/// Rewrite `graph` to **skip** the residual blocks at the given execution orders
/// (same ordering as [`inject_residual_stats`]): each targeted `Add(residual,
/// block_out)` is replaced by its residual input (`input[0]`), so the block's
/// output is bypassed — and its whole subgraph becomes dead code the compiler
/// drops, actually removing that layer's weight-stream + compute. The concrete
/// "replace-like-fusing" action for a near-identity layer (layer minimization).
pub fn skip_residual_blocks(graph: &Graph, skip_orders: &[usize]) -> Graph {
    let mut g = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut order = 0usize;
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        if matches!(node.op, Op::Binary(BinaryOp::Add)) && inputs.len() == 2 {
            let this = order;
            order += 1;
            if skip_orders.contains(&this) {
                // Bypass: the residual (input[0]) flows straight through; the
                // block_out branch (input[1]) is left dangling → dead-code removed.
                id_map.insert(node.id, inputs[0]);
                continue;
            }
        }
        let new_id = g.add_node(node.op.clone(), inputs, node.shape.clone());
        id_map.insert(node.id, new_id);
    }
    let outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
    g.set_outputs(outputs);
    g
}

#[cfg(test)]
mod fakequant_tests {
    use super::*;
    use rlx_runtime::{CompileOptions, Device, Session};

    #[test]
    fn fakequant_subgraph_matches_per_row_int8() {
        let (m, k) = (3usize, 8usize);
        let mut g = Graph::new("fq");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let fq = fakequant(&mut g, x, false);
        g.set_outputs(vec![fq]);
        let mut o = CompileOptions::default();
        o.fusion_opts.skip_fusion = true;
        let mut c = Session::new(Device::Cpu).compile_with(g, &o);
        let xd: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.37 - 1.5).collect();
        let got = c.run(&[("x", &xd)]).remove(0);
        // reference: per-row symmetric int8 round-trip (matches quantize_row_i8).
        let mut want = xd.clone();
        for r in 0..m {
            let row = &mut want[r * k..(r + 1) * k];
            let amax = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let s = if amax < 1e-20 { 1.0 } else { amax / 127.0 };
            for v in row.iter_mut() {
                *v = (*v / s).round().clamp(-127.0, 127.0) * s;
            }
        }
        let num: f32 = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let den: f32 = want.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-9;
        assert!(
            num / den < 0.01,
            "fakequant vs per-row int8 rel-err {}",
            num / den
        );
    }
}

// ─────────────────── Tier 1: inference-dynamics taps ───────────────────

/// Reduce a single `axis` of `t`, giving explicit output dims.
fn reduce_axis(g: &mut Graph, t: NodeId, axis: usize, op: ReduceOp) -> (NodeId, usize) {
    let dims: Vec<usize> = {
        let s = g.shape(t);
        (0..s.rank()).map(|i| s.dim(i).unwrap_static()).collect()
    };
    let out_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d)
        .collect();
    let len: usize = out_dims.iter().product::<usize>().max(1);
    let out_shape = Shape::new(&out_dims, DType::F32);
    (g.reduce(t, op, vec![axis], false, out_shape), len)
}

/// Append attention (`Op::Softmax`) and MoE-routing (`Op::TopK`) sketches as
/// extra graph outputs — the distinctly *inference* signals:
/// - `attn_qmax`  : per-query peak attention mass (concentration → sparse/
///   windowed attention, keep top-k keys).
/// - `attn_krecv` : per-key received mass (sinks → KV eviction / sink tokens).
/// - `route_load` : per-expert selection count from the top-k router (skew →
///   drop cold experts, prefetch/merge hot ones).
pub fn inject_infer_stats(graph: &Graph, _cfg: &StatConfig) -> (Graph, Vec<StatSpec>) {
    let mut g = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    // (kind, site, node-in-new-graph, extra usize e.g. num_experts)
    let mut taps: Vec<(&'static str, String, NodeId, usize)> = Vec::new();

    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = g.add_node(node.op.clone(), inputs.clone(), node.shape.clone());
        id_map.insert(node.id, new_id);
        match &node.op {
            Op::Softmax { .. } => {
                let site = match &node.name {
                    Some(n) => format!("{n}#{}", node.id.0),
                    None => format!("softmax#{}", node.id.0),
                };
                taps.push(("softmax", site, new_id, 0));
            }
            // Universal MoE routing tap: every expert-dispatch (TopK router OR a
            // custom group_limited_gate) feeds a GroupedMatMul with expert ids in
            // input 2 — histogram those → per-expert load. Covers DeepSeek/GLM4-MoE
            // (custom gate, no Op::TopK) as well as Llama4/Qwen3.5.
            Op::GroupedMatMul if node.inputs.len() >= 3 => {
                let site = format!("gmm#{}", node.id.0);
                let w = &graph.node(node.inputs[1]).shape; // [num_experts, K, N]
                let experts = w.dim(0).unwrap_static();
                taps.push(("topk", site, inputs[2], experts)); // tap the expert-idx input
            }
            Op::TopK { .. } => {
                let site = match &node.name {
                    Some(n) => format!("{n}#{}", node.id.0),
                    None => format!("topk#{}", node.id.0),
                };
                // experts = last dim of the gate (the TopK input).
                let e = &graph.node(node.inputs[0]).shape;
                let experts = e.dim(e.rank() - 1).unwrap_static();
                taps.push(("topk", site, new_id, experts));
            }
            _ => {}
        }
    }

    let mut outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
    let mut specs: Vec<StatSpec> = Vec::new();
    let mut emit = |g: &mut Graph,
                    outs: &mut Vec<NodeId>,
                    node,
                    site: &str,
                    role: &'static str,
                    stat: &'static str,
                    numel: usize| {
        let len = g.shape(node).num_elements().unwrap_or(1);
        specs.push(StatSpec {
            out_idx: outs.len(),
            site: site.into(),
            role,
            stat,
            len,
            numel,
            flops: 0,
        });
        outs.push(node);
    };

    for (kind, site, node, experts) in taps {
        let rank = g.shape(node).rank();
        let numel = g.shape(node).num_elements().unwrap_or(0);
        match kind {
            "softmax" if rank >= 2 => {
                let (qmax, _) = reduce_axis(&mut g, node, rank - 1, ReduceOp::Max);
                emit(
                    &mut g,
                    &mut outputs,
                    qmax,
                    &site,
                    "attn",
                    "attn_qmax",
                    numel,
                );
                let (krecv, _) = reduce_axis(&mut g, node, rank - 2, ReduceOp::Sum);
                emit(
                    &mut g,
                    &mut outputs,
                    krecv,
                    &site,
                    "attn",
                    "attn_krecv",
                    numel,
                );
            }
            "topk" if experts > 0 => {
                // Histogram of selected expert ids → per-expert load.
                let load = g.histogram(node, experts, 0.0, experts as f32);
                emit(
                    &mut g,
                    &mut outputs,
                    load,
                    &site,
                    "route",
                    "route_load",
                    numel,
                );
            }
            _ => {}
        }
    }

    g.set_outputs(outputs);
    (g, specs)
}

/// Tap **fused `Op::Attention`** nodes: decompose each to its softmax-weights
/// node (via rlx's own `attention_softmax_weights`) and append per-query peak
/// mass + per-key received mass as extra outputs. This is how attention
/// concentration / sink keys are recovered from a *fused* attention op (which
/// otherwise exposes no softmax to tap). Adds a shadow `Q·Kᵀ→softmax` per site.
pub fn inject_attention_stats(graph: &Graph, _cfg: &StatConfig) -> (Graph, Vec<StatSpec>) {
    use rlx_autodiff::decompose_backward_kernels::attention_softmax_weights;
    use rlx_ir::op::MaskKind;

    let mut g = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    // (site, q, k, v, mask, num_heads, head_dim, mask_kind)
    let mut attns: Vec<(
        String,
        NodeId,
        NodeId,
        NodeId,
        Option<NodeId>,
        usize,
        usize,
        MaskKind,
    )> = Vec::new();

    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = g.add_node(node.op.clone(), inputs.clone(), node.shape.clone());
        id_map.insert(node.id, new_id);
        if let Op::Attention {
            num_heads,
            head_dim,
            mask_kind,
            ..
        } = &node.op
        {
            let site = match &node.name {
                Some(n) => format!("{n}#{}", node.id.0),
                None => format!("attn#{}", node.id.0),
            };
            let mask = if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
                inputs.get(3).copied()
            } else {
                None
            };
            attns.push((
                site, inputs[0], inputs[1], inputs[2], mask, *num_heads, *head_dim, *mask_kind,
            ));
        }
    }

    let mut outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
    let mut specs: Vec<StatSpec> = Vec::new();
    for (site, q, k, v, mask, nh, hd, mk) in attns {
        let mask_shape = mask.map(|m| g.shape(m).clone());
        let w = attention_softmax_weights(&mut g, q, k, v, nh, hd, mk, mask, mask_shape.as_ref());
        let rank = g.shape(w).rank(); // [B*H, S_q, S_k]
        let numel = g.shape(w).num_elements().unwrap_or(0);
        // Per-(head,query) peak attention mass.
        let (qmax, _) = reduce_axis(&mut g, w, rank - 1, ReduceOp::Max);
        specs.push(StatSpec {
            out_idx: outputs.len(),
            site: site.clone(),
            role: "attn",
            stat: "attn_qmax",
            len: g.shape(qmax).num_elements().unwrap_or(1),
            numel,
            flops: 0,
        });
        outputs.push(qmax);
        // Per-(head,key) received mass (sum over queries) — sinks.
        let (krecv, _) = reduce_axis(&mut g, w, rank - 2, ReduceOp::Sum);
        specs.push(StatSpec {
            out_idx: outputs.len(),
            site,
            role: "attn",
            stat: "attn_krecv",
            len: g.shape(krecv).num_elements().unwrap_or(1),
            numel,
            flops: 0,
        });
        outputs.push(krecv);
    }

    g.set_outputs(outputs);
    (g, specs)
}

// ─────────────────────────── synthetic data ───────────────────────────

/// Synthetic input distributions. Each is chosen to make a *different* sketch
/// pop, so the miner can be validated against ground truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dist {
    /// N(0,1) — the null hypothesis (dense, bell-shaped, no structure).
    Gaussian,
    /// U(-1,1) — dense, flat histogram.
    Uniform,
    /// 90% exact zeros — unstructured sparsity → sparse-GEMM candidate.
    Sparse90,
    /// Rank-4 product — low effective rank → factored-matmul candidate.
    LowRank,
    /// Gaussian with ~1% ×30 spikes → per-channel outliers → SmoothQuant/AWQ.
    Outlier,
    /// Gaussian snapped to 0.5 steps → spiky histogram → quantize/LUT.
    Quantized,
    /// Row random-walk — adjacent rows cohere → within-call sequence structure
    /// → delta-compute candidate (proves the `adj_sumsq` detector).
    Coherent,
}

impl Dist {
    pub const ALL: [Dist; 7] = [
        Dist::Gaussian,
        Dist::Uniform,
        Dist::Sparse90,
        Dist::LowRank,
        Dist::Outlier,
        Dist::Quantized,
        Dist::Coherent,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Dist::Gaussian => "gaussian",
            Dist::Uniform => "uniform",
            Dist::Sparse90 => "sparse90",
            Dist::LowRank => "lowrank",
            Dist::Outlier => "outlier",
            Dist::Quantized => "quantized",
            Dist::Coherent => "coherent",
        }
    }
}

#[inline]
fn u01(rng: &mut Philox4x32) -> f32 {
    rng.next_u32() as f32 / u32::MAX as f32
}

/// Generate a `rows × cols` matrix (row-major) drawn from `dist`, seeded for
/// reproducibility.
pub fn sample(dist: Dist, rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let n = rows * cols;
    let mut rng = Philox4x32::new(seed);
    let mut base = vec![0f32; n];
    rng.fill_normal(&mut base); // N(0,1) starting point for most dists
    let mut mask = Philox4x32::new(seed ^ 0x9E37_79B9_7F4A_7C15);

    match dist {
        Dist::Gaussian => base,
        Dist::Uniform => (0..n).map(|_| 2.0 * u01(&mut rng) - 1.0).collect(),
        Dist::Sparse90 => base
            .iter()
            .map(|&x| {
                if mask.next_u32().is_multiple_of(10) {
                    x
                } else {
                    0.0
                }
            })
            .collect(),
        Dist::LowRank => {
            let r = 4usize;
            let mut u = vec![0f32; rows * r];
            let mut v = vec![0f32; r * cols];
            rng.fill_normal(&mut u);
            mask.fill_normal(&mut v);
            let scale = 1.0 / (r as f32).sqrt();
            let mut out = vec![0f32; n];
            for i in 0..rows {
                for j in 0..cols {
                    let mut acc = 0f32;
                    for k in 0..r {
                        acc += u[i * r + k] * v[k * cols + j];
                    }
                    out[i * cols + j] = acc * scale;
                }
            }
            out
        }
        Dist::Outlier => base
            .iter()
            .map(|&x| {
                if mask.next_u32().is_multiple_of(100) {
                    x * 30.0
                } else {
                    x
                }
            })
            .collect(),
        Dist::Quantized => base.iter().map(|&x| (x * 2.0).round() / 2.0).collect(),
        Dist::Coherent => {
            // Row t = row t-1 + small step; adjacent rows stay close.
            let mut out = vec![0f32; n];
            let mut row = vec![0f32; cols];
            rng.fill_normal(&mut row); // row 0
            let mut step = vec![0f32; cols];
            for i in 0..rows {
                if i > 0 {
                    mask.fill_normal(&mut step);
                    for j in 0..cols {
                        row[j] += 0.15 * step[j];
                    }
                }
                out[i * cols..(i + 1) * cols].copy_from_slice(&row);
            }
            out
        }
    }
}

// ─────────────────────────── recorder ───────────────────────────

/// A dependency-free tidy-CSV sink: one row per sketch element. Columns:
/// `run_id,backend,dist,M,K,N,site,role,stat,idx,value`. Tidy/long form so
/// scalars (idx=0), per-channel vectors (idx=channel), and histograms
/// (idx=bin) all share one flat schema that pandas/polars can `groupby`.
pub struct Recorder<W: Write> {
    w: W,
}

impl Recorder<io::BufWriter<std::fs::File>> {
    /// Create a recorder writing to `path`, emitting the header row.
    pub fn create(path: &str) -> io::Result<Self> {
        let f = std::fs::File::create(path)?;
        Self::new(io::BufWriter::new(f))
    }
}

impl<W: Write> Recorder<W> {
    pub fn new(mut w: W) -> io::Result<Self> {
        // `numel` is the tapped tensor's own element count — density/normalization
        // use it directly so multi-site graphs (different shapes per matmul) work
        // without decoding M/K/N per role.
        writeln!(
            w,
            "run_id,step,backend,dist,M,K,N,numel,site,role,stat,idx,value"
        )?;
        Ok(Self { w })
    }

    /// Record one call's sketches. `run_id` groups calls into a sequence and
    /// `step` orders them within it (0 for independent runs); `outs` is
    /// `run()`'s output; `specs` the injection manifest.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        run_id: u64,
        step: u64,
        backend: &str,
        dist: &str,
        m: usize,
        k: usize,
        n: usize,
        specs: &[StatSpec],
        outs: &[Vec<f32>],
    ) -> io::Result<()> {
        for spec in specs {
            let data = &outs[spec.out_idx];
            let site = spec.site.replace(',', "_");
            for (idx, &val) in data.iter().enumerate() {
                writeln!(
                    self.w,
                    "{run_id},{step},{backend},{dist},{m},{k},{n},{},{site},{},{},{idx},{val}",
                    spec.numel, spec.role, spec.stat,
                )?;
            }
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }
}
