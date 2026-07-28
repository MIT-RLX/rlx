// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower an `rlx_ir::Graph` into a chain of MLX `Array` handles.
//!
//! Strategy is "fresh graph per run": every call rebuilds the MLX
//! graph from scratch using current input/param data.

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use rlx_ir::RegionPrologue;
use rlx_ir::op::{
    Activation, AdaNormKind, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp,
    RopeStyle, ScaleMode, SteKind, TransformStep,
};
use rlx_ir::shape::{Dim, DimBinding, Shape};
use rlx_ir::{DType, Graph, NodeId, Op};

use crate::array::{Array, MlxError, async_eval, eval};
use crate::ffi::{MlxMask, MlxReduce, MlxUnary};
use crate::ops;

mod env;
mod helpers;
mod host_eval;
mod subgraph;

pub use env::lower_with_env;
pub(crate) use helpers::*;
pub(crate) use subgraph::broadcast_leaf_data;
pub use subgraph::{build_leaf_for, expand_leaf_env, lower_subgraph};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlxMode {
    /// Eval after every op. Slower but useful for debugging — failures
    /// surface at the offending op rather than at the final eval.
    Eager,
    /// Build the full graph, eval all outputs in one shot. Default.
    /// Lets MLX's optimizer schedule the whole DAG.
    #[default]
    Lazy,
    /// Build the full graph and `async_eval` the outputs, but don't
    /// wait for completion. Used by `commit_no_wait` to amortize sync
    /// latency across pipelined runs.
    AsyncCommit,
    /// Compile the graph once via `mlx::compile` and replay the
    /// optimized trace on every subsequent `run()`. First call pays
    /// the trace cost; subsequent calls skip the per-run rebuild.
    Compiled,
}

/// What kind of host-side data each leaf node needs. Built once at
/// compile time; re-used at run time to materialize MLX leaves in the
/// same order across calls (essential for the mlx::compile path —
/// position determines which placeholder the compiled trace expects).
#[derive(Debug, Clone)]
pub enum LeafKey {
    Input(String),
    Param(String),
    Constant, // node id is implicit from leaf_order's NodeId
}

/// Walk `graph` in topo order and return the (NodeId, LeafKey) pairs
/// for every Input/Param/Constant node, in declaration order.
pub fn leaf_order(graph: &Graph) -> Vec<(NodeId, LeafKey)> {
    let mut out = Vec::new();
    for node in graph.nodes() {
        match &node.op {
            Op::Input { name } => out.push((node.id, LeafKey::Input(name.clone()))),
            Op::Param { name } => out.push((node.id, LeafKey::Param(name.clone()))),
            Op::Constant { .. } => out.push((node.id, LeafKey::Constant)),
            _ => {}
        }
    }
    out
}

/// Positional compile/run inputs: one slot per unique Input/Param name,
/// every Constant node. Graph builders often call `g.param("shared", …)`
/// once per block, producing many `NodeId`s with the same name; feeding
/// each as a separate mlx::compile leaf misbinds inputs on replay.
pub fn compile_leaf_order(graph: &Graph) -> Vec<(NodeId, LeafKey)> {
    let mut seen_input = HashSet::new();
    let mut seen_param = HashSet::new();
    let mut out = Vec::new();
    for node in graph.nodes() {
        match &node.op {
            Op::Input { name } if seen_input.insert(name.clone()) => {
                out.push((node.id, LeafKey::Input(name.clone())));
            }
            Op::Param { name } if seen_param.insert(name.clone()) => {
                out.push((node.id, LeafKey::Param(name.clone())));
            }
            Op::Constant { .. } => out.push((node.id, LeafKey::Constant)),
            _ => {}
        }
    }
    out
}

/// If `graph` contains an op whose MLX lowering eagerly evaluates a
/// tensor on the host (`to_f32` / `to_bytes`), return a short label
/// for the first offender. MLX's `mlx::compile` callback forbids
/// host eval; entering Compiled mode on such a graph triggers the
/// `[eval] Attempting to eval an array during function
/// transformations…` panic. Backends should check this up front and
/// fall back to Lazy.
pub fn first_host_eval_op(graph: &Graph) -> Option<&'static str> {
    for node in graph.nodes() {
        match &node.op {
            Op::DequantMatMul { scheme }
                if (scheme.is_gguf()
                    || matches!(scheme, rlx_ir::QuantScheme::Nvfp4Block)
                    || (!cfg!(feature = "native-mxfp")
                        && matches!(
                            scheme,
                            rlx_ir::QuantScheme::MlxMxfp4 { .. }
                                | rlx_ir::QuantScheme::MlxMxfp8 { .. }
                        ))) =>
            {
                return Some(if scheme.is_mlx() {
                    "DequantMatMul[MLX mxfp] (host dequant + cache)"
                } else {
                    "DequantMatMul[GGUF|NVFP4] (host dequant + cache)"
                });
            }
            Op::DequantGroupedMatMul { scheme } if scheme.is_gguf() => {
                return Some("DequantGroupedMatMul[GGUF] (host dequant)");
            }
            Op::DequantGroupedMatMulMlx { .. } => {
                return Some("DequantGroupedMatMulMlx (host dequant)");
            }
            Op::GaussianSplatRender { .. } => return Some("GaussianSplatRender (host kernel)"),
            Op::GaussianSplatRenderBackward { .. } => {
                return Some("GaussianSplatRenderBackward (host kernel)");
            }
            Op::LogMel | Op::LogMelBackward => return Some("LogMel (host filterbank)"),
            Op::WelchPeaks { .. } => return Some("WelchPeaks (host PSD top-K)"),
            Op::Custom { .. } => return Some("Custom (host kernel)"),
            Op::RngNormal { .. } | Op::RngUniform { .. } => {
                return Some("RngNormal/RngUniform (host fill)");
            }
            Op::Scan { .. } => return Some("Scan (host packed body)"),
            Op::ScanBackward { .. } | Op::ScanBackwardXs { .. } => {
                return Some("ScanBackward (host VJP loop)");
            }
            Op::ScatterNd { .. } => return Some("ScatterNd (host reference)"),
            Op::ScatterElements { .. } => return Some("ScatterElements (host reference)"),
            Op::GatherNd { .. } => return Some("GatherNd (host reference)"),
            Op::GatherElements { .. } => return Some("GatherElements (host reference)"),
            // Oversized CT2d (Vocos/ISTFT k≈1024) — MLX im2col OOMs; small
            // CT2d lowers natively via `ops::conv_transpose2d`.
            Op::ConvTranspose2d {
                kernel_size,
                groups,
                ..
            } if mlx_conv_im2col_too_large(graph, node, kernel_size, *groups) => {
                return Some("ConvTranspose2d (oversized im2col → host)");
            }
            // Oversized / grouped CT3d — MLX 3D transpose is groups=1 only;
            // oversized im2col still host-evals like forward Conv.
            Op::ConvTranspose3d { groups, .. } => {
                let w_shape = node_input_shape(graph, node.inputs[1]);
                if w_shape.len() >= 5 {
                    let kernel_size = [
                        w_shape[2].max(0) as usize,
                        w_shape[3].max(0) as usize,
                        w_shape[4].max(0) as usize,
                    ];
                    if *groups > 1 || mlx_conv_im2col_too_large(graph, node, &kernel_size, *groups)
                    {
                        return Some(if *groups > 1 {
                            "ConvTranspose3d (groups>1 → host)"
                        } else {
                            "ConvTranspose3d (oversized im2col → host)"
                        });
                    }
                }
            }
            Op::CustomFn { .. } => return Some("CustomFn (host body)"),
            Op::GaussianSplatPrepare { .. } | Op::GaussianSplatRasterize { .. } => {
                return Some("GaussianSplat prepare/rasterize (host)");
            }
            Op::ScaledQuantize { .. } => {
                return Some("ScaledQuantize (host typed encode)");
            }
            Op::ScaledMatMul {
                lhs_format,
                rhs_format,
                scale_layout,
                ..
            } if !helpers::scaled_fp8_mlx_ok(*lhs_format, *scale_layout)
                || !helpers::scaled_fp8_mlx_ok(*rhs_format, *scale_layout) =>
            {
                return Some("ScaledMatMul (non-PerTensor-FP8 → host typed)");
            }
            Op::ScaledQuantScale {
                format,
                scale_layout,
            }
            | Op::ScaledDequantize {
                format,
                scale_layout,
            } if !helpers::scaled_fp8_mlx_ok(*format, *scale_layout) => {
                return Some("ScaledQuantScale/Dequantize (non-PerTensor-FP8 → host typed)");
            }
            Op::BiMap
            | Op::ReEig { .. }
            | Op::LogEig { .. }
            | Op::SpdBatchNorm { .. }
            | Op::SpdKarcherMean { .. }
            | Op::ReEigBackward { .. }
            | Op::LogEigBackward { .. }
            | Op::SpdBatchNormBackwardX { .. }
            | Op::SpdBatchNormBackwardG { .. }
            | Op::SpdKarcherMeanWeighted { .. }
            | Op::SpdLogMap
            | Op::SpdExpMap
            | Op::SpdParallelTransport
            | Op::SpdMatrixFnBatch { .. }
            | Op::SpdLogMapBackward
            | Op::SpdExpMapBackward
            | Op::SpdParallelTransportBackward
            | Op::SpdMatrixFnBatchBackward { .. }
            | Op::Eigh
            | Op::EighBackward
            | Op::EighBatch
            | Op::EighBatchBackward => return Some("SPD/Eigh (host typed)"),
            Op::Interpolate3d { .. } => return Some("Interpolate3d (host typed)"),
            // Oversized ISTFT-as-conv (legacy decompose path) — MLX's conv1d
            // materializes a c_in·k·out_len im2col. Force Lazy + host naive.
            Op::Conv {
                kernel_size,
                groups,
                ..
            } if mlx_conv_im2col_too_large(graph, node, kernel_size, *groups) => {
                return Some("Conv (oversized im2col → host naive)");
            }
            // Only `carry = true` LSTM host-evals via `execute_lstm_f32`
            // (host `to_f32` inside the transform → forbids `mlx::compile`,
            // forces Lazy). `carry = false` (incl. BiLSTM / multi-layer) is
            // native on-device (`native_lstm`) and stays compilable.
            Op::Lstm { carry: true, .. } => return Some("Lstm carry (host execute_lstm_f32)"),
            Op::Gru { carry: true, .. } => return Some("Gru carry (host execute_gru_f32)"),
            Op::Rnn { carry: true, .. } => return Some("Rnn carry (host execute_rnn_f32)"),
            // A cast touching a complex dtype (source or dest) has no native
            // MLX astype (no complex dtype); it host-evaluates via
            // `mlx_cast_c64` / `mlx_cast_c128` (host readback), which is
            // forbidden inside `mlx::compile` — force Lazy. F64 casts stay
            // native (CPU-stream astype in the shim), so only complex needs
            // this.
            Op::Cast { to }
                if to.is_complex()
                    || node
                        .inputs
                        .first()
                        .is_some_and(|&i| graph.node(i).shape.dtype().is_complex()) =>
            {
                return Some("Cast[complex] (host complex cast)");
            }
            _ => {}
        }
    }
    None
}

/// Per-op-kind wall-time accumulators for `RLX_MLX_PROFILE=1`.
static MLX_PROFILE: std::sync::Mutex<Option<std::collections::BTreeMap<&'static str, (u64, u64)>>> =
    std::sync::Mutex::new(None);

fn mlx_profile_enabled() -> bool {
    std::env::var_os("RLX_MLX_PROFILE").is_some()
}

fn mlx_profile_kind(op: &Op) -> &'static str {
    // Coarse buckets — enough for RLX_MLX_PROFILE triage.
    match op.kind() {
        rlx_ir::OpKind::MatMul => "MatMul",
        rlx_ir::OpKind::Binary => "Binary",
        rlx_ir::OpKind::Activation => "Activation",
        rlx_ir::OpKind::Reduce => "Reduce",
        rlx_ir::OpKind::Scan => "Scan",
        rlx_ir::OpKind::ScanBackward | rlx_ir::OpKind::ScanBackwardXs => "ScanBackward",
        rlx_ir::OpKind::SelectiveScan => "SelectiveScan",
        rlx_ir::OpKind::Fft => "Fft",
        rlx_ir::OpKind::Conv => "Conv",
        _ => "Other",
    }
}

fn mlx_profile_record(kind: &'static str, ns: u64) {
    if !mlx_profile_enabled() {
        return;
    }
    let mut guard = MLX_PROFILE.lock().expect("mlx profile lock");
    let map = guard.get_or_insert_with(std::collections::BTreeMap::new);
    let e = map.entry(kind).or_insert((0, 0));
    e.0 += 1;
    e.1 = e.1.saturating_add(ns);
}

/// Print accumulated MLX lower/run timing when `RLX_MLX_PROFILE` is set.
pub fn mlx_profile_report() {
    if !mlx_profile_enabled() {
        return;
    }
    let guard = MLX_PROFILE.lock().expect("mlx profile lock");
    let Some(map) = guard.as_ref() else {
        eprintln!("[rlx-mlx] RLX_MLX_PROFILE: no lower samples recorded");
        return;
    };
    let mut rows: Vec<_> = map.iter().collect();
    rows.sort_by_key(|(_, (_, ns))| std::cmp::Reverse(*ns));
    eprintln!("[rlx-mlx] RLX_MLX_PROFILE (lower_with_env wall time):");
    for (kind, (n, ns)) in rows {
        eprintln!(
            "  {kind:16}  calls={n:6}  total_ms={:.3}  mean_us={:.1}",
            *ns as f64 / 1e6,
            (*ns as f64 / (*n as f64).max(1.0)) / 1e3
        );
    }
}

pub fn is_fusable(op: &Op) -> bool {
    matches!(
        op,
        Op::Binary(_)
            | Op::Activation(_)
            | Op::Compare(_)
            | Op::Cast { .. }
            | Op::Where
            | Op::Expand { .. }
            | Op::Fma
            | Op::Reshape { .. }
            | Op::Transpose { .. }
    )
}

/// Build the MLX graph and return the array handles for the graph's
/// declared outputs (in `graph.outputs` order).
///
/// Host-data variant: leaves are constructed from f32 input/param
/// buffers. The compile path uses [`lower_with_env`] directly with a
/// pre-built leaf binding instead.
pub fn lower_and_run(
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
    inputs: &HashMap<String, Vec<f32>>,
    mode: MlxMode,
) -> Result<Vec<Array>, MlxError> {
    // PLAN L3: coarse Perfetto span around the whole MLX lower+eval
    // pass. MLX is lazy (graph build → eval); per-node spans would
    // measure build time, not GPU compute. One span per run() is the
    // honest cross-backend marker for an MLX execution.
    let _perf = rlx_ir::perfetto::TraceSpan::new("lower_and_run", "mlx");
    lower_and_run_typed(
        graph,
        params,
        &HashMap::new(),
        inputs,
        &HashMap::new(),
        mode,
    )
}

/// Same as `lower_and_run` but accepts parallel typed maps. When a
/// name appears in `params_typed` / `inputs_typed`, the typed bytes
/// are bound directly via `Array::from_bytes` (no f32 round-trip).
/// Existing f32 callers thread empty maps through `lower_and_run`.
///
/// Dynamic shapes (`Dim::Dynamic`) get resolved here too: we infer
/// symbol→size bindings from the actual data lengths of each Input,
/// rebuild the graph with bound shapes, and lower against the
/// concretized version. MLX's per-shape trace caching handles the
/// re-shape efficiency on subsequent calls.
pub fn lower_and_run_typed(
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
    params_typed: &HashMap<String, (Vec<u8>, DType)>,
    inputs: &HashMap<String, Vec<f32>>,
    inputs_typed: &HashMap<String, (Vec<u8>, DType)>,
    mode: MlxMode,
) -> Result<Vec<Array>, MlxError> {
    lower_and_run_typed_with_extent(
        graph,
        params,
        params_typed,
        inputs,
        inputs_typed,
        mode,
        /*active_extent=*/ None,
        None,
        rlx_ir::RngOptions::default(),
    )
}

/// Variant of [`lower_and_run_typed`] honoring a PLAN L1 active-extent
/// hint (`Some((actual, upper))`). When set AND the graph passes
/// [`is_safe_for_active_extent`], every input leaf whose outer dim
/// equals `upper` is sliced along axis 0 to `actual` before
/// composition. MLX's lazy eval propagates the smaller shapes through
/// the rest of the trace, so most ops just produce smaller outputs
/// naturally — no per-op kernel scaling needed. Falls back to the full
/// extent when the hint is `None` or the graph contains an unsafe op.
pub fn lower_and_run_typed_with_extent(
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
    params_typed: &HashMap<String, (Vec<u8>, DType)>,
    inputs: &HashMap<String, Vec<f32>>,
    inputs_typed: &HashMap<String, (Vec<u8>, DType)>,
    mode: MlxMode,
    active_extent: Option<(usize, usize)>,
    gpu_inputs: Option<&HashMap<String, Array>>,
    rng: rlx_ir::RngOptions,
) -> Result<Vec<Array>, MlxError> {
    // Resolve dynamic dims if any. The graph as-given may have
    // Dim::Dynamic entries in Input shapes (and propagated through
    // inferred internal shapes). We gather concrete bindings from the
    // supplied data and rebuild the graph with every shape bound.
    let resolved_owner;
    let graph: &Graph = if has_dynamic_dims(graph) {
        let binding = collect_bindings(graph, inputs, inputs_typed)?;
        resolved_owner = resolve_graph(graph, &binding);
        &resolved_owner
    } else {
        graph
    };

    let order = compile_leaf_order(graph);
    let mut env: HashMap<NodeId, Array> = HashMap::with_capacity(graph.nodes().len());
    for (id, _key) in &order {
        env.insert(
            *id,
            build_leaf_for(
                graph,
                *id,
                params,
                inputs,
                params_typed,
                inputs_typed,
                gpu_inputs,
            )?,
        );
    }
    env = expand_leaf_env(graph, env)?;

    // PLAN L1 active-extent: when hinted + safe, slice each Input leaf
    // along axis 0 from `upper` to `actual`. Only Input leaves get
    // sliced — Param/Constant tensors don't carry a batch dim that
    // matches the bucket axis. MLX's lazy graph propagates the smaller
    // shapes naturally through downstream element-wise / reduction-on-
    // inner / matmul ops.
    if let Some((actual, upper)) = active_extent
        && actual < upper
        && is_safe_for_active_extent(graph, upper)
    {
        for (id, _key) in &order {
            let node = graph.node(*id);
            if !matches!(node.op, Op::Input { .. }) {
                continue;
            }
            let dims = node.shape.dims();
            if dims.is_empty() {
                continue;
            }
            let outer = match dims[0] {
                Dim::Static(d) => d,
                _ => continue,
            };
            if outer != upper {
                continue;
            }
            let leaf = env.get(id).unwrap();
            let in_shape: Vec<usize> = dims.iter().map(|d| d.unwrap_static()).collect();
            let mut start = vec![0i32; in_shape.len()];
            let mut stop: Vec<i32> = in_shape.iter().map(|&d| d as i32).collect();
            start[0] = 0;
            stop[0] = actual as i32;
            let sliced = ops::slice(leaf, &start, &stop)?;
            env.insert(*id, sliced);
        }
    }

    // Eager mode wants per-op eval for debugging; the env-walker's
    // construction is pure (no eval), so we trigger it here against
    // outputs after lowering. For interleaved per-op eval we'd need
    // a separate walker variant — currently no caller asks for that.
    let outs = lower_with_env(graph, env, params, params_typed, rng, true)?;

    let refs: Vec<&Array> = outs.iter().collect();
    match mode {
        MlxMode::Eager => {
            // Eval outputs one at a time. Functionally equivalent to
            // per-op eval since outputs are dependency roots; only
            // the failure-localization aspect is weaker.
            for o in &outs {
                eval(&[o])?;
            }
        }
        MlxMode::Lazy => {
            for (i, o) in refs.iter().enumerate() {
                let oid = graph.outputs.get(i).copied();
                let name = oid
                    .and_then(|id| graph.node(id).name.clone())
                    .unwrap_or_else(|| format!("{oid:?}"));
                eval(&[*o]).map_err(|e| MlxError(format!("eval output[{i}] {name}: {e}")))?;
            }
        }
        MlxMode::AsyncCommit => {
            async_eval(&refs)?;
        }
        MlxMode::Compiled => {
            // Compiled mode shouldn't reach this code path —
            // backend.rs dispatches to run_compiled before calling
            // here. If we did get here it means the host-data path
            // was used, so just eval normally (correct, just misses
            // the trace-cache benefit).
            eval(&refs)?;
        }
    }

    Ok(outs)
}

/// PLAN L1 — true when the graph is safe for active-extent dispatch
/// at the given `upper` extent. Conservative: rejects ops that either
/// (a) hardcode the outer dim in their parameters
/// (`Op::Reshape { new_shape }` / `Op::Expand { target_shape }` / etc.
/// when those shapes mention `upper`), (b) operate along axis 0
/// (`Op::Reduce` / `Op::Cumsum` / `Op::Concat` / `Op::Narrow` with
/// axis 0; `Op::Transpose` whose perm permutes axis 0), or (c) have
/// outer-dim semantics that can't be honored by simply slicing the
/// input (`Op::Gather` / `Op::ScatterAdd` / `Op::Sample` / `Op::TopK`
/// / `Op::SelectiveScan` / `Op::GroupedMatMul` / `Op::Pool` /
/// `Op::Conv` / `Op::FusedTransformerLayer` / sub-graph control flow).
pub fn is_safe_for_active_extent(graph: &Graph, upper: usize) -> bool {
    let upper_i64 = upper as i64;
    for node in graph.nodes() {
        match &node.op {
            // Leaves & element-wise ops: always safe (slicing inputs
            // produces correctly-sized intermediates via lazy eval).
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => {}
            Op::Activation(_)
            | Op::Cast { .. }
            | Op::Binary(_)
            | Op::Compare(_)
            | Op::Where
            | Op::ElementwiseRegion { .. }
            | Op::BatchElementwiseRegion { .. }
            | Op::TransformRegion { .. } => {}
            // Per-row normalizations: operate on inner axes, batch is
            // pass-through. Safe.
            Op::Softmax { axis: _ }
            | Op::LayerNorm { .. }
            | Op::LayerNorm2d { .. }
            | Op::GroupNorm { .. }
            | Op::BatchNormInference { .. }
            | Op::RmsNorm { .. }
            | Op::ResizeNearest2x => {}
            // Rope / Attention / matmul: batch in outer dim, computation
            // on inner axes. Safe by construction.
            Op::Rope { .. }
            | Op::Attention { .. }
            | Op::MatMul
            | Op::DotGeneral { .. }
            | Op::FusedMatMulBiasAct { .. }
            | Op::FusedSwiGLU { .. }
            | Op::FusedResidualLN { .. }
            | Op::FusedResidualRmsNorm { .. }
            | Op::AdaLayerNorm { .. }
            | Op::GatedResidual
            | Op::AdaLayerNormBackward { .. }
            | Op::GatedResidualBackward
            | Op::FusedAttentionBlock { .. } => {}
            // DequantMatMul / LoraMatMul follow MatMul's batch-outer
            // contract.
            Op::DequantMatMul { .. } | Op::LoraMatMul { .. } => {}
            // Real INT8 ops lower natively via f32 GEMM/conv + requant, but
            // active-extent bucketing is still unsafe (full-tensor zp/mult).
            Op::QMatMul { .. } | Op::QConv2d { .. } => return false,
            // Reduce / Cumsum: safe iff the operation doesn't touch
            // axis 0.
            Op::Reduce { axes, .. } => {
                if axes.contains(&0) {
                    return false;
                }
            }
            Op::Cumsum { axis, .. } => {
                if *axis == 0 {
                    return false;
                }
            }
            // Concat: safe iff axis != 0 (concatenating along the batch
            // axis would mix batches across the slice boundary).
            Op::Concat { axis } => {
                if *axis == 0 {
                    return false;
                }
            }
            // Narrow on axis 0 changes the bucket itself — unsafe.
            Op::Narrow { axis, .. } => {
                if *axis == 0 {
                    return false;
                }
            }
            // Transpose is safe iff perm[0] == 0 (axis 0 stays put;
            // inner axes can permute freely).
            Op::Transpose { perm } => {
                if perm.first().copied() != Some(0) {
                    return false;
                }
            }
            // Reshape / Expand: reject if their target shape mentions
            // `upper` — that hardcoded dim won't survive the slice.
            Op::Reshape { new_shape } => {
                if new_shape.contains(&upper_i64) {
                    return false;
                }
            }
            Op::Expand { target_shape } => {
                if target_shape.contains(&upper_i64) {
                    return false;
                }
            }
            // Gather operates on axis 0 of its lookup table; the
            // batch contract isn't compatible with bucket slicing.
            Op::Gather { .. } => return false,
            // Conservatively unsafe — these have batch-touching
            // semantics (or sub-graph leaves) that the slice trick
            // doesn't handle.
            Op::ScatterAdd
            | Op::ScatterNd { .. }
            | Op::ScatterElements { .. }
            | Op::GatherNd { .. }
            | Op::GatherElements { .. }
            | Op::Sample { .. }
            | Op::RngNormal { .. }
            | Op::RngUniform { .. }
            | Op::TopK { .. }
            | Op::SelectiveScan { .. }
            | Op::GatedDeltaNet { .. }
            | Op::GroupedMatMul
            | Op::Pool { .. }
            | Op::Conv { .. }
            | Op::ConvTranspose2d { .. }
            | Op::FusedTransformerLayer { .. }
            | Op::DenseSolve
            // Full-matrix host-staged linalg — batch-bucket slicing unsafe.
            | Op::Cholesky
            | Op::TriangularSolve { .. }
            | Op::Det
            | Op::LogDet
            // Sort / ArgSort reorder along an axis — bucket slicing unsafe.
            | Op::Sort { .. } | Op::Svd { .. } | Op::Qr { .. }
            | Op::ArgSort { .. }
            | Op::Custom { .. }
            | Op::If { .. }
            | Op::While { .. } => return false,
            // Quantize/Dequantize/LSQ/FakeQuantize lower natively (`fq_*` /
            // INT8 helpers) but reduce / broadcast over the full tensor, so
            // active-extent bucketing is unsafe.
            Op::Quantize { .. }
            | Op::Dequantize { .. }
            | Op::FakeQuantize { .. }
            | Op::FakeQuantizeBackward { .. }
            | Op::FakeQuantizeLSQ { .. }
            | Op::FakeQuantizeLSQBackwardX { .. }
            | Op::FakeQuantizeLSQBackwardScale { .. } => return false,
            // Backward / training ops: active-extent dispatch is an
            // inference-only batch-bucketing optimization, so the safe
            // default for any training-graph node is `false` regardless
            // of whether MLX can lower it. Tier 1 (Relu/Activation/SCE/
            // LayerNorm/RmsNorm/Rope/Cumsum/Gather backward) DOES lower
            // on MLX — see `lower_with_env`.
            Op::ReluBackward
            | Op::ActivationBackward { .. }
            | Op::MaxPool2dBackward { .. }
            | Op::Conv2dBackwardInput { .. }
            | Op::Conv2dBackwardWeight { .. }
            | Op::MaxPool3dBackward { .. }
            | Op::Conv3dBackwardInput { .. }
            | Op::Conv3dBackwardWeight { .. }
            | Op::SoftmaxCrossEntropyWithLogits
            | Op::SoftmaxCrossEntropyBackward
            | Op::LayerNormBackwardInput { .. }
            | Op::LayerNormBackwardGamma { .. }
            | Op::RmsNormBackwardInput { .. }
            | Op::RmsNormBackwardGamma { .. }
            | Op::RmsNormBackwardBeta { .. }
            | Op::RopeBackward { .. }
            | Op::CumsumBackward { .. }
            | Op::GatherBackward { .. }
            | Op::GroupNormBackwardInput { .. }
            | Op::GroupNormBackwardGamma { .. }
            | Op::GroupNormBackwardBeta { .. }
            | Op::BatchNormInferenceBackwardInput { .. }
            | Op::BatchNormInferenceBackwardGamma { .. }
            | Op::BatchNormInferenceBackwardBeta => return false,
            Op::Scan { .. }
            | Op::ScanBackward { .. }
            | Op::ScanBackwardXs { .. }
            | Op::BatchedDenseSolve => return false,
            // CustomFn is opaque to active-extent analysis — the body
            // graph may have arbitrary internal structure. Fall back
            // to full extent for graphs that contain them. (Op::Custom
            // is already rejected in the conservatively-unsafe arm.)
            Op::CustomFn { .. } => return false,
            // FFT lowered natively via `mlx::fft::fft` FFI shim.
            Op::Fft { .. } => return true,
            // C64 ops lower via interleaved f32 even/odd lanes; still unsafe
            // for active-extent (flat interleaved layout).
            Op::ComplexNormSq | Op::ComplexNormSqBackward | Op::Conjugate => return false,
            Op::Conv3d { .. }
            | Op::ConvTranspose3d { .. }
            | Op::FusedConvBiasAct { .. }
            | Op::PartitionedConv { .. }
            | Op::AxialRope2d { .. }
            | Op::Mamba2 { .. }
            | Op::FftButterflyStage { .. } => return false,
            _ => return false,
        }
    }
    true
}

/// True if any node in the graph has a Dim::Dynamic entry. Cheap
/// scan; lets us skip the resolve step for fully-static graphs.
fn has_dynamic_dims(graph: &Graph) -> bool {
    graph
        .nodes()
        .iter()
        .any(|n| n.shape.dims().iter().any(|d| !d.is_static()))
}

/// Walk the graph, infer concrete sizes for each `Dim::Dynamic` symbol
/// from the supplied input data. Each Input with exactly one dynamic
/// dim contributes a binding (data_nelems / static_dim_product). The
/// inference is conservative: if a single input has multiple dynamic
/// dims it errors, since the data length is one number and we can't
/// distribute it across multiple unknowns. Multi-dynamic inputs would
/// need an externally-supplied DimBinding; out of scope today.
fn collect_bindings(
    graph: &Graph,
    inputs: &HashMap<String, Vec<f32>>,
    inputs_typed: &HashMap<String, (Vec<u8>, DType)>,
) -> Result<DimBinding, MlxError> {
    let mut binding = DimBinding::new();
    for node in graph.nodes() {
        if let Op::Input { name } = &node.op {
            // Element count from the supplied data (typed wins).
            let n_elems = if let Some((bytes, dt)) = inputs_typed.get(name) {
                let elem_size = dt.size_bytes();
                if elem_size == 0 || bytes.len() % elem_size != 0 {
                    return Err(MlxError(format!(
                        "Input '{name}': typed bytes len {} not aligned to dtype size",
                        bytes.len()
                    )));
                }
                bytes.len() / elem_size
            } else if let Some(data) = inputs.get(name) {
                data.len()
            } else {
                // No data yet — skip; the leaf-build step will error
                // with a clearer "missing input" diagnostic.
                continue;
            };

            // Walk the shape's dims, accumulating the static product
            // and identifying the (single allowed) dynamic position.
            let mut static_prod: usize = 1;
            let mut dynamic_sym: Option<u32> = None;
            for d in node.shape.dims().iter() {
                match d {
                    Dim::Static(n) => {
                        static_prod = static_prod.checked_mul(*n).ok_or_else(|| {
                            MlxError(format!("Input '{name}': static dim product overflow"))
                        })?;
                    }
                    Dim::Dynamic(sym) => {
                        if dynamic_sym.is_some() {
                            return Err(MlxError(format!(
                                "Input '{name}' has multiple dynamic dims; \
                                 explicit DimBinding required"
                            )));
                        }
                        dynamic_sym = Some(*sym);
                    }
                }
            }

            if let Some(sym) = dynamic_sym {
                if static_prod == 0 {
                    return Err(MlxError(format!(
                        "Input '{name}': can't infer dynamic dim against zero \
                         static product"
                    )));
                }
                if n_elems % static_prod != 0 {
                    return Err(MlxError(format!(
                        "Input '{name}': nelems {n_elems} not divisible by \
                         static dim product {static_prod}"
                    )));
                }
                let dim_size = n_elems / static_prod;
                if let Some(prev) = binding.get(sym) {
                    if prev != dim_size {
                        return Err(MlxError(format!(
                            "Dynamic dim ?{sym}: inconsistent values across \
                             inputs ({prev} vs {dim_size})"
                        )));
                    }
                } else {
                    binding.set(sym, dim_size);
                }
            }
        }
    }
    Ok(binding)
}

/// Rebuild the graph with every Shape bound against `binding`. Node
/// IDs are preserved because we re-add ops in the same order via the
/// public `Graph::add_node` API (which allocates IDs sequentially).
fn resolve_graph(graph: &Graph, binding: &DimBinding) -> Graph {
    let mut fresh = Graph::new(&graph.name);
    for node in graph.nodes() {
        let bound: Shape = node.shape.bind(binding);
        // add_node preserves declaration order → preserves NodeIds.
        fresh.add_node(node.op.clone(), node.inputs.clone(), bound);
    }
    fresh.set_outputs(graph.outputs.clone());
    fresh
}
