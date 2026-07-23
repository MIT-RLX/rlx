// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// The executable: ties lowering + packaging + CoreML execution together.
// Exposes an inherent API (compile / set_param / finalize / run) — the
// `rlx_runtime::Backend` trait impl lives in rlx-runtime to avoid a
// dependency cycle (same split rlx-metal uses).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rlx_ir::{DType, Graph, Op, OpKind, Shape};

use crate::ffi::CoremlModel;
use crate::host_exec::run_host_node;
use crate::hybrid::{self, ExecutionPlan, MilSegment, Segment};
use crate::mil::bytes_to_f32;
use crate::mil::{LowerOptions, LoweredProgram, TypedParams, lower_graph_with_options};
use crate::{ChipInfo, ComputeUnits, CoremlError, Result};

struct MilSlot {
    graph: Graph,
    lowered: Option<LoweredProgram>,
    model: Option<CoremlModel>,
    pkg_dir: Option<PathBuf>,
}

/// Content hash of the serialized model + weight blob, used as the
/// compiled-model cache key. SipHash (std) — collisions are negligible for
/// this and it needs no extra dependency.
fn content_hash(proto: &[u8], blob: &[u8]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    proto.hash(&mut h);
    blob.hash(&mut h);
    format!("{:016x}", h.finish())
}

static PKG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A compiled CoreML graph.
///
/// CoreML bakes weights into the model at build time, so the lifecycle is:
/// [`compile`](Self::compile) → [`set_param`](Self::set_param) for each
/// weight → [`finalize`](Self::finalize) (writes the `.mlpackage` and
/// loads it) → [`run`](Self::run). `run` auto-finalizes on first call.
pub struct CoremlExecutable {
    graph: Graph,
    params: HashMap<String, Vec<f32>>,
    typed_params: TypedParams,
    compute_units: ComputeUnits,
    lower_opts: LowerOptions,
    plan: ExecutionPlan,
    mil_slots: Vec<MilSlot>,
}

/// CoreML's ML Program is value-typed with no I64 storage and strict per-op type
/// rules, unlike the f32-uniform CPU/wgpu arenas. Rather than special-case every op,
/// promote the graph to an f32 flow once: rewrite every integer tensor (I64/I32/U32/
/// I16/I8/U8) to F32 — node output dtypes, integer `Constant` data, and
/// `Cast { to: int }` targets. `Bool` is preserved (CoreML `select`/logical ops need
/// it) and floats are untouched. Integer-only consumers (e.g. `gather` indices) cast
/// back to int32 in the MIL lowering. This makes index/shape arithmetic flow as exact
/// integer-valued f32. `I16` is included here (CoreML has no int16 storage) so it
/// mirrors the other ints; the `mil_cast_dtype`/`mil_data_type` widening of I16→int32
/// is the fallback for the direct `lower_graph` path that skips this promotion.
fn promote_int_to_f32(graph: &mut Graph) {
    fn is_int(dt: DType) -> bool {
        matches!(
            dt,
            DType::I64 | DType::I32 | DType::U32 | DType::I16 | DType::I8 | DType::U8
        )
    }
    for node in graph.nodes_mut() {
        let dt = node.shape.dtype();
        if is_int(dt) {
            if let Op::Constant { data } = &mut node.op {
                let floats: Vec<f32> = match dt {
                    DType::I64 => data
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect(),
                    DType::I32 => data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect(),
                    DType::U32 => data
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect(),
                    DType::I16 => data
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect(),
                    DType::U8 => data.iter().map(|&b| b as f32).collect(),
                    DType::I8 => data.iter().map(|&b| (b as i8) as f32).collect(),
                    _ => unreachable!(),
                };
                *data = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
            }
        }
        if let Op::Cast { to } = &mut node.op {
            if is_int(*to) {
                *to = DType::F32;
            }
        }
        if is_int(dt) {
            node.shape = node.shape.clone().with_dtype(DType::F32);
        }
    }
}

/// CoreML/MIL has no complex storage. Rewrite every `C64` tensor to interleaved
/// F32 by doubling the last axis (`[…, n] C64` → `[…, 2n] F32`). Byte payloads
/// for `Constant` are already interleaved re/im pairs, so they are left as-is.
/// `Cast { to: C64 }` becomes `Cast { to: F32 }` (shape rewrite covers the
/// element count). Must run before hybrid planning so complex ops land in MIL.
pub fn promote_c64_to_interleaved_f32(graph: &mut Graph) {
    for node in graph.nodes_mut() {
        if let Op::Cast { to } = &mut node.op {
            if *to == DType::C64 {
                *to = DType::F32;
            }
        }
        if node.shape.dtype() == DType::C64 {
            let mut dims: Vec<usize> = node
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            if dims.is_empty() {
                dims.push(2);
            } else {
                let last = dims.len() - 1;
                dims[last] = dims[last].saturating_mul(2);
            }
            node.shape = Shape::new(&dims, DType::F32);
        }
    }
}

/// CoreML/MIL has no f64 or bf16 storage. Demote every such tensor to the
/// nearest supported float once, before lowering: `F64 → F32`, `BF16 → F16`.
/// Rewrites node output dtypes, float `Constant` payloads (re-encoded to the
/// target width), and `Cast { to: F64 | BF16 }` targets — mirroring
/// `promote_int_to_f32`. This is required (not just cosmetic): CoreML rejects
/// an f64 MLMultiArray model output outright, so the demotion must reach the
/// graph's output node dtypes, which the per-op MIL type mappers alone don't.
fn demote_unsupported_floats(graph: &mut Graph) {
    for node in graph.nodes_mut() {
        let dt = node.shape.dtype();
        if let Op::Constant { data } = &mut node.op {
            match dt {
                DType::F64 => {
                    let f32s: Vec<f32> = data
                        .chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect();
                    *data = f32s.iter().flat_map(|f| f.to_le_bytes()).collect();
                }
                DType::BF16 => {
                    // bf16 (2 bytes) → f16 (2 bytes), value-preserving via f32.
                    *data = data
                        .chunks_exact(2)
                        .flat_map(|c| {
                            let bf = u16::from_le_bytes(c.try_into().unwrap());
                            let f = f32::from_bits((bf as u32) << 16);
                            half::f16::from_f32(f).to_le_bytes()
                        })
                        .collect();
                }
                _ => {}
            }
        }
        if let Op::Cast { to } = &mut node.op {
            match *to {
                DType::F64 => *to = DType::F32,
                DType::BF16 => *to = DType::F16,
                _ => {}
            }
        }
        match dt {
            DType::F64 => node.shape = node.shape.clone().with_dtype(DType::F32),
            DType::BF16 => node.shape = node.shape.clone().with_dtype(DType::F16),
            _ => {}
        }
    }
}

/// When compiling at F16, demote float tensor dtypes so MIL uses fp16 storage.
fn demote_float_to_f16(graph: &mut Graph) {
    for node in graph.nodes_mut() {
        if node.shape.dtype() == DType::F32 {
            node.shape = node.shape.clone().with_dtype(DType::F16);
        }
        if let Op::Cast { to } = &mut node.op {
            if *to == DType::F32 {
                *to = DType::F16;
            }
        }
    }
}

pub fn default_lower_options(graph: &Graph) -> LowerOptions {
    // RLX_COREML_F16=1 stores activations (and the dequant output) in f16, ~half
    // the RAM + half the BNNS-compile working set. Needed for 27B-class models
    // where the f32 decode-graph IR overruns disk during coremlc/BNNS compile.
    let float_dtype = if std::env::var("RLX_COREML_F16").ok().as_deref() == Some("1") {
        DType::F16
    } else {
        DType::F32
    };
    LowerOptions {
        float_dtype,
        flexible_inputs: rlx_ir::dynamic::has_dynamic_dims(graph)
            || std::env::var("RLX_COREML_FLEXIBLE_INPUTS").ok().as_deref() == Some("1"),
        ondevice_dequant: std::env::var("RLX_COREML_HOST_DEQUANT").ok().as_deref() != Some("1"),
        q1_mode: None, // None → RLX_COREML_Q1_MODE env (default Lut)
    }
}

/// Whether `graph` computes gradients — it carries a loss-cotangent seed input
/// (`grad_with_loss` names it `d_output`) or still contains a native `*Backward`
/// op (RMSNorm / MaxPool2d, which CoreML lowers directly).
///
/// Kept for callers / tests that want to classify training graphs; compute-unit
/// policy no longer branches on it because **all** fp32 graphs use CPU+GPU
/// (see [`default_compute_units`]).
#[allow(dead_code)]
fn is_backward_graph(graph: &Graph) -> bool {
    graph.nodes().iter().any(|n| {
        if let Op::Input { name } = &n.op {
            if name == "d_output" {
                return true;
            }
        }
        matches!(
            n.op.kind(),
            OpKind::ReluBackward
                | OpKind::ActivationBackward
                | OpKind::RmsNormBackwardInput
                | OpKind::RmsNormBackwardGamma
                | OpKind::RmsNormBackwardBeta
                | OpKind::LayerNormBackwardInput
                | OpKind::LayerNormBackwardGamma
                | OpKind::MaxPool2dBackward
                | OpKind::SoftmaxCrossEntropyBackward
                | OpKind::AttentionBackward
                | OpKind::Conv2dBackwardInput
                | OpKind::Conv2dBackwardWeight
        )
    })
}

/// CoreML compute units — the speed↔precision knob for training, the natural
/// partner of the Automatic-Floating-Point policy. The Neural Engine runs
/// **f16+f16** (fast); CPU/GPU run **fp32** (precise). So:
///
/// - `RLX_COREML_UNITS=cpu|gpu|all|ane` overrides explicitly;
/// - **an f16 graph → Neural Engine** — the caller chose low precision via an AFP
///   policy (`AutoMixedPrecision` makes compute ops f16), so run it fast on the
///   ANE (training included);
/// - **an fp32 backward/training graph → `CpuAndGpu`** for gradient precision (the
///   ANE would silently downcast fp32 gradients to f16);
/// - **fp32 inference → `CpuAndGpu`** — Apple's on-device BNNS AOT path
///   (`bnns::GraphCompile`) SIGSEGVs on many large imported graphs (TTS CFM /
///   VITS / DiT, multi-k-node ONNX). CoreML GPU units compile the same MIL
///   cleanly. Opt into ANE with `RLX_COREML_UNITS=ane` (or ship an f16 graph).
///
/// Precision↔speed is the user's choice: default fp32 (CPU+GPU, accurate /
/// BNNS-safe) or an AFP f16 policy / `RLX_COREML_UNITS=ane` (ANE, fast).
pub fn default_compute_units(graph: &Graph) -> ComputeUnits {
    match std::env::var("RLX_COREML_UNITS").as_deref() {
        Ok("cpu") => ComputeUnits::CpuOnly,
        Ok("gpu") => ComputeUnits::CpuAndGpu,
        Ok("all") => ComputeUnits::All,
        Ok("ane") => ComputeUnits::CpuAndNeuralEngine,
        _ if graph_has_f16(graph) => ComputeUnits::CpuAndNeuralEngine,
        // Host-split graphs (Op::Scan-family: SPD eigensolvers, IIR/SSM
        // recurrences) alternate host and MIL segments; EACH MIL segment
        // compiles its own CoreML model, and the GPU/Metal compile cost per
        // segment (~20s even for a tiny segment) dwarfs any GPU speedup — the
        // GPU can't touch the host-scan portion at all. CPU-only compute units
        // skip the Metal-path compile entirely: spdnet/coreml 64s → 1s with no
        // runtime loss. (A size-based CpuAndGpu carve-out for large-batch nets
        // was tried but gave no stable win — the GPU-compile-vs-matmul tradeoff
        // is noise-dominated for these graphs.) `RLX_COREML_UNITS` overrides.
        _ if graph_has_host_split(graph) => ComputeUnits::CpuOnly,
        // fp32 inference + training: CPU+GPU. Avoids Neural-Engine BNNS crashes
        // on large ML Programs; matches the precision path documented above.
        _ => ComputeUnits::CpuAndGpu,
    }
}

/// True if the graph contains a recurrence/scan — either host-split ops
/// (`Op::Scan`-family that run on the host between MIL segments) OR an `Op::Scan`
/// now lowered natively to an on-device `while_loop`. Both cases run faster
/// CPU-only on CoreML: the host-split form pays a ~20s Metal compile per segment,
/// and the native `while_loop` pays a GPU compile + per-iteration dispatch that
/// the CPU avoids (tensorcspnet 97s CpuAndGpu → 68s CpuOnly). See the host-split
/// arm in [`default_compute_units`].
fn graph_has_host_split(graph: &Graph) -> bool {
    graph
        .nodes()
        .iter()
        .any(|n| crate::host_exec::is_host_node(graph, n.id) || matches!(n.op, Op::Scan { .. }))
}

/// True if any node carries an f16 tensor — the signature of an
/// `AutoMixedPrecision` / fp16 graph (the caller traded precision for speed).
fn graph_has_f16(graph: &Graph) -> bool {
    graph.nodes().iter().any(|n| n.shape.dtype() == DType::F16)
}

impl CoremlExecutable {
    /// Stage a graph for CoreML execution under [`default_compute_units`].
    pub fn compile(graph: Graph) -> Self {
        let units = default_compute_units(&graph);
        Self::compile_with_units(graph, units)
    }

    /// Stage a graph with an explicit compute-unit policy.
    pub fn compile_with_units(graph: Graph, compute_units: ComputeUnits) -> Self {
        let opts = default_lower_options(&graph);
        Self::compile_with_options(graph, compute_units, opts)
    }

    /// Stage with lowering options; compute units follow [`default_compute_units`].
    pub fn compile_with_lower_opts(graph: Graph, lower_opts: LowerOptions) -> Self {
        let units = default_compute_units(&graph);
        Self::compile_with_options(graph, units, lower_opts)
    }

    /// Stage a graph with compute units and lowering options (precision, flex shapes).
    pub fn compile_with_options(
        mut graph: Graph,
        compute_units: ComputeUnits,
        mut lower_opts: LowerOptions,
    ) -> Self {
        // Expand control flow + DotGeneral + elementwise regions before hybrid
        // planning so claimed OpKinds that have no MIL / CPU HostOp path become
        // primitives CoreML already lowers (or host-evals).
        {
            use rlx_opt::pass::Pass as _;
            graph = rlx_opt::LowerControlFlow.run(graph);
            graph = rlx_opt::LowerDotGeneral.run(graph);
            graph = rlx_opt::UnfuseElementwiseRegions::FOR_CPU.run(graph);
        }
        // FusedConvBiasAct / PartitionedConv / FusedTransformerLayer: CPU would
        // Nop them under HostOp — expand to primitives first.
        graph = crate::hybrid::lower_cpu_nop_fused_for_coreml(graph);
        // `FusedAttentionBlock` is a claimed op (so it legalizes and the
        // fusion pipeline may emit it), but the MIL lowering has no
        // fused-attention op — decompose it to the primitive chain
        // (matmul → narrow → rope → attention → matmul) that CoreML lowers.
        // FAB-only so native LoraMatMul / FusedSwiGLU-style ops are kept.
        // No-op when no FAB node is present.
        graph = rlx_opt::unfuse::unfuse_attention_block(graph);
        promote_int_to_f32(&mut graph);
        // C64 → interleaved F32 before hybrid plan (MIL has no complex dtype).
        promote_c64_to_interleaved_f32(&mut graph);
        // CoreML has no f64/bf16 — demote to f32/f16 before the optional
        // whole-graph f16 pass (so F64→F32→F16 stays consistent under F16 mode).
        demote_unsupported_floats(&mut graph);
        if lower_opts.float_dtype == DType::F16 {
            demote_float_to_f16(&mut graph);
        }
        lower_opts.flexible_inputs =
            lower_opts.flexible_inputs || rlx_ir::dynamic::has_dynamic_dims(&graph);

        let plan = hybrid::plan_execution(&graph).unwrap_or_else(|e| {
            panic!("CoreML hybrid plan failed: {e}");
        });
        let mil_slots = mil_slots_for_plan(&graph, &plan);
        CoremlExecutable {
            graph,
            params: HashMap::new(),
            typed_params: TypedParams::new(),
            compute_units,
            lower_opts,
            plan,
            mil_slots,
        }
    }

    /// Clone the staged graph + params for the runtime's per-`(component,device,len)`
    /// graph cache. The built MLModel (an FFI handle) isn't cloned — the copy
    /// re-finalizes lazily on first run. The graph is already int→f32 promoted.
    pub fn clone_for_cache(&self) -> Self {
        CoremlExecutable {
            graph: self.graph.clone(),
            params: self.params.clone(),
            typed_params: self.typed_params.clone(),
            compute_units: self.compute_units,
            lower_opts: self.lower_opts,
            plan: self.plan.clone(),
            mil_slots: self
                .mil_slots
                .iter()
                .map(|s| MilSlot {
                    graph: s.graph.clone(),
                    lowered: None,
                    model: None,
                    pkg_dir: None,
                })
                .collect(),
        }
    }

    pub fn lower_options(&self) -> LowerOptions {
        self.lower_opts
    }

    /// Provide the f32 weights for an IR `Param`. Must be called for every
    /// parameter before [`finalize`](Self::finalize)/[`run`](Self::run).
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.params.insert(name.to_string(), data.to_vec());
        self.invalidate_models();
    }

    /// Provide non-f32 (e.g. GGUF-quantized) weight bytes for an IR
    /// `Param`. Quantized weights are dequantized or constexpr-expanded at
    /// lowering time depending on [`LowerOptions::ondevice_dequant`].
    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: DType) {
        if dtype == DType::F32 {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            self.set_param(name, &floats);
            return;
        }
        self.typed_params
            .insert(name.to_string(), (data.to_vec(), dtype));
        self.invalidate_models();
    }

    fn invalidate_models(&mut self) {
        for slot in &mut self.mil_slots {
            slot.lowered = None;
            slot.model = None;
            slot.pkg_dir = None;
        }
    }

    /// Lower → package → load. Idempotent; a no-op once all models are loaded.
    pub fn finalize(&mut self) -> Result<()> {
        for (i, slot) in self.mil_slots.iter_mut().enumerate() {
            if slot.model.is_some() {
                continue;
            }
            let lowered = lower_graph_with_options(
                &slot.graph,
                &self.params,
                &self.typed_params,
                &self.lower_opts,
            )?;
            let proto_bytes = crate::mlpackage::encode_model(&lowered.model)?;
            let key = content_hash(&proto_bytes, &lowered.blob);
            let cache_dir = std::env::temp_dir().join("rlx-coreml-cache");
            let cache_path = cache_dir.join(format!("{key}.mlmodelc"));

            let seq = PKG_COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!(
                "rlx-coreml-{pid}-{seq}-{i}-{}.mlpackage",
                sanitize(&self.graph.name)
            ));
            // Skip re-serialising the `.mlpackage` (incl. a possibly multi-GB
            // weight blob) when the compiled-model cache is already warm — the
            // native loader loads straight from the `.mlmodelc`. But if that
            // cache turns out to be stale, the native side discards it and
            // falls back to recompiling from the `.mlpackage`; write it and
            // retry so a bad cache entry can't permanently wedge loading.
            let compute = self.compute_units.code();
            let wrote_pkg = !cache_path.exists();
            if wrote_pkg {
                crate::mlpackage::write_mlpackage_bytes(&proto_bytes, &lowered.blob, &dir)?;
            }
            let model = match CoremlModel::load(&dir, compute, Some(cache_path.as_path())) {
                Ok(m) => m,
                Err(_) if !wrote_pkg => {
                    crate::mlpackage::write_mlpackage_bytes(&proto_bytes, &lowered.blob, &dir)?;
                    CoremlModel::load(&dir, compute, Some(cache_path.as_path()))?
                }
                Err(e) => {
                    // Dump I/O feature shapes so "Error in declaring input X"
                    // failures are actionable (shape / dtype / name collision).
                    if std::env::var("RLX_COREML_DEBUG_IO").as_deref() == Ok("1") {
                        eprintln!("[coreml] finalize slot {i} load failed ({e}); inputs:");
                        for io in &lowered.inputs {
                            eprintln!(
                                "  in  '{}' ir='{}' dims={:?} flex={:?} dtype={:?}",
                                io.feature_name, io.ir_name, io.dims, io.flex_dims, io.dtype
                            );
                        }
                        for io in &lowered.outputs {
                            eprintln!(
                                "  out '{}' ir='{}' dims={:?} dtype={:?}",
                                io.feature_name, io.ir_name, io.dims, io.dtype
                            );
                        }
                        for n in slot.graph.nodes() {
                            if let Op::Input { name } = &n.op {
                                eprintln!(
                                    "  graph Input '{name}' shape={:?} dtype={:?}",
                                    n.shape.dims(),
                                    n.shape.dtype()
                                );
                            }
                        }
                    }
                    return Err(e);
                }
            };
            slot.lowered = Some(lowered);
            slot.pkg_dir = Some(dir);
            slot.model = Some(model);
        }
        Ok(())
    }

    /// Run a prediction. `inputs` are `(ir_input_name, f32_data)`. Outputs
    /// are returned in graph-output order.
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
        let outs = match self.plan.clone() {
            ExecutionPlan::MilOnly => {
                self.finalize()?;
                self.run_coreml_slot(0, inputs, &HashMap::new())?
            }
            ExecutionPlan::Segmented(segments) => self.run_segmented(inputs, &segments)?,
        };
        // NaN/Inf output-boundary scan (RLX_DEBUG_NANS). MIL segments run
        // opaquely on the Neural Engine, so per-op localization is only
        // possible for host segments (done in `run_segmented`); here we scan
        // the final outputs. For internal localization replay on CPU.
        let scanner = rlx_ir::numeric_check::DebugScanner::from_env("coreml");
        if scanner.enabled() {
            for (buf, &oid) in outs.iter().zip(self.graph.outputs.iter()) {
                scanner.check(&self.graph, oid, buf, &[]);
            }
        }
        Ok(outs)
    }

    fn run_segmented(
        &mut self,
        inputs: &[(&str, &[f32])],
        segments: &[Segment],
    ) -> Result<Vec<Vec<f32>>> {
        let mut env: HashMap<u32, Vec<f32>> = HashMap::new();
        seed_leaf_env(
            &self.graph,
            inputs,
            &self.params,
            &self.typed_params,
            &mut env,
        )?;
        // Host segments run op-by-op on the CPU, so we can localize a NaN to the
        // exact host op (culprit vs propagator) — MIL segments stay opaque.
        let scanner = rlx_ir::numeric_check::DebugScanner::from_env("coreml");
        let mut mil_idx = 0usize;
        for seg in segments {
            match seg {
                Segment::Host(ids) => {
                    for &id in ids {
                        let v =
                            run_host_node(&self.graph, id, &env, &self.params, &self.typed_params)?;
                        if scanner.enabled() {
                            let mut inbufs: Vec<(rlx_ir::NodeId, &[f32])> = Vec::new();
                            for &inp in &self.graph.node(id).inputs {
                                if let Some(buf) = env.get(&inp.0) {
                                    inbufs.push((inp, buf));
                                }
                            }
                            scanner.check(&self.graph, id, &v, &inbufs);
                        }
                        env.insert(id.0, v);
                    }
                }
                Segment::Mil(seg @ MilSegment { graph, .. }) => {
                    if hybrid::mil_body_is_trivial(graph) {
                        mil_idx += 1;
                        continue;
                    }
                    self.finalize()?;
                    let outs = self.run_coreml_slot(mil_idx, inputs, &env)?;
                    if let Some(lowered) =
                        self.mil_slots.get(mil_idx).and_then(|s| s.lowered.as_ref())
                    {
                        for (io, buf) in lowered.outputs.iter().zip(outs) {
                            // The MIL subgraph uses LOCAL node ids; translate each
                            // output back to its original GLOBAL id so `env` (and
                            // hence the graph's declared outputs) sees it. Falls
                            // back to the raw id for the first segment where
                            // local == global.
                            if let Some(local) = parse_vname(&io.ir_name) {
                                let global = seg
                                    .out_local_to_global
                                    .get(&local)
                                    .copied()
                                    .unwrap_or(local);
                                env.insert(global, buf);
                            } else if let Some(id) = io
                                .ir_name
                                .strip_prefix("host_v")
                                .and_then(|s| s.parse().ok())
                            {
                                env.insert(id, buf);
                            }
                        }
                    }
                    mil_idx += 1;
                }
            }
        }
        self.graph
            .outputs
            .iter()
            .map(|&oid| {
                env.get(&oid.0)
                    .cloned()
                    .ok_or_else(|| CoremlError::Runtime(format!("missing output v{}", oid.0)))
            })
            .collect()
    }

    fn run_coreml_slot(
        &mut self,
        slot_idx: usize,
        inputs: &[(&str, &[f32])],
        env: &HashMap<u32, Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>> {
        use crate::mil::IoTensor;
        use half::f16;
        use rlx_ir::DType;

        let slot = self
            .mil_slots
            .get(slot_idx)
            .ok_or_else(|| CoremlError::Runtime(format!("missing MIL slot {slot_idx}")))?;
        let lowered = slot.lowered.as_ref().expect("finalized");
        let mut in_byte_bufs: Vec<Vec<u8>> = Vec::new();
        let mut in_shapes: Vec<Vec<i64>> = Vec::new();
        let mut in_args: Vec<(std::ffi::CString, Vec<i64>, usize, i32)> = Vec::new();

        for io in &lowered.inputs {
            let data: &[f32] = if let Some(id) = io
                .ir_name
                .strip_prefix("host_v")
                .and_then(|s| s.parse().ok())
            {
                env.get(&id).map(|v| v.as_slice()).ok_or_else(|| {
                    CoremlError::Runtime(format!("missing host tensor '{}'", io.ir_name))
                })?
            } else {
                inputs
                    .iter()
                    .find(|(n, _)| *n == io.ir_name)
                    .map(|(_, d)| *d)
                    .ok_or_else(|| {
                        CoremlError::Runtime(format!("missing input '{}'", io.ir_name))
                    })?
            };
            let dims = io.runtime_dims(data.len());
            let (buf, dtype_code) = match io.dtype {
                DType::F16 => {
                    let mut b = Vec::with_capacity(data.len() * 2);
                    for f in data {
                        b.extend_from_slice(&f16::from_f32(*f).to_bits().to_le_bytes());
                    }
                    (b, 1i32)
                }
                _ => {
                    let mut b = Vec::with_capacity(data.len() * 4);
                    for f in data {
                        b.extend_from_slice(&f.to_le_bytes());
                    }
                    (b, 0i32)
                }
            };
            let cname = std::ffi::CString::new(io.feature_name.as_bytes())
                .map_err(|_| CoremlError::Runtime("feature name contains NUL".into()))?;
            let buf_idx = in_byte_bufs.len();
            in_byte_bufs.push(buf);
            in_shapes.push(dims);
            in_args.push((
                cname,
                in_shapes.last().unwrap().clone(),
                buf_idx,
                dtype_code,
            ));
        }

        let resolved_inputs: Vec<(&IoTensor, Vec<i64>)> = lowered
            .inputs
            .iter()
            .zip(in_shapes.iter())
            .map(|(io, d)| (io, d.clone()))
            .collect();

        let mut out_bufs: Vec<Vec<f32>> = Vec::with_capacity(lowered.outputs.len());
        for io in &lowered.outputs {
            let dims = resolve_output_dims(io, &resolved_inputs);
            let n = IoTensor::runtime_numel(&dims);
            out_bufs.push(vec![0.0f32; n]);
        }

        let predict_ins: Vec<(std::ffi::CString, Vec<i64>, &[u8], i32)> = in_args
            .iter()
            .map(|(name, shape, buf_idx, dt)| {
                (
                    name.clone(),
                    shape.clone(),
                    in_byte_bufs[*buf_idx].as_slice(),
                    *dt,
                )
            })
            .collect();
        let mut out_args: Vec<(std::ffi::CString, &mut [f32])> = Vec::new();
        for (io, buf) in lowered.outputs.iter().zip(out_bufs.iter_mut()) {
            let cname = std::ffi::CString::new(io.feature_name.as_bytes())
                .map_err(|_| CoremlError::Runtime("feature name contains NUL".into()))?;
            out_args.push((cname, buf.as_mut_slice()));
        }
        self.mil_slots
            .get_mut(slot_idx)
            .expect("slot")
            .model
            .as_mut()
            .expect("finalized")
            .predict(&predict_ins, &mut out_args)?;
        Ok(out_bufs)
    }

    /// Per-device op counts `{cpu, gpu, ane, unknown}` from MLComputePlan,
    /// or `None` if unsupported on this OS. Auto-finalizes.
    pub fn compute_plan(&mut self) -> Result<Option<[i32; 4]>> {
        self.finalize()?;
        Ok(self
            .mil_slots
            .first_mut()
            .and_then(|s| s.model.as_mut())
            .and_then(|m| m.compute_plan()))
    }

    /// Host chip / ANE identity.
    pub fn chip_info(&self) -> ChipInfo {
        crate::chip_info()
    }
}

impl Drop for CoremlExecutable {
    fn drop(&mut self) {
        for slot in &mut self.mil_slots {
            slot.model = None;
            if let Some(dir) = slot.pkg_dir.take() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }
}

fn resolve_output_dims(
    io: &crate::mil::IoTensor,
    inputs: &[(&crate::mil::IoTensor, Vec<i64>)],
) -> Vec<i64> {
    if !io.flex_dims.iter().any(|&f| f) {
        return io.dims.clone();
    }
    let mut dims = io.dims.clone();
    if let Some((_, in_dims)) = inputs.first() {
        for (i, (d, flex)) in dims.iter_mut().zip(io.flex_dims.iter()).enumerate() {
            if *flex && i < in_dims.len() {
                *d = in_dims[i];
            }
        }
    }
    dims
}

fn mil_slots_for_plan(graph: &Graph, plan: &ExecutionPlan) -> Vec<MilSlot> {
    match plan {
        ExecutionPlan::MilOnly => vec![MilSlot {
            graph: graph.clone(),
            lowered: None,
            model: None,
            pkg_dir: None,
        }],
        ExecutionPlan::Segmented(segments) => segments
            .iter()
            .filter_map(|s| match s {
                Segment::Mil(m) => Some(MilSlot {
                    graph: m.graph.clone(),
                    lowered: None,
                    model: None,
                    pkg_dir: None,
                }),
                Segment::Host(_) => None,
            })
            .collect(),
    }
}

fn sanitize(raw: &str) -> String {
    raw.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn parse_vname(name: &str) -> Option<u32> {
    // Names are `v{id}`, but the MIL lowering may tag an output that is *also*
    // consumed by a later segment as `v{id}_g{n}` (disambiguating it from the
    // `host_v{id}` boundary-input name). Take the leading digits after `v` so
    // `v3_g1` → 3 — otherwise that output is silently dropped from `env` and a
    // downstream host op (e.g. a host-eval LSTM) fails with "missing value".
    let s = name.strip_prefix('v')?;
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

fn seed_leaf_env(
    graph: &Graph,
    inputs: &[(&str, &[f32])],
    params: &HashMap<String, Vec<f32>>,
    typed_params: &TypedParams,
    env: &mut HashMap<u32, Vec<f32>>,
) -> Result<()> {
    for node in graph.nodes() {
        match &node.op {
            Op::Input { name } => {
                let data = inputs
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, d)| d.to_vec())
                    .ok_or_else(|| CoremlError::Runtime(format!("missing input '{name}'")))?;
                env.insert(node.id.0, data);
            }
            Op::Param { name } => {
                if let Some(data) = params.get(name) {
                    env.insert(node.id.0, data.clone());
                } else if let Some((bytes, dtype)) = typed_params.get(name) {
                    // Widen every typed param to f32 for the host `env`. Packed
                    // I8/U8 quant weights are re-narrowed by `run_custom_f32`
                    // when a host Custom (QMatMul / DynamicQuantizeLSTM) needs
                    // the original integer dtype.
                    let floats = bytes_to_f32(bytes, &node.shape.clone().with_dtype(*dtype))?;
                    env.insert(node.id.0, floats);
                } else {
                    return Err(CoremlError::Runtime(format!("missing param '{name}'")));
                }
            }
            Op::Constant { data } => {
                let floats = bytes_to_f32(data, &node.shape)?;
                env.insert(node.id.0, floats);
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::Activation;
    use rlx_ir::{DType, Graph, Shape};
    use std::sync::Mutex;

    // Env mutation must be single-threaded across these cases.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn tiny_fp32() -> Graph {
        let mut g = Graph::new("fp32_units");
        let x = g.input("x", Shape::new(&[2, 2], DType::F32));
        let y = g.activation(Activation::Relu, x, Shape::new(&[2, 2], DType::F32));
        g.set_outputs(vec![y]);
        g
    }

    fn tiny_f16() -> Graph {
        let mut g = Graph::new("f16_units");
        let x = g.input("x", Shape::new(&[2, 2], DType::F16));
        let y = g.activation(Activation::Relu, x, Shape::new(&[2, 2], DType::F16));
        g.set_outputs(vec![y]);
        g
    }

    /// I16 is included in `promote_int_to_f32` (CoreML has no int16 storage),
    /// so both an I16 `Constant` and a `Cast { to: I16 }` are rewritten to F32
    /// — mirroring how I8/I32/I64 are already handled.
    #[test]
    fn promote_int_to_f32_handles_i16() {
        let mut g = Graph::new("i16_promote");
        let data: Vec<u8> = [1i16, -2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let k = g.add_node(Op::Constant { data }, vec![], Shape::new(&[3], DType::I16));
        let x = g.input("x", Shape::new(&[3], DType::F32));
        let c = g.add_node(
            Op::Cast { to: DType::I16 },
            vec![x],
            Shape::new(&[3], DType::I16),
        );
        g.set_outputs(vec![k, c]);

        promote_int_to_f32(&mut g);

        // No I16 dtype survives the promotion.
        for n in g.nodes() {
            assert_ne!(n.shape.dtype(), DType::I16, "I16 must be promoted to F32");
        }
        // The cast target was rewritten to F32.
        let cast = g
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Cast { .. }))
            .unwrap();
        assert!(matches!(cast.op, Op::Cast { to: DType::F32 }));
        // The I16 constant bytes were re-encoded as f32 [1.0, -2.0, 3.0].
        let konst = g
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Constant { .. }))
            .unwrap();
        let Op::Constant { data } = &konst.op else {
            unreachable!()
        };
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(floats, vec![1.0, -2.0, 3.0]);
    }

    fn with_units_env(val: Option<&str>, f: impl FnOnce()) {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("RLX_COREML_UNITS");
        match val {
            Some(v) => unsafe { std::env::set_var("RLX_COREML_UNITS", v) },
            None => unsafe { std::env::remove_var("RLX_COREML_UNITS") },
        }
        f();
        match prev {
            Some(v) => unsafe { std::env::set_var("RLX_COREML_UNITS", v) },
            None => unsafe { std::env::remove_var("RLX_COREML_UNITS") },
        }
    }

    #[test]
    fn default_compute_units_policy() {
        with_units_env(None, || {
            assert_eq!(default_compute_units(&tiny_fp32()), ComputeUnits::CpuAndGpu);
            assert_eq!(
                default_compute_units(&tiny_f16()),
                ComputeUnits::CpuAndNeuralEngine
            );
        });
        with_units_env(Some("ane"), || {
            assert_eq!(
                default_compute_units(&tiny_fp32()),
                ComputeUnits::CpuAndNeuralEngine
            );
        });
        with_units_env(Some("gpu"), || {
            assert_eq!(default_compute_units(&tiny_f16()), ComputeUnits::CpuAndGpu);
        });
        with_units_env(Some("cpu"), || {
            assert_eq!(default_compute_units(&tiny_fp32()), ComputeUnits::CpuOnly);
        });
        with_units_env(Some("all"), || {
            assert_eq!(default_compute_units(&tiny_fp32()), ComputeUnits::All);
        });
    }
}
