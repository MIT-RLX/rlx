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

use rlx_ir::{DType, Graph, Op, OpKind};

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
/// I8/U8) to F32 — node output dtypes, integer `Constant` data, and `Cast { to: int }`
/// targets. `Bool` is preserved (CoreML `select`/logical ops need it) and floats are
/// untouched. Integer-only consumers (e.g. `gather` indices) cast back to int32 in the
/// MIL lowering. This makes index/shape arithmetic flow as exact integer-valued f32.
fn promote_int_to_f32(graph: &mut Graph) {
    fn is_int(dt: DType) -> bool {
        matches!(
            dt,
            DType::I64 | DType::I32 | DType::U32 | DType::I8 | DType::U8
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
    LowerOptions {
        float_dtype: DType::F32,
        flexible_inputs: rlx_ir::dynamic::has_dynamic_dims(graph)
            || std::env::var("RLX_COREML_FLEXIBLE_INPUTS").ok().as_deref() == Some("1"),
        ondevice_dequant: std::env::var("RLX_COREML_HOST_DEQUANT").ok().as_deref() != Some("1"),
    }
}

/// Whether `graph` computes gradients — it carries a loss-cotangent seed input
/// (`grad_with_loss` names it `d_output`) or still contains a native `*Backward`
/// op (RMSNorm / MaxPool2d, which CoreML lowers directly). Used to pick a
/// precision-preserving compute policy for training.
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
/// - inference keeps the Neural Engine.
///
/// So precision↔speed is the user's choice: default fp32 (CPU+GPU, accurate) or an
/// AFP f16 policy (ANE, fast). `RLX_COREML_UNITS=ane` also forces the ANE on an
/// fp32 model (it downcasts internally).
pub fn default_compute_units(graph: &Graph) -> ComputeUnits {
    match std::env::var("RLX_COREML_UNITS").as_deref() {
        Ok("cpu") => ComputeUnits::CpuOnly,
        Ok("gpu") => ComputeUnits::CpuAndGpu,
        Ok("all") => ComputeUnits::All,
        Ok("ane") => ComputeUnits::CpuAndNeuralEngine,
        _ if graph_has_f16(graph) => ComputeUnits::CpuAndNeuralEngine,
        _ if is_backward_graph(graph) => ComputeUnits::CpuAndGpu,
        _ => ComputeUnits::CpuAndNeuralEngine,
    }
}

/// True if any node carries an f16 tensor — the signature of an
/// `AutoMixedPrecision` / fp16 graph (the caller traded precision for speed).
fn graph_has_f16(graph: &Graph) -> bool {
    graph.nodes().iter().any(|n| n.shape.dtype() == DType::F16)
}

impl CoremlExecutable {
    /// Stage a graph for CoreML execution under the default ([`All`])
    /// compute-unit policy.
    ///
    /// [`All`]: ComputeUnits::All
    pub fn compile(graph: Graph) -> Self {
        let units = match std::env::var("RLX_COREML_UNITS").as_deref() {
            Ok("cpu") => ComputeUnits::CpuOnly,
            Ok("gpu") => ComputeUnits::CpuAndGpu,
            Ok("all") => ComputeUnits::All,
            _ => ComputeUnits::CpuAndNeuralEngine,
        };
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
        // `FusedAttentionBlock` is a claimed op (so it legalizes and the
        // fusion pipeline may emit it), but the MIL lowering has no
        // fused-attention op — decompose it to the primitive chain
        // (matmul → narrow → rope → attention → matmul) that CoreML lowers.
        // FAB-only so native LoraMatMul / FusedSwiGLU-style ops are kept.
        // No-op when no FAB node is present.
        graph = rlx_opt::unfuse::unfuse_attention_block(graph);
        promote_int_to_f32(&mut graph);
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
            if !cache_path.exists() {
                crate::mlpackage::write_mlpackage_bytes(&proto_bytes, &lowered.blob, &dir)?;
            }
            let model =
                CoremlModel::load(&dir, self.compute_units.code(), Some(cache_path.as_path()))?;
            slot.lowered = Some(lowered);
            slot.pkg_dir = Some(dir);
            slot.model = Some(model);
        }
        Ok(())
    }

    /// Run a prediction. `inputs` are `(ir_input_name, f32_data)`. Outputs
    /// are returned in graph-output order.
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
        match self.plan.clone() {
            ExecutionPlan::MilOnly => {
                self.finalize()?;
                Ok(self.run_coreml_slot(0, inputs, &HashMap::new())?)
            }
            ExecutionPlan::Segmented(segments) => self.run_segmented(inputs, &segments),
        }
    }

    fn run_segmented(
        &mut self,
        inputs: &[(&str, &[f32])],
        segments: &[Segment],
    ) -> Result<Vec<Vec<f32>>> {
        let mut env: HashMap<u32, Vec<f32>> = HashMap::new();
        seed_leaf_env(&self.graph, inputs, &self.params, &mut env)?;
        let mut mil_idx = 0usize;
        for seg in segments {
            match seg {
                Segment::Host(ids) => {
                    for &id in ids {
                        let v = run_host_node(&self.graph, id, &env, &self.params)?;
                        env.insert(id.0, v);
                    }
                }
                Segment::Mil(MilSegment { graph, .. }) => {
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
                            if let Some(id) = parse_vname(&io.ir_name) {
                                env.insert(id, buf);
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
    name.strip_prefix('v').and_then(|s| s.parse().ok())
}

fn seed_leaf_env(
    graph: &Graph,
    inputs: &[(&str, &[f32])],
    params: &HashMap<String, Vec<f32>>,
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
                let data = params
                    .get(name)
                    .cloned()
                    .ok_or_else(|| CoremlError::Runtime(format!("missing param '{name}'")))?;
                env.insert(node.id.0, data);
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
