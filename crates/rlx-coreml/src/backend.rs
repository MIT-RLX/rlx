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

use rlx_ir::{DType, Graph, Op};

use crate::ffi::CoremlModel;
use crate::mil::{LoweredProgram, TypedParams, lower_graph};
use crate::{ChipInfo, ComputeUnits, CoremlError, Result};

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
    lowered: Option<LoweredProgram>,
    model: Option<CoremlModel>,
    pkg_dir: Option<PathBuf>,
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

impl CoremlExecutable {
    /// Stage a graph for CoreML execution under the default ([`All`])
    /// compute-unit policy.
    ///
    /// [`All`]: ComputeUnits::All
    pub fn compile(graph: Graph) -> Self {
        // Default to CPU+ANE. `ComputeUnits::All` lets CoreML's planner split a graph
        // across CPU+GPU+ANE simultaneously, which trips an MPSGraph "shape for
        // TensorData is not static" assertion at predict time on these VITS graphs;
        // CpuOnly / CpuAndGpu / CpuAndNeuralEngine all run correctly (bit-exact).
        let units = match std::env::var("RLX_COREML_UNITS").as_deref() {
            Ok("cpu") => ComputeUnits::CpuOnly,
            Ok("gpu") => ComputeUnits::CpuAndGpu,
            Ok("all") => ComputeUnits::All,
            _ => ComputeUnits::CpuAndNeuralEngine,
        };
        Self::compile_with_units(graph, units)
    }

    /// Stage a graph with an explicit compute-unit policy.
    pub fn compile_with_units(mut graph: Graph, compute_units: ComputeUnits) -> Self {
        promote_int_to_f32(&mut graph);
        CoremlExecutable {
            graph,
            params: HashMap::new(),
            typed_params: TypedParams::new(),
            compute_units,
            lowered: None,
            model: None,
            pkg_dir: None,
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
            lowered: None,
            model: None,
            pkg_dir: None,
        }
    }

    /// Provide the f32 weights for an IR `Param`. Must be called for every
    /// parameter before [`finalize`](Self::finalize)/[`run`](Self::run).
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.params.insert(name.to_string(), data.to_vec());
        // Invalidate any previously built model — params changed.
        self.model = None;
        self.lowered = None;
    }

    /// Provide non-f32 (e.g. GGUF-quantized) weight bytes for an IR
    /// `Param`. The lowering host-dequantizes these to f32 when baking the
    /// model. F32 bytes are routed to [`set_param`](Self::set_param).
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
        self.model = None;
        self.lowered = None;
    }

    /// Lower → package → load. Idempotent; a no-op once a model is loaded.
    pub fn finalize(&mut self) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        let lowered = lower_graph(&self.graph, &self.params, &self.typed_params)?;

        // Hash the serialized model + weights → a content-addressed cache
        // key for the compiled `.mlmodelc`. Same graph + weights reuses the
        // (expensive) CoreML compile across instances and process runs.
        let proto_bytes = crate::mlpackage::encode_model(&lowered.model)?;
        let key = content_hash(&proto_bytes, &lowered.blob);
        let cache_dir = std::env::temp_dir().join("rlx-coreml-cache");
        let cache_path = cache_dir.join(format!("{key}.mlmodelc"));

        let seq = PKG_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!(
            "rlx-coreml-{pid}-{seq}-{}.mlpackage",
            sanitize(&self.graph.name)
        ));
        // On a cache hit the compile is skipped, so we needn't write the
        // `.mlpackage` at all.
        if !cache_path.exists() {
            crate::mlpackage::write_mlpackage_bytes(&proto_bytes, &lowered.blob, &dir)?;
        }

        let model = CoremlModel::load(&dir, self.compute_units.code(), Some(cache_path.as_path()))?;

        self.lowered = Some(lowered);
        self.pkg_dir = Some(dir);
        self.model = Some(model);
        Ok(())
    }

    /// Run a prediction. `inputs` are `(ir_input_name, f32_data)`. Outputs
    /// are returned in graph-output order.
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
        self.finalize()?;
        let lowered = self.lowered.as_ref().expect("finalized");

        // Marshal inputs in the lowered order, matching IR names.
        let mut in_args = Vec::with_capacity(lowered.inputs.len());
        for io in &lowered.inputs {
            let data = inputs
                .iter()
                .find(|(n, _)| *n == io.ir_name)
                .map(|(_, d)| *d)
                .ok_or_else(|| CoremlError::Runtime(format!("missing input '{}'", io.ir_name)))?;
            let cname = std::ffi::CString::new(io.feature_name.as_bytes())
                .map_err(|_| CoremlError::Runtime("feature name contains NUL".into()))?;
            in_args.push((cname, io.dims.clone(), data));
        }

        // Pre-size output buffers.
        let mut out_bufs: Vec<Vec<f32>> = lowered
            .outputs
            .iter()
            .map(|io| vec![0.0f32; io.numel()])
            .collect();
        let mut out_args: Vec<(std::ffi::CString, &mut [f32])> = Vec::new();
        for (io, buf) in lowered.outputs.iter().zip(out_bufs.iter_mut()) {
            let cname = std::ffi::CString::new(io.feature_name.as_bytes())
                .map_err(|_| CoremlError::Runtime("feature name contains NUL".into()))?;
            out_args.push((cname, buf.as_mut_slice()));
        }

        self.model
            .as_mut()
            .expect("finalized")
            .predict(&in_args, &mut out_args)?;

        Ok(out_bufs)
    }

    /// Per-device op counts `{cpu, gpu, ane, unknown}` from MLComputePlan,
    /// or `None` if unsupported on this OS. Auto-finalizes.
    pub fn compute_plan(&mut self) -> Result<Option<[i32; 4]>> {
        self.finalize()?;
        Ok(self.model.as_mut().expect("finalized").compute_plan())
    }

    /// Host chip / ANE identity.
    pub fn chip_info(&self) -> ChipInfo {
        crate::chip_info()
    }
}

impl Drop for CoremlExecutable {
    fn drop(&mut self) {
        // Free the loaded model first, then remove the on-disk package.
        self.model = None;
        if let Some(dir) = self.pkg_dir.take() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn sanitize(raw: &str) -> String {
    raw.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
