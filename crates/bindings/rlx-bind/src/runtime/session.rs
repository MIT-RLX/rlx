// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Session` → `Compiled`: compile a graph for one device, set params, run.
//!
//! The binding-neutral form of `rlx_runtime::{Session, CompiledGraph}`: device
//! and precision by name, compile options from JSON, every call panic-guarded,
//! and `run` refusing outputs it would otherwise reinterpret.

use rlx_ir::logical_kernel::KernelDispatchPolicy;
use rlx_ir::{DType, Dim, Graph};
use rlx_opt::FusionOptions;
use rlx_runtime::{CompileOptions, CompiledGraph, Device, PrecisionPolicy};
use serde_json::Value;

use super::opts::Opts;
use super::policy;
use crate::enums::dtype_label;
use crate::panics::catch;
use crate::{ApiError, ApiResult};

/// A device by name, refusing one this build does not include.
///
/// Never a silent CPU fallback: a caller asking for CUDA on a CPU-only build
/// would otherwise report CPU numbers as CUDA ones.
pub fn device(label: &str) -> ApiResult<Device> {
    let device = rlx_runtime::parse_device(label).map_err(|e| ApiError::type_(e.to_string()))?;
    if !rlx_runtime::is_available(device) {
        let have = rlx_runtime::available_devices()
            .into_iter()
            .map(rlx_runtime::device_label)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ApiError::runtime(format!(
            "device '{label}' is not in this build — available: [{have}]. Rebuild with \
             `--features {label}` (pyrlx: `maturin develop --features {label}`)."
        )));
    }
    Ok(device)
}

pub fn kernel_dispatch(s: &str) -> ApiResult<KernelDispatchPolicy> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "native" | "force_native" => KernelDispatchPolicy::ForceNative,
        "common" | "force_common" => KernelDispatchPolicy::ForceCommon,
        "prefer_native" | "prefer" | "default" => KernelDispatchPolicy::PreferNative,
        other => {
            return Err(ApiError::type_(format!(
                "unknown kernel dispatch '{other}' (native, common, prefer_native)"
            )));
        }
    })
}

/// Fusion toggles from `{skip_fusion, fk_fusion, …}`.
///
/// An unnamed flag keeps its `FusionOptions` default rather than going to
/// `false` — `fk_fusion` and the two `fuse_*` flags default **on**, and
/// clearing them would make `{}` a different compile from no options at all.
pub fn fusion(v: &Value) -> ApiResult<FusionOptions> {
    let o = Opts::new("fusion", v)?;
    let base = FusionOptions::default();
    let out = FusionOptions {
        skip_fusion: o.bool("skip_fusion", base.skip_fusion)?,
        unfuse_elementwise_regions: o.bool(
            "unfuse_elementwise_regions",
            base.unfuse_elementwise_regions,
        )?,
        keep_elementwise_regions: o
            .bool("keep_elementwise_regions", base.keep_elementwise_regions)?,
        decompose_fusion_regions: o
            .bool("decompose_fusion_regions", base.decompose_fusion_regions)?,
        fk_fusion: o.bool("fk_fusion", base.fk_fusion)?,
        fuse_region_prologue: o.bool("fuse_region_prologue", base.fuse_region_prologue)?,
        fuse_batch_preprocess: o.bool("fuse_batch_preprocess", base.fuse_batch_preprocess)?,
        native_fk_regions: o.bool("native_fk_regions", base.native_fk_regions)?,
        disable_conv_bias_act_fusion: o.bool(
            "disable_conv_bias_act_fusion",
            base.disable_conv_bias_act_fusion,
        )?,
        ..base
    };
    o.finish()?;
    Ok(out)
}

/// What a compile call may override.
#[derive(Clone, Debug, Default)]
pub struct CompileRequest {
    pub fusion: Option<FusionOptions>,
    pub kernel_dispatch: Option<KernelDispatchPolicy>,
    pub policy: Option<PrecisionPolicy>,
}

impl CompileRequest {
    /// From `{fusion, kernel_dispatch, policy}`.
    pub fn from_json(v: &Value) -> ApiResult<Self> {
        let o = Opts::new("compile", v)?;
        let fusion = o.get("fusion").map(fusion).transpose()?;
        let kernel_dispatch = o.str("kernel_dispatch")?.map(kernel_dispatch).transpose()?;
        let policy = match o.get("policy") {
            Some(p) => policy::parse(p)?,
            None => None,
        };
        o.finish()?;
        Ok(Self {
            fusion,
            kernel_dispatch,
            policy,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.fusion.is_none() && self.kernel_dispatch.is_none() && self.policy.is_none()
    }
}

/// `CompileOptions` for `precision` with a request's overrides applied — for
/// compile paths that pick their device later (`FlexibleSession`, routing).
pub fn compile_options(precision: rlx_runtime::Precision, req: &CompileRequest) -> CompileOptions {
    let mut opts = CompileOptions::new().precision(precision);
    if let Some(f) = req.fusion {
        opts.fusion_opts = f;
    }
    if let Some(k) = req.kernel_dispatch {
        opts.kernel_dispatch.policy = k;
    }
    if let Some(p) = req.policy.clone() {
        opts = opts.policy(p);
    }
    opts
}

/// A device and precision to compile graphs for.
pub struct Session {
    pub inner: rlx_runtime::Session,
    /// Applies to every compile, not only those passing options, so
    /// `Session(policy=…)` does not silently need the long form.
    pub policy: Option<PrecisionPolicy>,
}

impl Session {
    pub fn new(
        device_label: &str,
        precision: &str,
        policy: Option<PrecisionPolicy>,
    ) -> ApiResult<Self> {
        let device = device(device_label)?;
        // The runtime's own reason (feature not built in, or device absent),
        // instead of the panic `Session::new` raises.
        rlx_runtime::Session::try_new(device).map_err(compile_error)?;
        let precision = policy::precision(precision)?;
        if let Some(p) = policy.as_ref() {
            policy::check_supported(device, p)?;
        }
        Ok(Self {
            inner: rlx_runtime::Session::new_with_precision(device, precision),
            policy,
        })
    }

    pub fn device(&self) -> Device {
        self.inner.device()
    }

    pub fn device_label(&self) -> &'static str {
        rlx_runtime::device_label(self.inner.device())
    }

    pub fn precision_label(&self) -> &'static str {
        policy::precision_label(self.inner.precision())
    }

    pub fn policy_label(&self) -> Option<&'static str> {
        self.policy.as_ref().map(policy::label)
    }

    /// The `CompileOptions` a request resolves to: a per-call policy wins over
    /// the session's.
    pub fn options(&self, req: &CompileRequest) -> ApiResult<CompileOptions> {
        let mut req = req.clone();
        if req.policy.is_none() {
            req.policy = self.policy.clone();
        }
        if let Some(p) = req.policy.as_ref() {
            policy::check_supported(self.device(), p)?;
        }
        Ok(compile_options(self.inner.precision(), &req))
    }

    /// Compile `graph` (consumed). Legalization, fusion and memory planning all
    /// assert; a graph the backend cannot take is an error, not an abort.
    pub fn compile(&self, graph: Graph, req: &CompileRequest) -> ApiResult<Compiled> {
        let (output_shapes, output_dtypes) = Compiled::outputs_of(&graph);
        let compiled = if req.is_empty() && self.policy.is_none() {
            catch("compile", || self.inner.try_compile(graph))?
        } else {
            let opts = self.options(req)?;
            catch("compile", || self.inner.try_compile_with(graph, &opts))?
        }
        .map_err(compile_error)?;
        Ok(Compiled::wrap(compiled, output_shapes, output_dtypes))
    }
}

/// `[{name, dtype, shape}]` for a compiled graph's declared tensors.
fn tensor_rows(infos: &[rlx_runtime::TensorInfo]) -> Value {
    Value::Array(
        infos
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "dtype": dtype_label(t.dtype),
                    "shape": t.shape,
                })
            })
            .collect(),
    )
}

/// A runtime [`rlx_runtime::CompileError`] as the binding error of the
/// matching kind: a malformed graph is a `Type` error, an unavailable device or
/// an op it cannot lower a `Runtime` one, and only a failure inside the
/// backend stays a `Panic`.
pub fn compile_error(e: rlx_runtime::CompileError) -> ApiError {
    use rlx_runtime::CompileError as E;
    let msg = e.to_string();
    match e {
        E::Invalid(_) => ApiError::type_(msg),
        E::DeviceUnavailable { .. } | E::Unsupported { .. } => ApiError::runtime(msg),
        E::Backend(_) => ApiError::new(crate::ErrorKind::Panic, msg),
    }
}

/// A boundary mistake (unknown input, wrong length, unbound names) as a
/// `Type` error — a wrong argument, not an rlx assertion.
pub fn io_error(e: rlx_runtime::IoError) -> ApiError {
    ApiError::type_(e.to_string())
}

/// A session that picks the device per graph from a device policy.
pub struct FlexibleSession {
    pub inner: rlx_runtime::FlexibleSession,
}

impl FlexibleSession {
    /// `policy`: null (every device), `"env"`, `"all"` or
    /// `{only, deny, prefer, benchmark_runs}`.
    pub fn new(policy: &Value) -> ApiResult<Self> {
        let p = super::devices::policy(policy)?;
        Ok(Self {
            inner: rlx_runtime::FlexibleSession::new().with_device_policy(p),
        })
    }

    /// The device policy from `RLX_DEVICE*`.
    pub fn from_env() -> Self {
        Self {
            inner: rlx_runtime::FlexibleSession::from_env(),
        }
    }

    /// Compile `graph` (consumed) on `device`, or the policy's pick for it.
    pub fn compile(
        &self,
        graph: Graph,
        device: Option<&str>,
        req: &CompileRequest,
    ) -> ApiResult<Compiled> {
        let hint = device.map(super::devices::parse).transpose()?;
        let (output_shapes, output_dtypes) = Compiled::outputs_of(&graph);
        let compiled = if req.is_empty() {
            catch("compile", || self.inner.compile_resolved(graph, hint))?
        } else {
            let opts = compile_options(self.inner.precision(), req);
            catch("compile", || {
                self.inner.compile_resolved_with(graph, hint, &opts)
            })?
        }
        .map_err(ApiError::runtime)?;
        Ok(Compiled::wrap(compiled, output_shapes, output_dtypes))
    }
}

/// Dims with a dynamic axis read as `0`.
pub fn static_dims(dims: &[Dim]) -> Vec<usize> {
    dims.iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Dynamic(_) => 0,
        })
        .collect()
}

/// A graph compiled for one device.
pub struct Compiled {
    pub inner: CompiledGraph,
    /// One entry per graph output; a dynamic dim is recorded as `0`.
    pub output_shapes: Vec<Vec<usize>>,
    /// Declared dtype per output.
    pub output_dtypes: Vec<DType>,
}

impl Compiled {
    /// Wrap a graph some other path compiled (`FlexibleSession`, routing).
    pub fn wrap(
        inner: CompiledGraph,
        output_shapes: Vec<Vec<usize>>,
        output_dtypes: Vec<DType>,
    ) -> Self {
        Self {
            inner,
            output_shapes,
            output_dtypes,
        }
    }

    /// `(static dims, dtype)` of each of `graph`'s outputs — read before the
    /// graph is consumed by a compile.
    pub fn outputs_of(graph: &Graph) -> (Vec<Vec<usize>>, Vec<DType>) {
        let shapes = graph
            .outputs
            .iter()
            .map(|n| static_dims(graph.shape(*n).dims()))
            .collect();
        let dtypes = graph
            .outputs
            .iter()
            .map(|n| graph.shape(*n).dtype())
            .collect();
        (shapes, dtypes)
    }

    pub fn device_label(&self) -> &'static str {
        rlx_runtime::device_label(self.inner.device())
    }

    pub fn output_dtype_labels(&self) -> Vec<&'static str> {
        self.output_dtypes.iter().map(|d| dtype_label(*d)).collect()
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) -> ApiResult<()> {
        catch("set_param", || self.inner.try_set_param(name, data))?.map_err(io_error)
    }

    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: &str) -> ApiResult<()> {
        let dtype = crate::enums::dtype(dtype)?;
        catch("set_param_typed", || {
            self.inner.set_param_typed(name, data, dtype)
        })
    }

    /// The graph's declared inputs as rows `{name, dtype, shape}` — `shape` is
    /// `null` when a dim is symbolic. What `run` expects.
    pub fn inputs(&self) -> Value {
        tensor_rows(&self.inner.inputs())
    }

    /// The graph's declared parameters, as [`Self::inputs`]. What
    /// `set_param` takes.
    pub fn params(&self) -> Value {
        tensor_rows(&self.inner.params())
    }

    /// The outputs in the order `run` returns them: `[{name, dtype, shape}]`,
    /// `name` empty for an output the graph does not name.
    pub fn outputs(&self) -> Value {
        tensor_rows(&self.inner.outputs())
    }

    /// The position in `run`'s result of the output named `name`; a `Range`
    /// error naming the closest names when there is none.
    pub fn output_index(&self, name: &str) -> ApiResult<usize> {
        self.inner
            .output_index(name)
            .map_err(|e| ApiError::range(e.to_string()))
    }

    /// A `Type` error naming the inputs `supplied` leaves unbound and the
    /// params never set; `Ok` when everything is bound.
    pub fn check_bindings(&self, supplied: &[&str]) -> ApiResult<()> {
        self.inner.check_bindings(supplied).map_err(io_error)
    }

    /// Declare inputs / params meant to read zeros (an empty KV state): no
    /// unbound warning, and `check_bindings` accepts them.
    pub fn allow_unbound(&mut self, names: &[&str]) {
        self.inner.allow_unbound(names);
    }

    /// Run with f32 inputs; one flat f32 buffer per output.
    ///
    /// Refuses a graph whose outputs are not all f32: the bytes coming back
    /// are reinterpreted, not converted, so a `Bool` output read this way is
    /// denormal garbage that looks like data (`2.37e-38` rather than `1`).
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> ApiResult<Vec<Vec<f32>>> {
        let offenders: Vec<String> = self
            .output_dtypes
            .iter()
            .enumerate()
            .filter(|(_, d)| **d != DType::F32)
            .map(|(i, d)| format!("{i}:{}", dtype_label(*d)))
            .collect();
        if !offenders.is_empty() {
            return Err(ApiError::type_(format!(
                "run(): output(s) [{}] are not f32, and run() reinterprets bytes rather \
                 than converting them. Use runTyped() / run_typed(), or cast the output to \
                 f32 in the graph.",
                offenders.join(", ")
            )));
        }
        catch("run", || self.inner.try_run(inputs))?.map_err(io_error)
    }

    /// Run with typed inputs; `(bytes, dtype)` per output.
    pub fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], &str)],
    ) -> ApiResult<Vec<(Vec<u8>, &'static str)>> {
        let mut typed = Vec::with_capacity(inputs.len());
        for (name, data, dtype) in inputs {
            typed.push((*name, *data, crate::enums::dtype(dtype)?));
        }
        let outs = catch("run_typed", || self.inner.run_typed(&typed))?;
        Ok(outs.into_iter().map(|(b, d)| (b, dtype_label(d))).collect())
    }
}
