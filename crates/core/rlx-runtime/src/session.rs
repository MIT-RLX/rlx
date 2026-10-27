// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session — the main entry point for compiling and executing graphs.

use crate::backend::Backend;
use crate::compiled::CompiledGraph;
use crate::precision::Precision;
use rlx_driver::Device;
use rlx_ir::Graph;
use rlx_ir::GraphModule;
use rlx_ir::hir::HirModule;
use rlx_opt::PrecisionPolicy;

/// A session manages graph compilation and execution on a device.
pub struct Session {
    device: Device,
    precision: Precision,
    /// Optional per-op precision policy. If set, runs AutoMixedPrecision
    /// rewrite before backend compile. Works identically across all modes
    /// (AOT compile, trace/JIT, proc-macro AOT) — it's just a graph pass.
    policy: Option<PrecisionPolicy>,
}

impl Session {
    /// Create a session for the given device at default (F32) precision.
    ///
    /// # Panics
    /// Panics if the device is not available (missing feature flag).
    pub fn new(device: Device) -> Self {
        Self::new_with_precision(device, Precision::F32)
    }

    /// Create a session targeting a specific numeric precision.
    /// Backends fall back to F32 if the requested precision isn't supported.
    pub fn new_with_precision(device: Device, precision: Precision) -> Self {
        // Availability has two halves — feature compiled in, and device present.
        // Name the one that is actually missing: "enable the `gpu` feature" is
        // wrong (and misleading) on a headless host where it already is enabled.
        assert!(
            crate::device_ext::is_available(device),
            "device {} is not available — {}\n\
             Run `cargo rlx doctor` for a diagnosis and the fix (see docs/known-issues.md).",
            device,
            if crate::device_ext::feature_compiled(device) {
                match crate::device_ext::unavailable_reason(device) {
                    Some(why) => format!(
                        "the `{}` feature is enabled but: {why}",
                        feature_name(device)
                    ),
                    None => format!(
                        "the `{}` feature is enabled but no such device/driver was found on this host",
                        feature_name(device)
                    ),
                }
            } else {
                format!("enable the `{}` Cargo feature", feature_name(device))
            }
        );
        Self {
            device,
            precision,
            policy: None,
        }
    }

    /// Builder: set a per-op precision policy. Applied as a graph rewrite
    /// before backend compile. Same mechanism works for AOT compile, JIT
    /// tracing, and proc-macro AOT — it's a graph pass, not a runtime mode.
    pub fn with_policy(mut self, policy: PrecisionPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    pub fn device(&self) -> Device {
        self.device
    }
    pub fn precision(&self) -> Precision {
        self.precision
    }
    pub fn policy(&self) -> Option<&PrecisionPolicy> {
        self.policy.as_ref()
    }

    /// Compile a MIR graph through the fusion-first pipeline (`GraphModule` → LIR).
    ///
    /// Prefer [`Self::compile_hir`] or [`Self::compile_module`] for new code.
    /// This entry wraps the graph as a MIR-stage [`GraphModule`].
    ///
    /// If the graph carries `Dim::Dynamic` dims it is compiled lazily: the first
    /// `run` infers the concrete shape and specializes, so the same graph runs at
    /// multiple input shapes on every backend. See `crate::deferred`.
    ///
    /// On ANE with `RLX_COREML_NATIVE_FLEX=1`, dynamic graphs compile once with
    /// CoreML `ShapeRange` and infer shapes at predict time instead.
    pub fn compile(&self, graph: Graph) -> CompiledGraph {
        crate::taps::tapped(graph, |g| self.compile_untapped(g))
    }

    fn compile_untapped(&self, graph: Graph) -> CompiledGraph {
        if rlx_ir::dynamic::has_dynamic_dims(&graph) && !self.coreml_native_flex() {
            return self.compile_deferred(graph, self.default_options());
        }
        let opts = self.default_options();
        // A number format (`RLX_NUMBER_FORMAT`) or rounding half casts
        // (`RLX_HALF_CASTS=round`) rewrites the graph first.
        if opts
            .number_format
            .is_some_and(|f| f != rlx_ir::NumberFormat::F32)
            || opts.half_casts == crate::HalfCasts::Round
        {
            return self.compile_with(graph, &opts);
        }
        let (graph, df) = self.lower_f64(graph, &opts);
        let mut compiled = if opts.cache_param_invariant
            && let Some(staged) = self.try_hoist_param_invariant(&graph, &opts)
        {
            staged
        } else {
            self.compile_module(GraphModule::from_graph(graph))
                .expect("compile MIR graph through fusion pipeline")
        };
        compiled.set_df_boundary(df);
        compiled
    }

    /// If `graph` has a hoistable param-invariant closure, compile it as a
    /// separate run-once `prepare` graph and attach it to the compiled main
    /// graph. Returns `None` when there is nothing to hoist. Sub-compiles clear
    /// the flag so this does not recurse.
    fn try_hoist_param_invariant(
        &self,
        graph: &Graph,
        options: &crate::CompileOptions,
    ) -> Option<CompiledGraph> {
        let split = rlx_compile::split_param_invariant(graph)?;
        let mut sub = options.clone();
        sub.cache_param_invariant = false;
        let prepare = self.compile_with(split.prepare, &sub);
        let main = self.compile_with(split.main, &sub);
        Some(main.with_staging(
            prepare,
            split.boundary,
            split.prepare_params,
            split.main_params,
        ))
    }

    /// Wrap a dynamic-shape graph so it specializes + recompiles per input shape.
    fn compile_deferred(&self, graph: Graph, options: crate::CompileOptions) -> CompiledGraph {
        let backend = self.create_backend();
        let io_dtypes: Vec<(String, rlx_ir::DType)> = graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                rlx_ir::Op::Input { name } | rlx_ir::Op::Param { name } => {
                    Some((name.clone(), n.shape.dtype()))
                }
                _ => None,
            })
            .collect();
        let inner = Box::new(crate::deferred::DeferredExecutable::new(
            graph,
            backend,
            options,
            self.device,
        ));
        let mut compiled = CompiledGraph::new(inner, self.device);
        compiled.set_io_dtypes(io_dtypes);
        compiled
    }

    /// Explicit legacy alias — same as [`Self::compile`].
    pub fn compile_graph(&self, graph: Graph) -> CompiledGraph {
        self.compile(graph)
    }

    /// Compile with explicit options (full control over the pipeline).
    /// Most callers use `compile()` and configure the session via
    /// `new_with_precision` / `with_policy`. This escape hatch is for
    /// callers that need finer control (e.g., disable DCE for debugging).
    pub fn compile_with(&self, graph: Graph, options: &crate::CompileOptions) -> CompiledGraph {
        crate::taps::tapped(graph, |g| self.compile_with_untapped(g, options))
    }

    fn compile_with_untapped(
        &self,
        graph: Graph,
        options: &crate::CompileOptions,
    ) -> CompiledGraph {
        // Opt-in native low-precision GEMM: rewrite every 2-D matmul into a
        // dynamically-quantized `ScaledMatMul` in the requested format before
        // the rest of the pipeline (must precede constant folding). Requires
        // static matmul shapes.
        let graph = match options.scaled_quant {
            Some(cfg) => {
                rlx_opt::rlx_compile::scaled_quant_insert::insert_scaled_matmul(graph, cfg)
            }
            None => graph,
        };
        // `assoc-scan`: affine-recurrence Scans become a log-depth prefix scan
        // of tensor ops, before fusion sees the graph. Enabled where measured
        // to win: Metal, where a Scan is a host-driven loop (5-10x at
        // T=256..4096). The CPU's native loop beats the rewrite's O(T log T)
        // work by 2.5-4x. wgpu host-stages large `Concat`/`Expand` (bind-window
        // limit), which made T=4096 x 512 1.6x slower. CUDA/ROCm/MLX/Vulkan
        // are unmeasured.
        #[cfg(feature = "assoc-scan")]
        let graph = if self.device == Device::Metal {
            use rlx_opt::pass::Pass as _;
            rlx_opt::rlx_fusion::assoc_scan::ParallelizeAffineScan::default().run(graph)
        } else {
            graph
        };
        let graph = if options.half_casts == crate::HalfCasts::Round {
            crate::cpu_low_precision::round_half_casts(graph)
        } else {
            graph
        };
        let upload_rounding = upload_rounding(&graph, options);
        let (graph, widened) = apply_number_format(graph, options);
        self.check_native_f64(&graph, options);
        // Fused kernels (SwiGLU, matmul+bias+act, residual norms, elementwise
        // regions) compute in f32, so an f64 graph on the CPU runs unfused.
        let unfused;
        let options = if self.device == Device::Cpu && !widened.is_empty() {
            let mut o = options.clone();
            o.fusion_opts.skip_fusion = true;
            unfused = o;
            &unfused
        } else {
            options
        };
        let (graph, df) = self.lower_f64(graph, options);
        let mut compiled = self.compile_graph_stage(graph, options);
        compiled.set_upload_rounding(upload_rounding);
        compiled.set_widened_params(widened);
        compiled.set_df_boundary(df);
        compiled
    }

    /// `NumberFormat::F64` on the CPU runs every F64 value through an f64
    /// kernel. An op without one would be handed to its f32 kernel over f64
    /// buffers (or skipped), so refuse the graph instead.
    // `graph` is only read by the `cpu` arm; without that feature it is unused.
    #[cfg_attr(not(feature = "cpu"), allow(unused_variables))]
    fn check_native_f64(&self, graph: &Graph, options: &crate::CompileOptions) {
        if self.device != Device::Cpu || options.number_format != Some(rlx_ir::NumberFormat::F64) {
            return;
        }
        #[cfg(feature = "cpu")]
        {
            let f64_at = |id: &rlx_ir::NodeId| graph.node(*id).shape.dtype() == rlx_ir::DType::F64;
            let bad = graph.nodes().iter().find(|n| {
                (f64_at(&n.id) || n.inputs.iter().any(f64_at))
                    && !rlx_cpu::f64_kernels::native_f64(&n.op)
            });
            if let Some(n) = bad {
                panic!(
                    "number format f64: the CPU has no f64 kernel for {} (node {:?} in `{}`)",
                    n.op, n.id, graph.name
                );
            }
        }
    }

    /// Rewrite F64 compute into double-single pairs where the backend has no
    /// f64 of its own (everywhere but the CPU), or everywhere under
    /// `NumberFormat::F64x2`. Without it an F64 graph ran f32 kernels over f64
    /// buffers on Metal and Vulkan — wrong results, no error — and failed
    /// outright on wgpu and MLX.
    fn lower_f64(
        &self,
        graph: Graph,
        options: &crate::CompileOptions,
    ) -> (Graph, Option<crate::compiled::DfBoundary>) {
        use rlx_opt::rlx_compile::lower_f64;
        // F64x2 always, and F64 off the CPU, are requests the caller made:
        // failing to honour them is an error, never a silent f32 run.
        let forced = match options.number_format {
            Some(rlx_ir::NumberFormat::F64x2) => true,
            Some(rlx_ir::NumberFormat::F64) => self.device != Device::Cpu,
            _ => false,
        };
        if !lower_f64::has_f64(&graph) || (self.device == Device::Cpu && !forced) {
            return (graph, None);
        }
        // Norms, softmax, activations, RoPE and attention become df
        // primitives. Only on request: unrequested F64 graphs may rely on a
        // backend's own F64 host paths (left as they were below).
        let decomposed = if forced {
            match rlx_opt::rlx_compile::decompose_f64::decompose_f64(&graph) {
                Ok(d) => Some(d),
                Err(e) => panic!(
                    "{e}: number format {} on {:?}",
                    options.number_format.expect("forced implies a format"),
                    self.device
                ),
            }
        } else {
            None
        };
        let l = match lower_f64::lower_f64(decomposed.as_ref().unwrap_or(&graph)) {
            Ok(l) => l,
            Err(e) if forced => panic!(
                "{e}: number format {} needs double-single kernels on {:?}{}",
                options.number_format.expect("forced implies a format"),
                self.device,
                match options.number_format {
                    Some(rlx_ir::NumberFormat::F64) => " (use --device cpu for native f64)",
                    _ => "",
                }
            ),
            // Ops a backend already runs in F64 (host fallbacks such as the
            // SPD family, Metal's F64 FFT) have no df lowering; leave such a
            // graph as it compiled before rather than refuse it.
            Err(e) => {
                if rlx_ir::env::flag("RLX_LOG_F64_LOWERING") {
                    eprintln!("rlx: F64 graph left as is on {:?}: {e}", self.device);
                }
                return (graph, None);
            }
        };
        // A backend without double-single kernels keeps running the graph as
        // it did; asking for f64x2 there is an error, not a silent downgrade.
        if let Some((_, op)) =
            crate::device_ext::first_unsupported_op_with_options(self.device, &l.graph, options)
        {
            if forced {
                panic!(
                    "number format f64x2: {:?} has no kernel for {op}",
                    self.device
                );
            }
            return (graph, None);
        }
        let df = crate::compiled::DfBoundary {
            inputs: l.df_inputs.into_iter().collect(),
            params: l.df_params.into_iter().collect(),
            outputs: l.df_outputs,
        };
        (l.graph, Some(df))
    }

    fn compile_graph_stage(&self, graph: Graph, options: &crate::CompileOptions) -> CompiledGraph {
        if rlx_ir::dynamic::has_dynamic_dims(&graph) && !self.coreml_native_flex() {
            return self.compile_deferred(graph, options.clone());
        }
        if options.cache_param_invariant
            && let Some(staged) = self.try_hoist_param_invariant(&graph, options)
        {
            return staged;
        }
        self.compile_module_with(GraphModule::from_graph(graph), options)
            .expect("compile MIR graph through fusion pipeline")
    }

    /// Compile a fusion-first HIR module through HIR → MIR → LIR.
    pub fn compile_hir(&self, hir: HirModule) -> Result<CompiledGraph, rlx_ir::hir::LowerError> {
        self.compile_hir_with(hir, &self.default_options())
    }

    /// Compile HIR with explicit compile options.
    pub fn compile_hir_with(
        &self,
        hir: HirModule,
        options: &crate::CompileOptions,
    ) -> Result<CompiledGraph, rlx_ir::hir::LowerError> {
        // The number-format rewrite works on MIR; take the MIR route so it
        // is not silently skipped for HIR-built graphs.
        if options
            .number_format
            .is_some_and(|f| f != rlx_ir::NumberFormat::F32)
        {
            return Ok(self.compile_with(hir.lower_to_mir()?.into_graph(), options));
        }
        let backend = self.create_backend();
        let executable = backend.compile_hir(hir, self.device, options)?;
        Ok(CompiledGraph::new(executable, self.device))
    }

    /// Compile a [`GraphModule`] (HIR/MIR/LIR stage) through the pipeline.
    pub fn compile_module(
        &self,
        module: GraphModule,
    ) -> Result<CompiledGraph, rlx_ir::hir::LowerError> {
        self.compile_module_with(module, &self.default_options())
    }

    /// Compile a [`GraphModule`] with explicit compile options.
    pub fn compile_module_with(
        &self,
        module: GraphModule,
        options: &crate::CompileOptions,
    ) -> Result<CompiledGraph, rlx_ir::hir::LowerError> {
        let backend = self.create_backend();
        // Read the declared boundary dtypes before `module` is consumed: the
        // f32 entry points need them to convert rather than reinterpret an
        // F16/BF16 boundary. Every `Session::compile*` path funnels here.
        let io_dtypes = module.io_dtypes();
        let io_spec = module.as_graph().map(|g| {
            let mut spec = crate::io_check::IoSpec::from_graph(g);
            if let Some(b) = &options.param_bindings {
                spec.mark_params(b.keys());
            }
            if let Some(b) = &options.quant_param_bindings {
                spec.mark_params(b.keys());
            }
            spec
        });
        let executable = backend.compile_module(module, self.device, options)?;
        let mut compiled = CompiledGraph::new(executable, self.device);
        compiled.set_io_dtypes(io_dtypes);
        if let Some(spec) = io_spec {
            compiled.set_io_spec(spec);
        }
        Ok(compiled)
    }

    fn default_options(&self) -> crate::CompileOptions {
        let opts = crate::CompileOptions::new().precision(self.precision);
        match &self.policy {
            Some(p) => opts.policy(p.clone()),
            None => opts,
        }
    }

    /// ANE + `RLX_COREML_NATIVE_FLEX=1`: one CoreML model with `ShapeRange`.
    fn coreml_native_flex(&self) -> bool {
        #[cfg(all(
            feature = "coreml",
            target_vendor = "apple",
            not(target_os = "watchos")
        ))]
        {
            self.device == Device::Ane
                && rlx_ir::env::var("RLX_COREML_NATIVE_FLEX").as_deref() == Some("1")
        }
        #[cfg(not(all(
            feature = "coreml",
            target_vendor = "apple",
            not(target_os = "watchos")
        )))]
        {
            false
        }
    }

    fn create_backend(&self) -> Box<dyn Backend> {
        // Single dispatch point: consult the registry. Backends register
        // themselves (builtins via cfg-gated `register_builtin`; external
        // crates via `register_backend`). No hardcoded match here.
        crate::registry::backend_for(self.device).unwrap_or_else(|| {
            panic!(
                "no backend registered for device {} — enable feature `{}` \
                 (or call `rlx_runtime::register_backend` for an external backend)",
                self.device,
                feature_name(self.device)
            )
        })
    }
}

pub(crate) fn feature_name(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Ane => "ane",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Xdna => "xdna",
        Device::Egpu => "egpu",
        Device::OneApi => "oneapi",
        Device::Tpu => "tpu",
        Device::Hexagon => "qnn",
        Device::Gpu => "gpu",
        Device::Vulkan => "vulkan",
        Device::OpenGl => "opengl",
        Device::DirectX => "directx",
        Device::WebGpu => "webgpu",
    }
}

/// Which weights to round on upload, and their row lengths: every float
/// `Param` under a minifloat format, unless the graph rounds them itself
/// (`params_in_graph`), the caller already did (`params_prerounded`), the
/// param is kept (`keep_params`), it is a RoPE position table, or it is a
/// vector under a format of 8 bits or fewer (`rounds_param_shape`).
fn upload_rounding(
    graph: &Graph,
    options: &crate::CompileOptions,
) -> Option<crate::compiled::UploadRounding> {
    let (format, scaling) = options.number_format.and_then(|f| f.weights())?;
    let o = &options.number_format_options;
    if o.params_in_graph || o.params_prerounded {
        return None;
    }
    let tables = rlx_opt::rlx_compile::emulate_format::position_table_params(graph);
    let row_len = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            rlx_ir::Op::Param { name }
                if n.shape.dtype().is_float()
                    && !tables.contains(name)
                    && rlx_opt::rlx_compile::emulate_format::rounds_param_shape(
                        format, &n.shape,
                    )
                    && !o.keep_params.iter().any(|k| name.contains(k.as_str())) =>
            {
                let row = match n.shape.dims().last() {
                    Some(rlx_ir::Dim::Static(r)) => *r,
                    _ => 1,
                };
                Some((name.clone(), row.max(1)))
            }
            _ => None,
        })
        .collect();
    Some(crate::compiled::UploadRounding {
        format,
        scaling,
        row_len,
    })
}

/// Apply [`CompileOptions::number_format`](crate::CompileOptions::number_format).
///
/// Returns the rewritten graph and, for `F64` / `F64x2`, the params promoted
/// from F32 to F64 (callers keep uploading f32; the compiled graph widens).
fn apply_number_format(graph: Graph, options: &crate::CompileOptions) -> (Graph, Vec<String>) {
    use rlx_ir::NumberFormat;
    match options.number_format {
        None | Some(NumberFormat::F32) => (graph, Vec::new()),
        Some(f @ (NumberFormat::Float { .. } | NumberFormat::Mixed { .. })) => (
            rlx_opt::rlx_compile::emulate_format::emulate_number_format(
                graph,
                f.weights(),
                f.activations(),
                &options.number_format_options,
            ),
            Vec::new(),
        ),
        // Already F64: `Session::lower_f64` decides how to run it.
        Some(NumberFormat::F64 | NumberFormat::F64x2)
            if rlx_opt::rlx_compile::lower_f64::has_f64(&graph) =>
        {
            (graph, Vec::new())
        }
        Some(NumberFormat::F64 | NumberFormat::F64x2) => {
            let p = rlx_opt::rlx_compile::promote_f64::promote_to_f64(&graph);
            (p.graph, p.widened_params)
        }
    }
}
