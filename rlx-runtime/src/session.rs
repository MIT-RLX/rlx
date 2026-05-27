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
        assert!(
            crate::device_ext::is_available(device),
            "device {} is not available — enable the `{}` Cargo feature",
            device,
            feature_name(device)
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
    pub fn compile(&self, graph: Graph) -> CompiledGraph {
        self.compile_module(GraphModule::from_graph(graph))
            .expect("compile MIR graph through fusion pipeline")
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
        let executable = backend.compile_module(module, self.device, options)?;
        Ok(CompiledGraph::new(executable, self.device))
    }

    fn default_options(&self) -> crate::CompileOptions {
        let opts = crate::CompileOptions::new().precision(self.precision);
        match &self.policy {
            Some(p) => opts.policy(p.clone()),
            None => opts,
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

fn feature_name(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal => "metal",
        Device::Mlx => "mlx",
        Device::Ane => "ane",
        Device::Cuda => "cuda",
        Device::Rocm => "rocm",
        Device::Tpu => "tpu",
        Device::Gpu => "gpu",
        Device::Vulkan => "vulkan",
        Device::OpenGl => "opengl",
        Device::DirectX => "directx",
        Device::WebGpu => "webgpu",
    }
}
