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

//! Shared HIR → MIR → LIR compile stages for runtime backends.

use rlx_driver::Device;
use rlx_ir::Graph;
use rlx_ir::GraphModule;
use rlx_ir::OpKind;
use rlx_ir::hir::HirModule;
use rlx_ir::lir::LirModule;
use rlx_opt::{
    CompilePipeline, CompileResult, FusionLimits, FusionOptions, FusionReport, FusionTarget,
    fusion_limits_for_target,
};

use crate::CompileOptions;

/// Map a runtime [`Device`] to the fusion pass pipeline target.
pub fn fusion_target_for(device: Device) -> FusionTarget {
    match device {
        Device::Cpu => FusionTarget::Cpu,
        Device::Metal => FusionTarget::Metal,
        Device::Mlx => FusionTarget::Mlx,
        Device::Cuda => FusionTarget::Cuda,
        Device::Rocm => FusionTarget::Rocm,
        Device::Gpu | Device::Vulkan | Device::WebGpu => FusionTarget::Wgpu,
        Device::Tpu => FusionTarget::Tpu,
        // ANE and other devices without dedicated fusion pipelines use CPU
        // patterns (safe superset for graph-level rewrites).
        _ => FusionTarget::Cpu,
    }
}

/// Build a [`CompilePipeline`] from session/device options.
pub fn pipeline_for(device: Device, options: &CompileOptions) -> CompilePipeline {
    let target = options
        .fusion_target
        .unwrap_or_else(|| fusion_target_for(device));
    let mut opts = options.fusion_opts;
    if matches!(target, FusionTarget::Cpu) && !opts.unfuse_elementwise_regions {
        opts.unfuse_elementwise_regions = true;
    }
    if matches!(target, FusionTarget::Metal) {
        let metal_env = FusionOptions::from_metal_env();
        if !rlx_ir::env::flag("RLX_METAL_NO_FUSION") {
            // Preserve caller overrides; only merge env flags the caller left default.
        }
        if metal_env.skip_fusion {
            opts.skip_fusion = true;
        }
        if metal_env.unfuse_elementwise_regions {
            opts.unfuse_elementwise_regions = true;
        }
    }
    let mut pipe = CompilePipeline::new(target);
    pipe.opts = opts;
    if pipe.opts.fusion_limits == FusionLimits::default() {
        pipe.opts.fusion_limits = fusion_limits_for_target(target);
    }
    pipe.arena_alignment = options.arena_alignment;
    pipe.assert_fusion_clean = options.assert_fusion_clean;
    if let Some(ops) = options.supported_ops {
        pipe.supported_ops = Some(ops);
    } else if let Some(backend) = crate::registry::backend_for(device) {
        let ops = backend.supported_ops();
        if !ops.is_empty() {
            pipe.supported_ops = Some(ops);
        }
    }
    pipe.kernel_dispatch = options.kernel_dispatch;
    pipe
}

/// Attach a backend op claim set for backend-aware fusion.
pub fn options_with_supported_ops(
    options: &CompileOptions,
    supported_ops: &'static [OpKind],
) -> CompileOptions {
    let mut opts = options.clone();
    opts.supported_ops = Some(supported_ops);
    opts
}

/// Run the MIR fusion pipeline on a graph and return optimized LIR +
/// fusion diagnostics.
pub fn compile_graph_stages(
    device: Device,
    graph: Graph,
    options: &CompileOptions,
) -> CompileResult {
    let pipe = pipeline_for(device, options);
    maybe_specialize(pipe.compile_graph(graph), &pipe, options)
}

fn maybe_specialize(
    result: CompileResult,
    pipe: &CompilePipeline,
    options: &CompileOptions,
) -> CompileResult {
    match &options.dim_binding {
        Some(binding) => result.specialize(pipe, binding),
        None => result,
    }
}

/// Same as [`compile_graph_stages`] with an explicit backend op claim.
pub fn compile_graph_stages_for_backend(
    device: Device,
    graph: Graph,
    options: &CompileOptions,
    supported_ops: &'static [OpKind],
) -> CompileResult {
    let opts = options_with_supported_ops(options, supported_ops);
    compile_graph_stages(device, graph, &opts)
}

/// HIR → LIR with fusion diagnostics.
pub fn compile_hir_stages(
    device: Device,
    hir: HirModule,
    options: &CompileOptions,
) -> Result<CompileResult, rlx_ir::hir::LowerError> {
    let pipe = pipeline_for(device, options);
    pipe.compile_hir(hir).map(|r| maybe_specialize(r, &pipe, options))
}

/// Compile a [`GraphModule`] (HIR, MIR, or LIR stage) through the pipeline.
pub fn compile_module_stages(
    device: Device,
    module: GraphModule,
    options: &CompileOptions,
) -> Result<CompileResult, rlx_ir::hir::LowerError> {
    let pipe = pipeline_for(device, options);
    pipe.compile_module(module)
        .map(|r| maybe_specialize(r, &pipe, options))
}

/// Print fusion diagnostics when `RLX_FUSION_REPORT=1`.
pub fn maybe_log_fusion(report: &FusionReport) {
    if rlx_ir::env::flag("RLX_FUSION_REPORT") {
        eprintln!("{report}");
    }
}

/// Extract the optimized graph from LIR (convenience for backends that
/// replan memory after precision rewrites).
pub fn graph_from_lir(lir: LirModule) -> Graph {
    lir.into_graph()
}
