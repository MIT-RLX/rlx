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

//! Engine-layer extensions for [`rlx_driver::Device`] (plan #58).
//!
//! `is_available` and `available_devices` consult the runtime's
//! backend registry + Cargo features, both of which are
//! engine-layer concerns. Keeping them here preserves the
//! one-way dep direction (driver doesn't know about engine).

use rlx_driver::Device;
use rlx_ir::{Graph, Op};

use crate::CompileOptions;

/// Check whether `device` has a compiled-in backend or has been
/// registered by an external crate.
///
/// GPU-family builtins (CUDA / ROCm / wgpu / TPU) additionally probe
/// for a live driver or adapter at runtime so CI hosts that compile
/// with `--features cuda` but have no NVIDIA stack don't report
/// false positives. Other devices are Cargo-feature-gated; externally
/// registered backends are discovered via the registry.
pub fn is_available(device: Device) -> bool {
    #[cfg(feature = "cuda")]
    if device == Device::Cuda {
        return rlx_cuda::is_available();
    }
    #[cfg(feature = "rocm")]
    if device == Device::Rocm {
        return rlx_rocm::is_available();
    }
    #[cfg(feature = "gpu")]
    if device == Device::Gpu {
        return rlx_wgpu::is_available();
    }
    #[cfg(feature = "vulkan")]
    if device == Device::Vulkan {
        return rlx_wgpu::is_vulkan_available();
    }
    #[cfg(feature = "tpu")]
    if device == Device::Tpu {
        return rlx_tpu::is_available();
    }

    let feature_gated = match device {
        Device::Cpu => cfg!(feature = "cpu"),
        Device::Metal => cfg!(feature = "metal"),
        Device::Mlx => cfg!(feature = "mlx"),
        Device::Ane => cfg!(feature = "ane"),
        Device::Cuda => cfg!(feature = "cuda"),
        Device::Rocm => cfg!(feature = "rocm"),
        Device::Tpu => cfg!(feature = "tpu"),
        Device::Gpu => cfg!(feature = "gpu"),
        Device::Vulkan => cfg!(feature = "vulkan"),
        Device::OpenGl => cfg!(feature = "opengl"),
        Device::DirectX => cfg!(feature = "directx"),
        Device::WebGpu => cfg!(feature = "webgpu"),
    };
    if feature_gated {
        return true;
    }
    crate::registry::registered_devices().contains(&device)
}

/// Apple backends enabled in this build (`metal`, `mlx`, `gpu` on macOS).
#[cfg(all(feature = "apple", target_os = "macos"))]
pub fn available_apple_devices() -> Vec<Device> {
    [Device::Metal, Device::Mlx, Device::Gpu]
        .into_iter()
        .filter(|d| is_available(*d))
        .collect()
}

/// Every variant currently available — Cargo-feature-gated or
/// runtime-registered.
pub fn available_devices() -> Vec<Device> {
    Device::all()
        .iter()
        .copied()
        .filter(|d| is_available(*d))
        .collect()
}

/// Pretty name with engine-known BLAS variant for the CPU device.
/// Gives `"CPU (Accelerate)"` etc. when the relevant feature is
/// on; falls back to the bare driver-side `Device::name()` when
/// no BLAS feature is selected.
pub fn full_name(device: Device) -> &'static str {
    if let Device::Cpu = device {
        if cfg!(feature = "blas-accelerate") {
            return "CPU (Accelerate)";
        }
        if cfg!(feature = "blas-mkl") {
            return "CPU (MKL)";
        }
        if cfg!(feature = "blas-openblas") {
            return "CPU (OpenBLAS)";
        }
    }
    device.name()
}

// ── Per-device op-support introspection ──────────────────────────
//
// Callers that want to dispatch graphs to a particular device need
// to know up front whether the device's backend has every op the
// graph uses wired up. Before this API, the only signal was a
// runtime panic ("not yet implemented"), which forced downstream
// crates (e.g. `eda-magnetics::graph::pick_device_for`) to bake
// hand-maintained "what's missing on X" tables into their own
// source — those drift the moment a backend lands the missing op.
//
// [`supports`] consults the backend-side knowledge (CPU is the
// reference and assumed complete; MLX / Metal each name the ops
// they don't yet lower) so consumers can ask once and stop
// re-implementing the table.

/// Is `op` lowerable by the backend for `device` *in this build*?
///
/// - CPU is the reference; always returns `true`.
/// - GPU backends return `false` only for the specific ops/variants
///   their lowering currently rejects. As backends close gaps, the
///   matches here shrink and consumers automatically pick them up.
/// - For devices not feature-gated in, returns `false` (you can't
///   dispatch to a backend that isn't compiled in regardless).
pub fn supports(device: Device, op: &Op) -> bool {
    if !is_available(device) {
        return false;
    }
    match device {
        Device::Cpu => true, // reference backend; ground truth
        Device::Mlx => mlx_supports(op),
        Device::Metal => metal_supports(op),
        Device::Gpu | Device::Cuda | Device::Rocm => gpu_family_supports(op),
        // Other backends not yet characterised here. Conservative:
        // assume `false` so callers won't dispatch blind; tighten as
        // each backend grows a `<x>_supports` arm below.
        _ => false,
    }
}

/// Is every op in `graph` lowerable by `device`?
///
/// When a backend is registered, uses the same rewrite + legalization probe as
/// [`legalize_graph_for_device`] (see [`KernelDispatchReport::compile_ready`]).
/// Otherwise falls back to per-op [`supports`] heuristics.
pub fn supports_graph(device: Device, graph: &Graph) -> bool {
    supports_graph_with_options(device, graph, &CompileOptions::default())
}

/// Like [`supports_graph`] with explicit [`CompileOptions::kernel_dispatch`].
pub fn supports_graph_with_options(
    device: Device,
    graph: &Graph,
    options: &CompileOptions,
) -> bool {
    if !is_available(device) {
        return false;
    }
    if let Some(backend) = crate::registry::backend_for(device) {
        let (_, report) = rlx_opt::prepare_graph_for_backend_with_report(
            graph.clone(),
            device.name(),
            backend.supported_ops(),
            options.kernel_dispatch,
        );
        return report.compile_ready;
    }
    graph.nodes().iter().all(|n| supports(device, &n.op))
}

/// Legalize `graph` for `device` using that backend's claimed [`OpKind`] set.
///
/// Applies the same rewrite + legalization path as [`Backend::compile`] (e.g.
/// CUDA/ROCm rewrites before the legality check). Returns an error when the
/// backend feature is not enabled or the graph contains unsupported ops.
///
/// Does not require a live GPU/TPU driver — only that the backend crate is
/// compiled in.
pub fn legalize_graph_for_device(graph: Graph, device: Device) -> Result<Graph, String> {
    let (graph, _report) = legalize_graph_for_device_with_report(graph, device)?;
    Ok(graph)
}

/// Like [`legalize_graph_for_device`] but returns a [`KernelDispatchReport`] for tooling.
pub fn legalize_graph_for_device_with_report(
    graph: Graph,
    device: Device,
) -> Result<(Graph, rlx_opt::KernelDispatchReport), String> {
    legalize_graph_for_device_with_options(graph, device, &CompileOptions::default())
}

/// Like [`legalize_graph_for_device_with_report`] using [`CompileOptions::kernel_dispatch`]
/// (and the same rewrite path as [`Backend::compile`]).
pub fn legalize_graph_for_device_with_options(
    graph: Graph,
    device: Device,
    options: &CompileOptions,
) -> Result<(Graph, rlx_opt::KernelDispatchReport), String> {
    let backend = crate::registry::backend_for(device).ok_or_else(|| {
        format!(
            "no backend registered for {device:?} — enable the matching \
             `rlx-runtime` Cargo feature (e.g. `metal`, `gpu`, `cuda`)"
        )
    })?;
    let ops = backend.supported_ops();
    let (graph, report) = rlx_opt::prepare_graph_for_backend_with_report(
        graph,
        device.name(),
        ops,
        options.kernel_dispatch,
    );
    if !report.compile_ready {
        return Err(format!(
            "{}\n{}",
            rlx_opt::format_legalize_error(device.name(), &report.still_unsupported),
            rlx_opt::format_dispatch_report(&report)
        ));
    }
    Ok((graph, report))
}

/// Dispatch report for `graph` on `device` without mutating the graph (static common-ir probe).
pub fn dispatch_report_for_device(
    graph: &Graph,
    device: Device,
) -> Result<rlx_opt::KernelDispatchReport, String> {
    dispatch_report_for_device_with_options(graph, device, &CompileOptions::default())
}

/// Like [`dispatch_report_for_device`] with explicit [`CompileOptions::kernel_dispatch`].
pub fn dispatch_report_for_device_with_options(
    graph: &Graph,
    device: Device,
    options: &CompileOptions,
) -> Result<rlx_opt::KernelDispatchReport, String> {
    let backend = crate::registry::backend_for(device)
        .ok_or_else(|| format!("no backend registered for {device:?}"))?;
    Ok(rlx_opt::analyze_dispatch(
        graph,
        device.name(),
        backend.supported_ops(),
        options.kernel_dispatch,
    ))
}

/// First op in `graph` that `device` cannot lower after rewrite, or `None`.
///
/// Prefer the backend claim-set probe when registered; otherwise [`supports`].
pub fn first_unsupported_op(device: Device, graph: &Graph) -> Option<(usize, &Op)> {
    first_unsupported_op_with_options(device, graph, &CompileOptions::default())
}

/// Like [`first_unsupported_op`] with explicit [`CompileOptions::kernel_dispatch`].
pub fn first_unsupported_op_with_options<'a>(
    device: Device,
    graph: &'a Graph,
    options: &CompileOptions,
) -> Option<(usize, &'a Op)> {
    if !is_available(device) {
        return graph.nodes().first().map(|n| (0, &n.op));
    }
    if let Some(backend) = crate::registry::backend_for(device) {
        let (_, report) = rlx_opt::prepare_graph_for_backend_with_report(
            graph.clone(),
            device.name(),
            backend.supported_ops(),
            options.kernel_dispatch,
        );
        if let Some((id, kind)) = report.still_unsupported.first() {
            let idx = graph.nodes().iter().position(|n| n.id == *id).unwrap_or(0);
            let op = graph
                .nodes()
                .iter()
                .find(|n| n.id == *id)
                .map(|n| &n.op)
                .unwrap_or(&graph.nodes()[0].op);
            let _ = kind;
            return Some((idx, op));
        }
        return None;
    }
    graph
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(i, n)| (!supports(device, &n.op)).then_some((i, &n.op)))
}

#[allow(unused_variables)]
fn mlx_supports(op: &Op) -> bool {
    // After Sin/Cos wiring (forward + backward), MLX's `Activation`
    // dispatch is complete for every variant in `rlx_ir::Activation`.
    // Add narrow guards here only when a future Op or Activation
    // variant lands without an MLX lowering.
    true
}

#[allow(unused_variables)]
fn metal_supports(op: &Op) -> bool {
    // No characterized gaps for the activations rlx-eda exercises.
    // The Sin/Cos/Tan/Atan MSL kernels landed in `rlx-metal/src/kernels.rs`
    // (`{sin,cos,tan,atan}_inplace`) alongside the dispatch slots in
    // `backend.rs:1764`. Narrow this back down if a future Op or
    // Activation variant lands without a Metal kernel.
    let _ = op;
    true
}

#[allow(unused_variables)]
fn gpu_family_supports(op: &Op) -> bool {
    // CUDA / ROCm / wgpu share the same IR surface area as CPU for the
    // ops V-JEPA2 and other vision models exercise. Narrow when a backend
    // reports a concrete lowering gap.
    let _ = op;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::{DType, Graph, Shape};

    fn scalar_shape() -> Shape {
        Shape::new(&[1], DType::F32)
    }

    #[test]
    fn cpu_supports_everything_built_in() {
        assert!(supports(Device::Cpu, &Op::Activation(Activation::Sin)));
        assert!(supports(Device::Cpu, &Op::Activation(Activation::Cos)));
        assert!(supports(Device::Cpu, &Op::Activation(Activation::Exp)));
        assert!(supports(Device::Cpu, &Op::Binary(BinaryOp::Add)));
    }

    #[test]
    fn unbuilt_device_supports_nothing() {
        // OpenGl isn't a workspace feature; should report false.
        assert!(!supports(Device::OpenGl, &Op::Activation(Activation::Relu)));
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_supports_full_activation_set() {
        // After the {sin,cos,tan,atan}_inplace MSL kernels landed in
        // rlx-metal/src/kernels.rs, Metal has every Activation variant
        // rlx-eda exercises.
        for act in [
            Activation::Sin,
            Activation::Cos,
            Activation::Tan,
            Activation::Atan,
            Activation::Exp,
        ] {
            assert!(
                supports(Device::Metal, &Op::Activation(act)),
                "Metal should support Activation::{act:?}"
            );
        }
    }

    #[test]
    fn graph_walk_reports_first_blocker() {
        let mut g = Graph::new("walk");
        let s = scalar_shape();
        let x = g.input("x", s.clone());
        let _e = g.activation(Activation::Exp, x, s.clone());
        let _sin = g.activation(Activation::Sin, x, s);
        // CPU always supports.
        assert!(supports_graph(Device::Cpu, &g));
        assert!(first_unsupported_op(Device::Cpu, &g).is_none());
    }
}
