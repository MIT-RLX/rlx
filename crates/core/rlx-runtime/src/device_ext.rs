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

/// Preferred probe order for ML workloads (highest throughput first).
///
/// Used by [`fastest_device`] and by [`crate::cost::fastest_device_for`] when
/// calibrated cost models are unavailable for every candidate backend.
pub(crate) const DEVICE_PRIORITY: &[Device] = &[
    Device::Tpu,
    Device::Cuda,
    Device::Rocm,
    Device::OneApi,
    Device::Mlx,
    Device::Metal,
    Device::Ane,
    Device::Hexagon,
    Device::Gpu,
    Device::Vulkan,
    Device::DirectX,
    Device::OpenGl,
    Device::WebGpu,
    Device::Cpu,
];

/// Browser backend probe order: WebGPU first, WebGL fallback, then CPU.
pub const BROWSER_DEVICE_PRIORITY: &[Device] = &[Device::WebGpu, Device::OpenGl, Device::Cpu];

/// Check whether `device` has a compiled-in backend or has been
/// registered by an external crate.
///
/// GPU-family builtins (CUDA / ROCm / wgpu / TPU) additionally probe
/// for a live driver or adapter at runtime so CI hosts that compile
/// with `--features cuda` but have no NVIDIA stack don't report
/// false positives. Other devices are Cargo-feature-gated; externally
/// registered backends are discovered via the registry.
/// Whether [`crate::CompiledGraph::run_slots`] + [`crate::CompiledGraph::arena_ptr`]
/// are implemented (host readback layout; not a GPU-mapped arena on CUDA).
pub fn supports_run_slots(device: Device) -> bool {
    matches!(
        device,
        Device::Cpu | Device::Metal | Device::Mlx | Device::Cuda | Device::Rocm
    )
}

/// Whether `device`'s RoPE kernel indexes its cos/sin table **per token**
/// (one row per batch·seq element) instead of per sequence position.
///
/// Required for *ragged* batched decode, where each sequence in the batch sits
/// at a different absolute position and so needs its own RoPE row.
///
/// Validated against the CPU reference on:
///   - **CPU** (`rlx-cpu` executor + thunk), and
///   - **Metal** (`cos_per_token` kernel path; `metal_rope_parity` ragged test).
///
/// The remaining GPU kernels still index by seq position (CUDA `rope.cu`, wgpu
/// `rope.wgsl`), which collapses for decode (seq = 1) and would apply row 0's
/// position to the whole batch — so callers (e.g. the server's fused batcher)
/// fall back to per-length **uniform** grouping there. Add a device here only
/// once its rope + rope_backward kernels are fixed *and* validated against the
/// CPU reference.
pub fn supports_ragged_rope(device: Device) -> bool {
    matches!(device, Device::Cpu | Device::Metal)
}

/// Drop backend-held device arena pools between large compiles (CUDA).
/// No-op on backends without a pool (CPU, Metal unified memory, ROCm, …).
pub fn trim_accelerator_arena_pool(device: Device) {
    #[cfg(feature = "cuda")]
    if device == Device::Cuda {
        rlx_cuda::trim_device_memory_pool();
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
}

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
        #[cfg(target_arch = "wasm32")]
        {
            return rlx_wgpu::device::wgpu_device().is_some();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            return rlx_wgpu::is_available();
        }
    }
    #[cfg(feature = "webgpu")]
    if device == Device::WebGpu {
        #[cfg(target_arch = "wasm32")]
        {
            return rlx_wgpu::device::wgpu_device().is_some();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            return rlx_wgpu::is_available();
        }
    }
    #[cfg(feature = "opengl")]
    if device == Device::OpenGl {
        return true;
    }
    #[cfg(feature = "vulkan")]
    if device == Device::Vulkan {
        return rlx_vulkan::is_available();
    }
    #[cfg(feature = "oneapi")]
    if device == Device::OneApi {
        return rlx_oneapi::is_available();
    }
    #[cfg(feature = "tpu")]
    if device == Device::Tpu {
        return rlx_tpu::is_available();
    }
    // Metal / MLX probe the live device, not just the Cargo cfg: a build can
    // be Metal-capable yet run where no Metal device exists (e.g. a headless
    // iOS-simulator process), and runtime selection must not pick a dead
    // backend.
    #[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
    if device == Device::Metal {
        return rlx_metal::is_available();
    }
    #[cfg(all(feature = "mlx", rlx_mlx_host))]
    if device == Device::Mlx {
        return rlx_mlx::is_available();
    }

    let feature_gated = match device {
        Device::Cpu => cfg!(feature = "cpu"),
        Device::Metal => cfg!(all(
            feature = "metal",
            target_vendor = "apple",
            not(target_os = "watchos")
        )),
        Device::Mlx => cfg!(feature = "mlx"),
        Device::Ane => cfg!(any(feature = "coreml", feature = "ane")),
        Device::Cuda => cfg!(feature = "cuda"),
        Device::Rocm => cfg!(feature = "rocm"),
        Device::OneApi => cfg!(feature = "oneapi"),
        Device::Tpu => cfg!(feature = "tpu"),
        Device::Hexagon => cfg!(feature = "qnn"),
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

/// Apple backends enabled in this build (`metal`, `mlx`, `gpu`, `ane`) on
/// any Apple platform. `is_available` filters per-device, so e.g. watchOS
/// (no Metal) yields just the CoreML/ANE entry.
#[cfg(all(feature = "apple", target_vendor = "apple"))]
pub fn available_apple_devices() -> Vec<Device> {
    [Device::Metal, Device::Mlx, Device::Gpu, Device::Ane]
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

/// Browser backends currently runnable (`WebGpu` → `OpenGl` → `Cpu`).
pub fn available_browser_devices() -> Vec<Device> {
    BROWSER_DEVICE_PRIORITY
        .iter()
        .copied()
        .filter(|d| is_available(*d))
        .collect()
}

/// Highest-priority browser backend, or `None` when no browser path is live.
pub fn preferred_browser_device() -> Option<Device> {
    available_browser_devices().into_iter().next()
}

/// Intersection of [`available_devices`] and [`supports_graph`]. Use with
/// [`crate::GraphDevices`] or [`crate::DevicePolicy`] to restrict the set.
pub fn devices_for(graph: &Graph) -> Vec<Device> {
    crate::device_policy::devices_for_with_policy(graph, &crate::DevicePolicy::default())
}

/// Highest-priority backend that is compiled in and live on this host.
///
/// Probes [`DEVICE_PRIORITY`] in order (TPU → CUDA → ROCm → MLX → Metal → …
/// → CPU). Use this when you want a sensible default `Session` target without
/// building a graph first. For workload-specific selection, prefer
/// [`crate::cost::fastest_device_for`].
pub fn fastest_device() -> Device {
    fastest_among(&available_devices())
}

/// Pick the highest-priority entry from `candidates` (see [`DEVICE_PRIORITY`]).
pub fn fastest_among(candidates: &[Device]) -> Device {
    for &d in DEVICE_PRIORITY {
        if candidates.contains(&d) {
            return d;
        }
    }
    candidates.first().copied().unwrap_or(Device::Cpu)
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
        Device::Ane => coreml_supports(op),
        Device::Gpu | Device::Cuda | Device::Rocm => gpu_family_supports(op),
        #[cfg(feature = "vulkan")]
        Device::Vulkan => vulkan_supports(op),
        #[cfg(feature = "oneapi")]
        Device::OneApi => oneapi_supports(op),
        Device::Hexagon => qnn_supports(op),
        // Other backends not yet characterised here. Conservative:
        // assume `false` so callers won't dispatch blind; tighten as
        // each backend grows a `<x>_supports` arm below.
        _ => false,
    }
}

/// Per-op support for the QNN (Hexagon NPU) backend — the ops the FFI runtime
/// (`rlx_qnn::runtime`) lowers to QNN, plus fused forms decomposed at compile
/// (`FusedAttentionBlock`).
fn qnn_supports(op: &Op) -> bool {
    use rlx_ir::op::Activation;
    match op {
        Op::Input { .. }
        | Op::Param { .. }
        | Op::Constant { .. }
        | Op::MatMul
        | Op::Binary(_)
        | Op::Softmax { .. }
        | Op::Reshape { .. }
        | Op::Transpose { .. }
        | Op::LayerNorm { .. }
        | Op::RmsNorm { .. }
        | Op::Concat { .. }
        | Op::Narrow { .. }
        | Op::Rope { .. }
        | Op::Attention { .. }
        | Op::FusedAttentionBlock { .. }
        | Op::Expand { .. }
        | Op::Reduce { .. }
        | Op::Conv { .. }
        | Op::Gather { .. }
        | Op::Quantize { .. }
        | Op::Dequantize { .. }
        | Op::DequantMatMul { .. } => true,
        Op::Activation(a) => matches!(
            a,
            Activation::Relu
                | Activation::Gelu
                | Activation::Sigmoid
                | Activation::Tanh
                | Activation::Neg
                | Activation::Silu
        ),
        _ => false,
    }
}

/// Per-op heuristic for the native Vulkan backend: native primitive set plus
/// the op families `legalize_or_rewrite_for_backend` decomposes into it. The
/// authoritative check is `supports_graph` (runs the real legalize probe);
/// this is the cheap single-op approximation.
#[cfg(feature = "vulkan")]
fn vulkan_supports(op: &Op) -> bool {
    use rlx_ir::OpKind::*;
    let k = op.kind();
    rlx_vulkan::backend::SUPPORTED_OPS.contains(&k)
        || matches!(
            k,
            // Decomposed to the primitive set by the rewrite pass.
            DotGeneral
                | Fma
                | GroupNorm
                | BatchNormInference
                | ResizeNearest2x
                | ElementwiseRegion
                | FusedMatMulBiasAct
                | FusedResidualLN
                | FusedResidualRmsNorm
                | FusedSwiGLU
                | FusedAttentionBlock
                | FusedTransformerLayer
        )
}

/// Per-op heuristic for the native oneAPI (Level Zero) backend — the same
/// primitive claim set as rlx-vulkan (the rewrite pass decomposes the rest).
/// The authoritative check remains `supports_graph` (real legalize probe).
#[cfg(feature = "oneapi")]
fn oneapi_supports(op: &Op) -> bool {
    use rlx_ir::OpKind::*;
    let k = op.kind();
    rlx_oneapi::backend::SUPPORTED_OPS.contains(&k)
        || matches!(
            k,
            DotGeneral
                | Fma
                | GroupNorm
                | BatchNormInference
                | ResizeNearest2x
                | ElementwiseRegion
                | FusedMatMulBiasAct
                | FusedResidualLN
                | FusedResidualRmsNorm
                | FusedSwiGLU
                | FusedAttentionBlock
                | FusedTransformerLayer
        )
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

/// CoreML / ANE lowers a fixed, declared op set (see `rlx_coreml::mil`).
/// Unlike the GPU backends — whose lowering covers the whole IR surface —
/// CoreML is an inference compiler with a finite op claim, so we check
/// membership directly against the backend's published list.
///
/// Under the `training` feature the claim also covers the backward ops that the
/// legalize/rewrite pass decomposes into supported primitives (or, once landed,
/// lower through native MIL backward kernels) — so device selection picks
/// `Device::Ane` for autodiff-produced backward graphs. See
/// [`rlx_coreml::BACKWARD_OPS`].
fn coreml_supports(op: &Op) -> bool {
    #[cfg(feature = "coreml")]
    {
        let kind = op.kind();
        if rlx_coreml::SUPPORTED_OPS.contains(&kind) {
            return true;
        }
        #[cfg(feature = "training")]
        if rlx_coreml::BACKWARD_OPS.contains(&kind)
            || rlx_coreml::NATIVE_BACKWARD_OPS.contains(&kind)
        {
            return true;
        }
        false
    }
    #[cfg(not(feature = "coreml"))]
    {
        let _ = op;
        false
    }
}

#[allow(unused_variables)]
fn gpu_family_supports(op: &Op) -> bool {
    // CUDA / ROCm / wgpu share the same IR surface area as CPU for the
    // ops V-JEPA2 and other vision models exercise. Narrow when a backend
    // reports a concrete lowering gap.
    let _ = op;
    true
}

/// Block until `device`'s queue is idle. Metal drains the global queue;
/// other backends are no-ops.
pub fn drain_device(device: Device) {
    #[cfg(all(target_vendor = "apple", not(target_os = "watchos"), feature = "metal"))]
    {
        if device == Device::Metal {
            rlx_metal::device::drain_command_queue();
        }
    }
    #[cfg(not(all(target_vendor = "apple", not(target_os = "watchos"), feature = "metal")))]
    let _ = device;
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
    #[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
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

    #[test]
    fn fastest_device_returns_cpu_when_only_cpu_is_available() {
        let pick = fastest_device();
        assert!(is_available(pick));
        assert_eq!(pick, fastest_among(&available_devices()));
    }

    #[test]
    fn fastest_among_respects_priority_order() {
        let pick = fastest_among(&[Device::Cpu, Device::Metal, Device::Mlx]);
        assert_eq!(pick, Device::Mlx);
    }

    #[test]
    fn devices_for_is_subset_of_available() {
        let mut g = Graph::new("id");
        let x = g.input("x", scalar_shape());
        g.set_outputs(vec![x]);
        for d in devices_for(&g) {
            assert!(is_available(d));
            assert!(supports_graph(d, &g));
        }
    }
}
