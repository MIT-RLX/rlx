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

/// Check whether `device` has a compiled-in backend or has been
/// registered by an external crate.
///
/// Two-step check: first the cargo-feature gate (cheap, compile
/// time), then the registry (catches external backends that
/// called `register_backend`).
pub fn is_available(device: Device) -> bool {
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

use rlx_ir::{Graph, Op};

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
        // Other backends not yet characterised here. Conservative:
        // assume `false` so callers won't dispatch blind; tighten as
        // each backend grows a `<x>_supports` arm below.
        _ => false,
    }
}

/// Is every op in `graph` lowerable by `device`? Short-circuits on
/// the first unsupported op. Use this in front of `Session::compile`
/// when you want a clean fallback rather than a runtime panic.
pub fn supports_graph(device: Device, graph: &Graph) -> bool {
    graph.nodes().iter().all(|n| supports(device, &n.op))
}

/// First op in `graph` that `device` cannot lower, or `None` if every
/// op is supported. Returns the node index for diagnostics ("MLX
/// can't lower node 42: `Activation(Sin)`").
pub fn first_unsupported_op(device: Device, graph: &Graph) -> Option<(usize, &Op)> {
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
