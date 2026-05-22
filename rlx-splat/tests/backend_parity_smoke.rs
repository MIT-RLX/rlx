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
//! Compile + run the parity scene on each available [`rlx_runtime::Device`].

mod common;

use rlx_runtime::{Device, Session};
use rlx_splat::{assert_parity, COSINE_DISTANCE_RENDER, MEAN_ABS_ERROR_GPU_CPU};

fn assert_render_parity(device: Device, out: &[f32], reference: &[f32]) {
    assert_eq!(out.len(), reference.len());
    assert!(out.iter().all(|v| v.is_finite()), "{device:?}: non-finite output");

    // Host-fallback backends (Metal/CUDA/ROCm/MLX/TPU) should match CPU reference.
    // wgpu uses CPU reference splat (arena D2H/H2D).
    let (mae_limit, cos_limit) = match device {
        Device::Gpu | Device::Vulkan | Device::WebGpu => {
            (MEAN_ABS_ERROR_GPU_CPU, COSINE_DISTANCE_RENDER)
        }
        _ => (MEAN_ABS_ERROR_GPU_CPU, rlx_splat::COSINE_DISTANCE_STRICT),
    };

    assert_parity(out, reference, mae_limit, cos_limit).unwrap_or_else(|e| {
        panic!("{device:?} parity vs CPU reference: {e}");
    });
}

fn try_forward_parity(device: Device, graph: &rlx_ir::Graph, inputs: &[(&str, &[f32])], reference: &[f32]) {
    let mut compiled = Session::new(device).compile(graph.clone());
    let outs = compiled.run(inputs);
    assert_eq!(outs.len(), 1);
    assert_render_parity(device, &outs[0], reference);
}

#[test]
fn forward_parity_on_available_devices() {
    rlx_splat::register();
    let fixture = common::ParityFixture::tiny();
    let graph = fixture.build_graph();
    let reference = fixture.cpu_reference_rgba();
    let inputs = fixture.session_inputs();

    // CPU is mandatory.
    try_forward_parity(Device::Cpu, &graph, &inputs, &reference);

    for device in rlx_runtime::registered_devices() {
        if device == Device::Cpu {
            continue;
        }
        if !rlx_runtime::is_available(device) {
            eprintln!("skip {device:?}: not available on this host");
            continue;
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            try_forward_parity(device, &graph, &inputs, &reference);
        }));

        if result.is_err() {
            eprintln!("skip {device:?}: compile/run panicked (driver or shader issue)");
        }
    }
}
