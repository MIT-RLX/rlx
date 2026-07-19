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

//! End-to-end test for the **raw-GPU** wgpu custom-kernel seam
//! (`WgpuGpuKernel`): a downstream `Op::Custom` that dispatches a real WGSL
//! compute kernel straight against the arena buffer — no D2H/H2D host roundtrip
//! (contrast the host-delegate `Step::CustomHost` path). Runs on the portable
//! wgpu backend (`Device::Gpu`), which is Metal on macOS / Vulkan on Linux /
//! DX12 on Windows / WebGPU in the browser.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use rlx_ir::{DType, Graph, OpExtension, Shape, register_op};
use rlx_runtime::{Device, Session};
use rlx_wgpu::wgpu_gpu_custom::{WgpuGpuKernel, register_wgpu_gpu_kernel};

/// IR-level shape inference for `test.times_three_wgpu` (identity).
struct TimesThreeIr;
impl OpExtension for TimesThreeIr {
    fn name(&self) -> &str {
        "test.times_three_wgpu"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

/// WGSL following the fixed binding convention: arena storage @0, params @1.
/// `params = [out_off, out_len, n_inputs, _pad, in0_off, in0_len]`.
const WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read>       params: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_off = params[0];
    let out_len = params[1];
    let in0_off = params[4];
    if (i < out_len) {
        arena[out_off + i] = arena[in0_off + i] * 3.0;
    }
}
"#;

/// Raw-GPU wgpu kernel: `y = x * 3`, dispatched with no host roundtrip.
#[derive(Debug)]
struct TimesThreeWgpu;
impl WgpuGpuKernel for TimesThreeWgpu {
    fn name(&self) -> &str {
        "test.times_three_wgpu"
    }
    fn wgsl(&self) -> &str {
        WGSL
    }
}

#[test]
fn times_three_runs_on_wgpu_via_raw_gpu_kernel() {
    register_op(Arc::new(TimesThreeIr));
    register_wgpu_gpu_kernel(Arc::new(TimesThreeWgpu));

    let n = 8usize;
    let mut g = Graph::new("wgpu_gpu_custom");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let y = g.custom_op("test.times_three_wgpu", vec![], vec![x]);
    g.set_outputs(vec![y]);

    // Device::Gpu → the lower path recognizes the registered GPU kernel and
    // emits Step::WgpuGpuKernel (a compute-pass dispatch, no host staging).
    let mut c = Session::new(Device::Gpu).compile(g);
    let x0: Vec<f32> = (0..n).map(|i| i as f32 - 3.0).collect();
    let outs = c.run(&[("x", &x0)]);
    let out = &outs[0];

    let expect: Vec<f32> = x0.iter().map(|v| v * 3.0).collect();
    assert_eq!(out.len(), expect.len(), "out={out:?}");
    for (a, b) in out.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-4, "out={out:?} expect={expect:?}");
    }
}
