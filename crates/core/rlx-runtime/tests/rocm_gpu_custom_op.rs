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

//! End-to-end test for the **raw-GPU** ROCm custom-kernel seam
//! (`RocmGpuKernel`): a downstream `Op::Custom` that hipRTC-compiles a HIP-C
//! kernel and launches it straight against the arena device buffer — no host
//! roundtrip. Needs a real AMD/ROCm GPU; like the other rlx-rocm tests it
//! **early-returns when no ROCm device is present** (so it is a no-op on macOS /
//! CI and exercises the kernel only on an AMD box).

#![cfg(feature = "rocm")]

use std::sync::Arc;

use rlx_ir::{DType, Graph, OpExtension, Shape, register_op};
use rlx_rocm::rocm_gpu_kernels::{RocmGpuKernel, register_rocm_gpu_kernel};
use rlx_runtime::{Device, Session};

/// IR-level shape inference for `test.times_three_rocm` (identity).
struct TimesThreeIr;
impl OpExtension for TimesThreeIr {
    fn name(&self) -> &str {
        "test.times_three_rocm"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

/// HIP-C with the fixed launch signature (`float* arena` + element offsets).
/// `y = x * 3`. HIP compiles the same source shape as CUDA.
const HIP: &str = r#"
extern "C" __global__ void rlx_custom(
    float* arena,
    unsigned out_off, unsigned out_len, unsigned n_inputs,
    unsigned in0_off, unsigned in0_len,
    unsigned in1_off, unsigned in1_len,
    unsigned in2_off, unsigned in2_len,
    unsigned in3_off, unsigned in3_len)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_len) {
        arena[out_off + i] = arena[in0_off + i] * 3.0f;
    }
}
"#;

/// Raw-GPU ROCm kernel: `y = x * 3`, no host roundtrip.
#[derive(Debug)]
struct TimesThreeRocm;
impl RocmGpuKernel for TimesThreeRocm {
    fn name(&self) -> &str {
        "test.times_three_rocm"
    }
    fn hip_c(&self) -> &str {
        HIP
    }
}

#[test]
fn times_three_runs_on_rocm_via_raw_gpu_kernel() {
    if !rlx_rocm::is_available() {
        eprintln!("skipping rocm_gpu_custom_op: no ROCm device");
        return;
    }
    register_op(Arc::new(TimesThreeIr));
    register_rocm_gpu_kernel(Arc::new(TimesThreeRocm));

    let n = 8usize;
    let mut g = Graph::new("rocm_gpu_custom");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let y = g.custom_op("test.times_three_rocm", vec![], vec![x]);
    g.set_outputs(vec![y]);

    // Device::Rocm → the lower path emits Step::RocmGpuKernel (a hipRTC kernel
    // launch against the arena, no host staging).
    let mut c = Session::new(Device::Rocm).compile(g);
    let x0: Vec<f32> = (0..n).map(|i| i as f32 - 3.0).collect();
    let outs = c.run(&[("x", &x0)]);
    let out = &outs[0];

    let expect: Vec<f32> = x0.iter().map(|v| v * 3.0).collect();
    assert_eq!(out.len(), expect.len(), "out={out:?}");
    for (a, b) in out.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-4, "out={out:?} expect={expect:?}");
    }
}
