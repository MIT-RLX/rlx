// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end test for the **raw-GPU** CUDA custom-kernel seam
//! (`CudaGpuKernel`): a downstream `Op::Custom` that NVRTC-compiles a CUDA-C
//! kernel and launches it straight against the arena device buffer — no D2H/H2D
//! host roundtrip (contrast the host-delegate `Step::CustomHost` path). Needs a
//! real NVIDIA GPU (built with the `cuda` feature); runs on a CUDA rig.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use rlx_cuda::cuda_gpu_kernels::{CudaGpuKernel, register_cuda_gpu_kernel};
use rlx_ir::{DType, Graph, OpExtension, Shape, register_op};
use rlx_runtime::{Device, Session, is_available};

/// IR-level shape inference for `test.times_three_cuda` (identity).
struct TimesThreeIr;
impl OpExtension for TimesThreeIr {
    fn name(&self) -> &str {
        "test.times_three_cuda"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

/// CUDA-C with the fixed launch signature (`float* arena`, out/in element
/// offsets). `y = x * 3`.
const CU: &str = r#"
extern "C" __global__ void rlx_custom(
    float* arena,
    unsigned out_off, unsigned out_len, unsigned n_inputs,
    unsigned in0_off, unsigned in0_len,
    unsigned in1_off, unsigned in1_len,
    unsigned in2_off, unsigned in2_len,
    unsigned in3_off, unsigned in3_len,
    unsigned e0, unsigned e1, unsigned e2, unsigned e3)
{
    (void)n_inputs; (void)in1_off; (void)in1_len; (void)in2_off; (void)in2_len;
    (void)in3_off; (void)in3_len; (void)e0; (void)e1; (void)e2; (void)e3;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_len) {
        arena[out_off + i] = arena[in0_off + i] * 3.0f;
    }
}
"#;

/// Raw-GPU CUDA kernel: `y = x * 3`, no host roundtrip.
#[derive(Debug)]
struct TimesThreeCuda;
impl CudaGpuKernel for TimesThreeCuda {
    fn name(&self) -> &str {
        "test.times_three_cuda"
    }
    fn cuda_c(&self) -> &str {
        CU
    }
}

#[test]
fn times_three_runs_on_cuda_via_raw_gpu_kernel() {
    if !is_available(Device::Cuda) {
        eprintln!("skip: no CUDA device");
        return;
    }
    register_op(Arc::new(TimesThreeIr));
    register_cuda_gpu_kernel(Arc::new(TimesThreeCuda));

    let n = 8usize;
    let mut g = Graph::new("cuda_gpu_custom");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let y = g.custom_op("test.times_three_cuda", vec![], vec![x]);
    g.set_outputs(vec![y]);

    // Device::Cuda → the lower path emits Step::CudaGpuKernel (an NVRTC kernel
    // launch against the arena, no host staging).
    let mut c = Session::new(Device::Cuda).compile(g);
    let x0: Vec<f32> = (0..n).map(|i| i as f32 - 3.0).collect();
    let outs = c.run(&[("x", &x0)]);
    let out = &outs[0];

    let expect: Vec<f32> = x0.iter().map(|v| v * 3.0).collect();
    assert_eq!(out.len(), expect.len(), "out={out:?}");
    for (a, b) in out.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-4, "out={out:?} expect={expect:?}");
    }
}
