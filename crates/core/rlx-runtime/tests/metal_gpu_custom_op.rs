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

//! End-to-end test for the **raw-GPU** custom-kernel seam (`MetalGpuKernel`): a
//! downstream `Op::Custom` that dispatches a real MSL kernel onto the active
//! compute encoder with NO host roundtrip / queue sync. Runs on real Apple
//! Silicon (built with the `metal` feature). Contrast `metal_sparse_ops.rs`,
//! which exercises the host-delegate `MetalKernel` path.

#![cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use rlx_ir::{DType, Graph, OpExtension, Shape, register_op};
use rlx_metal::op_registry::{MetalGpuDispatch, MetalGpuKernel, register_metal_gpu_kernel};
use rlx_runtime::{Device, Session};

/// IR-level extension: shape inference for `test.times_three_gpu` (identity).
struct TimesThreeIr;
impl OpExtension for TimesThreeIr {
    fn name(&self) -> &str {
        "test.times_three_gpu"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

/// Raw-GPU Metal kernel: `y = x * 3`, dispatched straight onto the arena buffer.
/// Written exactly as a downstream crate would: compile a pipeline via the
/// public rlx-metal helpers, then bind arena sub-buffers + dispatch on the
/// provided encoder — no host copy, no `commit`/`wait`.
#[derive(Debug)]
struct TimesThreeMetal;

impl MetalGpuKernel for TimesThreeMetal {
    fn name(&self) -> &str {
        "test.times_three_gpu"
    }

    fn encode(&self, d: &MetalGpuDispatch) -> Result<(), String> {
        const SRC: &str = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void times_three(
                device const float* x [[buffer(0)]],
                device float* y [[buffer(1)]],
                constant uint& n [[buffer(2)]],
                uint gid [[thread_position_in_grid]])
            {
                if (gid < n) { y[gid] = x[gid] * 3.0f; }
            }
        "#;
        let dev = rlx_metal::device::metal_device().ok_or("no metal device")?;
        let lib = rlx_metal::pipeline_cache::load_or_compile_library(&dev.device, SRC);
        let func = lib
            .get_function("times_three", None)
            .map_err(|e| format!("get_function: {e}"))?;
        let pipe = dev
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("pipeline: {e}"))?;

        let (in_off, _in_len, _in_shape) = &d.inputs[0];
        let (out_off, out_len, _out_shape) = d.output;
        let n: u32 = *out_len;

        d.encoder.set_compute_pipeline_state(&pipe);
        // Operands are sub-ranges of the shared arena buffer — bind by byte offset.
        d.encoder.set_buffer(0, Some(d.arena), *in_off as u64);
        d.encoder.set_buffer(1, Some(d.arena), *out_off as u64);
        d.encoder
            .set_bytes(2, 4, &n as *const u32 as *const std::ffi::c_void);
        let tew = pipe.thread_execution_width().max(1).min(n.max(1) as u64);
        d.encoder.dispatch_threads(
            metal::MTLSize {
                width: n as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tew,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}

#[test]
fn times_three_runs_on_metal_via_raw_gpu_kernel() {
    register_op(Arc::new(TimesThreeIr));
    register_metal_gpu_kernel(Arc::new(TimesThreeMetal));

    let n = 8usize;
    let mut g = Graph::new("metal_gpu_custom");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let y = g.custom_op("test.times_three_gpu", vec![], vec![x]);
    g.set_outputs(vec![y]);

    // Device::Metal → the compile path lowers Op::Custom to Thunk::CustomGpuOp
    // (GPU registry beats the host registry) and the executor encodes it inline.
    let mut c = Session::new(Device::Metal).compile(g);
    let x0: Vec<f32> = (0..n).map(|i| i as f32 - 3.0).collect();
    let outs = c.run(&[("x", &x0)]);
    let out = &outs[0];

    let expect: Vec<f32> = x0.iter().map(|v| v * 3.0).collect();
    assert_eq!(out.len(), expect.len(), "out={out:?}");
    for (a, b) in out.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-4, "out={out:?} expect={expect:?}");
    }
}
