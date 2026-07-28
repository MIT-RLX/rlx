// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Double-single (2× f32 ≈ f64) reductions on Metal**, which has no native
//! `f64`. The arithmetic library comes straight from [`rlxsl::dw`]; the parallel
//! threadgroup sum below accumulates in the `(hi, lo)` pair via error-free
//! transforms, recovering ~49 significand bits.
//!
//! The library is compiled with **precise math** (`set_fast_math_enabled(false)`)
//! — unlike the main `RLX_KERNELS_MSL` library ([`crate::pipeline_cache`], which
//! uses default options). This is mandatory: Metal defaults to fast-math, whose
//! algebraic reassociation collapses `dw_two_sum`'s error term to zero.

use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::ffi::c_void;

const THREADS: u64 = 256;

/// Opt-in: route full `Op::Reduce{Sum}` (to a scalar, f32) through the
/// double-single accumulation kernel instead of the plain f32 reduce — trades
/// ~3–4× flops for a correctly-rounded (near-f64) result. Set `RLX_METAL_DW_SUM=1`.
pub fn dw_sum_reduce_enabled() -> bool {
    rlx_ir::env::flag("RLX_METAL_DW_SUM")
}

const DW_REDUCE_MSL: &str = r#"
kernel void dw_sum(device const float* x [[buffer(0)]],
                   device float* out     [[buffer(1)]],
                   constant uint& n      [[buffer(2)]],
                   uint tid       [[thread_position_in_threadgroup]],
                   uint nthreads  [[threads_per_threadgroup]]) {
    threadgroup DwF32 shared[256];
    DwF32 acc = DwF32{0.0f, 0.0f};
    for (uint i = tid; i < n; i += nthreads) { acc = dw_add(acc, DwF32{x[i], 0.0f}); }
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0u; s >>= 1) {
        if (tid < s) { shared[tid] = dw_add(shared[tid], shared[tid + s]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { out[0] = shared[0].hi; out[1] = shared[0].lo; }
}
"#;

fn source() -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\n{DW_REDUCE_MSL}",
        rlxsl::dw::double_single_prelude(rlxsl::Lang::Msl)
    )
}

/// A compiled double-single sum reducer for one Metal device. Build once (it
/// compiles the precise-math library and pipeline), then reuse across calls.
pub struct DwReduce {
    device: Device,
    pipeline: ComputePipelineState,
}

impl DwReduce {
    /// Compile the double-single reduction with **precise math**. Returns an
    /// error string if MSL compilation / pipeline creation fails.
    pub fn new(device: &Device) -> Result<Self, String> {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device.new_library_with_source(&source(), &opts)?;
        let func = lib.get_function("dw_sum", None)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| e.to_string())?;
        Ok(DwReduce {
            device: device.clone(),
            pipeline,
        })
    }

    /// Sum `data` in double-single precision; returns the reconstructed `f64`.
    /// Order-independent to ~49 bits, so the parallel tree reduction is fine.
    pub fn sum(&self, data: &[f32]) -> f64 {
        let (hi, lo) = self.sum_pair(data);
        hi as f64 + lo as f64
    }

    /// The raw `(hi, lo)` double-single result (for feeding further dw math).
    pub fn sum_pair(&self, data: &[f32]) -> (f32, f32) {
        let xbuf = self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            (data.len() * 4).max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let obuf = self
            .device
            .new_buffer(8, MTLResourceOptions::StorageModeShared);
        let n = data.len() as u32;
        let queue = self.device.new_command_queue();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&xbuf), 0);
        enc.set_buffer(1, Some(&obuf), 0);
        enc.set_bytes(2, 4, &n as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(THREADS, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let p = obuf.contents() as *const f32;
        unsafe { (*p, *p.add(1)) }
    }
}
