// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native Metal kernel for the **HC (hyper-connection) Sinkhorn gate**
//! (`Op::Custom("dsv4.hc_sinkhorn_gate")`). One GPU thread per row computes the
//! whole `[hc×hc]` Sinkhorn (sigmoid pre/post + softmax + iterative row/col
//! normalize) in registers — `hc ≤ 4` so the `[hc,hc]` matrix is ≤16 floats. This
//! replaces the ~36 tiny `Div`/`Reduce` launches the decomposed gate emits with a
//! SINGLE on-device dispatch (no host roundtrip, no queue sync) — the launch
//! overhead was ~63% of DSV4 decode ops. Raw-GPU seam (`MetalGpuKernel`), so the
//! executor encodes it inline on the active compute encoder.
//!
//! Registered by the consumer (rlx-models-core, under the `metal` feature) via
//! [`register`] — not in rlx-metal's builtin init, matching the `llada2_gate`
//! convention for model-driven ops.
//!
//! Inputs: `mixes [rows, 2hc+hc²]`, `scale [3]`, `base [2hc+hc²]`.
//! Attrs (LE): `[hc: u32, iters: u32, eps: f32]`.
//! Output: `[rows, 2hc+hc²]` packed `[pre(hc) | post(hc) | comb(hc²)]`.

use std::sync::Arc;

use crate::op_registry::{MetalGpuDispatch, MetalGpuKernel, register_metal_gpu_kernel};

pub const OP_NAME: &str = "dsv4.hc_sinkhorn_gate";

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void hc_sinkhorn_gate(
    device const float* mixes [[buffer(0)]],   // [rows, mh]
    device const float* scale [[buffer(1)]],   // [3]
    device const float* base  [[buffer(2)]],   // [mh]
    device float*       out   [[buffer(3)]],   // [rows, mh]
    constant uint&  rows  [[buffer(4)]],
    constant uint&  hc    [[buffer(5)]],
    constant uint&  iters [[buffer(6)]],
    constant float& eps   [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    uint mh = 2u * hc + hc * hc;
    device const float* m = mixes + gid * mh;
    device float* o = out + gid * mh;
    float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    // pre = sigmoid(m*s0 + base) + eps ; post = 2*sigmoid(m*s1 + base)
    for (uint i = 0; i < hc; i++) o[i] = 1.0f / (1.0f + exp(-(m[i] * s0 + base[i]))) + eps;
    for (uint i = 0; i < hc; i++) o[hc + i] = 2.0f / (1.0f + exp(-(m[hc + i] * s1 + base[hc + i])));
    // comb [hc,hc]: softmax over k (last) + eps
    float c[16];
    for (uint j = 0; j < hc; j++) {
        float mx = -1e30f;
        for (uint k = 0; k < hc; k++) { float l = m[2u*hc + j*hc + k] * s2 + base[2u*hc + j*hc + k]; c[j*hc+k] = l; mx = max(mx, l); }
        float sm = 0.0f;
        for (uint k = 0; k < hc; k++) { float e = exp(c[j*hc+k] - mx); c[j*hc+k] = e; sm += e; }
        for (uint k = 0; k < hc; k++) c[j*hc+k] = c[j*hc+k] / sm + eps;
    }
    // sinkhorn: first / (colsum_j + eps)
    for (uint k = 0; k < hc; k++) { float cs = eps; for (uint j = 0; j < hc; j++) cs += c[j*hc+k]; for (uint j = 0; j < hc; j++) c[j*hc+k] /= cs; }
    for (uint it = 1u; it < iters; it++) {
        for (uint j = 0; j < hc; j++) { float rs = eps; for (uint k = 0; k < hc; k++) rs += c[j*hc+k]; for (uint k = 0; k < hc; k++) c[j*hc+k] /= rs; }
        for (uint k = 0; k < hc; k++) { float cs = eps; for (uint j = 0; j < hc; j++) cs += c[j*hc+k]; for (uint j = 0; j < hc; j++) c[j*hc+k] /= cs; }
    }
    for (uint idx = 0; idx < hc * hc; idx++) o[2u*hc + idx] = c[idx];
}
"#;

#[derive(Debug)]
struct HcSinkhornGateMetal;

impl MetalGpuKernel for HcSinkhornGateMetal {
    fn name(&self) -> &str {
        OP_NAME
    }

    fn encode(&self, d: &MetalGpuDispatch) -> Result<(), String> {
        let a = d.attrs;
        if a.len() < 12 {
            return Err("hc_sinkhorn_gate: attrs must be [hc:u32, iters:u32, eps:f32]".into());
        }
        let rd = |i: usize| u32::from_le_bytes([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let hc = rd(0);
        let iters = rd(4);
        let eps = f32::from_le_bytes([a[8], a[9], a[10], a[11]]);
        let mh = 2 * hc + hc * hc;
        let (mixes_off, _, _) = &d.inputs[0];
        let (scale_off, _, _) = &d.inputs[1];
        let (base_off, _, _) = &d.inputs[2];
        let (out_off, out_len, _) = d.output;
        let rows: u32 = out_len / mh.max(1);

        let dev = crate::device::metal_device().ok_or("no metal device")?;
        let lib = crate::pipeline_cache::load_or_compile_library(&dev.device, SRC);
        let func = lib
            .get_function("hc_sinkhorn_gate", None)
            .map_err(|e| format!("get_function: {e}"))?;
        let pipe = dev
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("pipeline: {e}"))?;

        d.encoder.set_compute_pipeline_state(&pipe);
        d.encoder.set_buffer(0, Some(d.arena), *mixes_off as u64);
        d.encoder.set_buffer(1, Some(d.arena), *scale_off as u64);
        d.encoder.set_buffer(2, Some(d.arena), *base_off as u64);
        d.encoder.set_buffer(3, Some(d.arena), *out_off as u64);
        d.encoder
            .set_bytes(4, 4, &rows as *const u32 as *const std::ffi::c_void);
        d.encoder
            .set_bytes(5, 4, &hc as *const u32 as *const std::ffi::c_void);
        d.encoder
            .set_bytes(6, 4, &iters as *const u32 as *const std::ffi::c_void);
        d.encoder
            .set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
        let tew = pipe.thread_execution_width().max(1).min(rows.max(1) as u64);
        d.encoder.dispatch_threads(
            metal::MTLSize {
                width: rows as u64,
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

/// Register the native Metal Sinkhorn-gate kernel. Idempotent; call from the
/// consumer under the `metal` feature.
pub fn register() {
    register_metal_gpu_kernel(Arc::new(HcSinkhornGateMetal));
}
