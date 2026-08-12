// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU decode for the `MxFp4x2` residual format, validated on Metal against the
//! CPU oracle (`rlx_ir::residual::residual_dequantize`). Proves the two-level
//! decode `out = s0·LUT[q0] + s1·LUT[q1]` runs correctly on a GPU — the
//! numeric core a `scaled_lowp` decode-GEMM would accumulate (no FP4 tensor
//! cores required; this is the dequantize path).

#![cfg(target_os = "macos")]

use rlx_ir::quant::ScaledFormat;
use rlx_ir::residual::{residual_dequantize, residual_quantize};
use rlx_metal::mtl::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::ffi::c_void;

/// Metal decode kernel — E2M1 LUT matches `rlx_ir::nvfp4::FP4_E2M1_LUT`.
const DECODE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
constant float E2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
kernel void mxfp4x2_decode(device const uint*  q0    [[buffer(0)]],
                           device const uint*  q1    [[buffer(1)]],
                           device const float* s0    [[buffer(2)]],
                           device const float* s1    [[buffer(3)]],
                           device float*       out   [[buffer(4)]],
                           constant uint&      n     [[buffer(5)]],
                           constant uint&      group [[buffer(6)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint shift = (i & 7u) * 4u;
    uint nib0 = (q0[i >> 3] >> shift) & 0xFu;
    uint nib1 = (q1[i >> 3] >> shift) & 0xFu;
    uint blk = i / group;
    out[i] = s0[blk] * E2M1[nib0] + s1[blk] * E2M1[nib1];
}
"#;

/// Pack 4-bit codes 8-per-u32 (low nibble first).
fn pack_nibbles(codes: &[u8]) -> Vec<u32> {
    let mut out = vec![0u32; codes.len().div_ceil(8)];
    for (i, &c) in codes.iter().enumerate() {
        out[i >> 3] |= ((c & 0xF) as u32) << ((i & 7) * 4);
    }
    out
}

fn buf_u32(d: &Device, v: &[u32]) -> rlx_metal::mtl::Buffer {
    d.new_buffer_with_data(
        v.as_ptr() as *const c_void,
        (v.len() * 4).max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}
fn buf_f32(d: &Device, v: &[f32]) -> rlx_metal::mtl::Buffer {
    d.new_buffer_with_data(
        v.as_ptr() as *const c_void,
        (v.len() * 4).max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[test]
fn mxfp4x2_gpu_decode_matches_cpu_oracle() {
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };
    let fmt = ScaledFormat::F4E2M1;
    let group = 32usize;
    let nblocks = 6usize;
    let n = group * nblocks;

    // Deterministic data; quantize each block to 2 residual levels.
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin() * 3.7).collect();
    let (mut q0c, mut q1c, mut s0, mut s1, mut cpu) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for b in 0..nblocks {
        let rb = residual_quantize(&data[b * group..(b + 1) * group], fmt, 2);
        s0.push(rb.scales[0]);
        s1.push(rb.scales[1]);
        q0c.extend_from_slice(&rb.codes[0]);
        q1c.extend_from_slice(&rb.codes[1]);
        cpu.extend(residual_dequantize(&rb));
    }

    // GPU decode.
    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(DECODE_MSL, &opts).unwrap();
    let f = lib.get_function("mxfp4x2_decode", None).unwrap();
    let pipe = device.new_compute_pipeline_state_with_function(&f).unwrap();

    let (p0, p1) = (
        buf_u32(&device, &pack_nibbles(&q0c)),
        buf_u32(&device, &pack_nibbles(&q1c)),
    );
    let (b0, b1) = (buf_f32(&device, &s0), buf_f32(&device, &s1));
    let out = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
    let (nn, gg) = (n as u32, group as u32);

    let q = device.new_command_queue();
    let cmd = q.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(0, Some(&p0), 0);
    enc.set_buffer(1, Some(&p1), 0);
    enc.set_buffer(2, Some(&b0), 0);
    enc.set_buffer(3, Some(&b1), 0);
    enc.set_buffer(4, Some(&out), 0);
    enc.set_bytes(5, 4, &nn as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &gg as *const u32 as *const c_void);
    let tg = 64u64;
    enc.dispatch_thread_groups(
        MTLSize::new((n as u64).div_ceil(tg), 1, 1),
        MTLSize::new(tg, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let p = out.contents() as *const f32;
    let gpu: Vec<f32> = (0..n).map(|i| unsafe { *p.add(i) }).collect();

    let worst = cpu
        .iter()
        .zip(&gpu)
        .map(|(&c, &g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    // Reference vs GPU decode use the same LUT + scales → bit-exact.
    let ref_err = {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&d, &g) in data.iter().zip(&gpu) {
            num += (d - g).powi(2) as f64;
            den += (d as f64).powi(2);
        }
        (num / den).sqrt()
    };
    eprintln!("MxFp4x2 GPU decode: worst |gpu-cpu|={worst:.2e}  rms vs f32={ref_err:.2e}");
    assert!(
        worst < 1e-6,
        "GPU decode must match CPU oracle, worst={worst:e}"
    );
    assert!(
        ref_err < 2e-2,
        "2-level decode should be ~1% of f32, got {ref_err:e}"
    );
}
