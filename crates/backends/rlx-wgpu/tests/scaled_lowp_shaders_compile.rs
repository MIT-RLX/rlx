// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `scaled_lowp.wgsl` entry points must compile and validate.
//!
//! naga rejects a shader at `create_shader_module`, i.e. at first *use* on a
//! device — so a WGSL mistake in a rarely-taken lowering arm would otherwise
//! surface as a panic in whatever graph first hits it, long after the edit.
//! Forcing every entry point through the pipeline builder here turns that into
//! an immediate, local failure.

#![cfg(feature = "splat")]

use rlx_wgpu::kernels;

fn device() -> Option<wgpu::Device> {
    let inst = wgpu::Instance::default();
    let adapter =
        pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(device)
}

#[test]
fn every_scaled_lowp_entry_point_builds() {
    // `rlx_ir::env::skip_unless_device` rather than a bare early return: the
    // bare form reports `ok` on a rig with no adapter, so a box that lost its
    // GPU would look green. This honours `RLX_REQUIRE_DEVICE=1` instead.
    let dev = device();
    if rlx_ir::env::skip_unless_device("wgpu", true, dev.is_some()) {
        return;
    }
    let device = dev.expect("availability checked above");
    // Each builder creates the shader module and the compute pipeline, so naga
    // parse + validate + backend codegen all run here.
    let _ = kernels::scaled_quant_scale_kernel(&device);
    let _ = kernels::scaled_quantize_kernel(&device);
    let _ = kernels::scaled_dequantize_kernel(&device);
    let _ = kernels::scaled_matmul_decode_kernel(&device);
    // Same reasoning for the INT8 QAT kernels.
    let _ = kernels::quantize_i8_kernel(&device);
    let _ = kernels::dequantize_i8_kernel(&device);
    let _ = kernels::q_matmul_kernel(&device);
    let _ = kernels::q_conv2d_kernel(&device);
    let _ = kernels::batch_norm_inference_kernel(&device);
    let _ = kernels::batch_norm_inference_bwd_input_kernel(&device);
    let _ = kernels::batch_norm_inference_bwd_gamma_kernel(&device);
    let _ = kernels::batch_norm_inference_bwd_beta_kernel(&device);
}
