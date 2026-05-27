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
//! GPU Adam step for packed Gaussian splat parameters.

#![cfg(all(feature = "native-splat", target_os = "macos"))]

use crate::device::metal_device;
use crate::kernels::kernels;
use metal::{Buffer, ComputePipelineState, MTLResourceOptions};

#[repr(C)]
#[derive(Clone, Copy)]
struct AdamParamGpu {
    lr: f32,
    grad_clip: f32,
    vmin: f32,
    vmax: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AdamHyperGpu {
    param_count: u32,
    splat_count: u32,
    beta1: f32,
    beta2: f32,
    bias_corr1: f32,
    bias_corr2: f32,
    adam_eps: f32,
    huge_value: f32,
    max_update: f32,
}

/// One Adam step on Metal; reads/writes `params` and `moments` in place.
pub fn adam_step_metal(
    pipeline: &ComputePipelineState,
    params: &mut [f32],
    grads: &[f32],
    moments: &mut [[f32; 2]],
    splat_count: u32,
    beta1: f32,
    beta2: f32,
    step_index: u32,
    adam_eps: f32,
    huge_value: f32,
    max_update: f32,
    settings: &[AdamParamSetting],
) {
    assert_eq!(params.len(), grads.len());
    assert_eq!(params.len(), moments.len());
    let dev = metal_device().expect("Metal device");
    let n = params.len();
    let param_count = n as u32;
    let splat_count = splat_count.max(1);
    let param_types = (param_count / splat_count).max(1) as usize;

    let params_buf = dev.device.new_buffer_with_data(
        params.as_ptr() as *const _,
        (n * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let grads_buf = dev.device.new_buffer_with_data(
        grads.as_ptr() as *const _,
        (n * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut moments_flat = vec![0.0f32; n * 2];
    for (i, m) in moments.iter().enumerate() {
        moments_flat[i * 2] = m[0];
        moments_flat[i * 2 + 1] = m[1];
    }
    let moments_buf = dev.device.new_buffer_with_data(
        moments_flat.as_ptr() as *const _,
        (n * 2 * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut settings_gpu = vec![
        AdamParamGpu {
            lr: 0.0,
            grad_clip: 10.0,
            vmin: -1e8,
            vmax: 1e8,
        };
        param_types
    ];
    for s in settings {
        let idx = s.param_id as usize;
        if idx < settings_gpu.len() {
            settings_gpu[idx] = AdamParamGpu {
                lr: s.lr,
                grad_clip: s.grad_clip_abs,
                vmin: s.value_min,
                vmax: s.value_max,
            };
        }
    }
    let settings_buf = dev.device.new_buffer_with_data(
        settings_gpu.as_ptr() as *const _,
        (settings_gpu.len() * std::mem::size_of::<AdamParamGpu>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let beta1_pow = beta1.powi(step_index as i32);
    let beta2_pow = beta2.powi(step_index as i32);
    let hyper = AdamHyperGpu {
        param_count,
        splat_count,
        beta1,
        beta2,
        bias_corr1: (1.0 - beta1_pow).max(adam_eps),
        bias_corr2: (1.0 - beta2_pow).max(adam_eps),
        adam_eps,
        huge_value,
        max_update,
    };
    let hyper_buf = dev.device.new_buffer_with_data(
        &hyper as *const _ as *const std::ffi::c_void,
        std::mem::size_of::<AdamHyperGpu>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = dev.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&params_buf), 0);
    enc.set_buffer(1, Some(&grads_buf), 0);
    enc.set_buffer(2, Some(&moments_buf), 0);
    enc.set_buffer(3, Some(&settings_buf), 0);
    enc.set_buffer(4, Some(&hyper_buf), 0);
    let tg = pipeline.thread_execution_width().max(1);
    let groups = (n as u64 + tg as u64 - 1) / tg as u64;
    enc.dispatch_thread_groups(
        metal::MTLSize::new(groups, 1, 1),
        metal::MTLSize::new(tg, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    unsafe {
        std::ptr::copy_nonoverlapping(params_buf.contents() as *const f32, params.as_mut_ptr(), n);
        let mom_ptr = moments_buf.contents() as *const f32;
        for (i, m) in moments.iter_mut().enumerate() {
            m[0] = *mom_ptr.add(i * 2);
            m[1] = *mom_ptr.add(i * 2 + 1);
        }
    }
}

/// Per packed param type (position x, scale x, …).
#[derive(Clone, Copy)]
pub struct AdamParamSetting {
    pub param_id: u32,
    pub lr: f32,
    pub grad_clip_abs: f32,
    pub value_min: f32,
    pub value_max: f32,
}

/// Build GPU Adam constant buffers from host settings.
pub fn build_adam_gpu_buffers(args: &FusedAdamHostArgs) -> (Buffer, Buffer) {
    let dev = crate::device::metal_device().expect("Metal device");
    let param_types = args
        .settings
        .iter()
        .map(|s| s.param_id)
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(1);
    let mut settings_gpu = vec![
        AdamParamGpu {
            lr: 0.0,
            grad_clip: 10.0,
            vmin: -1e8,
            vmax: 1e8,
        };
        param_types
    ];
    for s in &args.settings {
        let idx = s.param_id as usize;
        if idx < settings_gpu.len() {
            settings_gpu[idx] = AdamParamGpu {
                lr: s.lr,
                grad_clip: s.grad_clip_abs,
                vmin: s.value_min,
                vmax: s.value_max,
            };
        }
    }
    let settings_buf = dev.device.new_buffer_with_data(
        settings_gpu.as_ptr() as *const _,
        (settings_gpu.len() * std::mem::size_of::<AdamParamGpu>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let beta1_pow = args.beta1.powi(args.step_index as i32);
    let beta2_pow = args.beta2.powi(args.step_index as i32);
    let hyper = AdamHyperGpu {
        param_count: args.packed_param_count,
        splat_count: args.splat_count,
        beta1: args.beta1,
        beta2: args.beta2,
        bias_corr1: (1.0 - beta1_pow).max(args.adam_eps),
        bias_corr2: (1.0 - beta2_pow).max(args.adam_eps),
        adam_eps: args.adam_eps,
        huge_value: args.huge_value,
        max_update: args.max_update,
    };
    let hyper_buf = dev.device.new_buffer_with_data(
        &hyper as *const _ as *const std::ffi::c_void,
        std::mem::size_of::<AdamHyperGpu>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    (settings_buf, hyper_buf)
}

/// Host-side Adam config; GPU buffers are created inside the fused encoder.
pub struct FusedAdamHostArgs {
    pub settings: Vec<AdamParamSetting>,
    pub beta1: f32,
    pub beta2: f32,
    pub step_index: u32,
    pub adam_eps: f32,
    pub huge_value: f32,
    pub max_update: f32,
    pub splat_count: u32,
    pub packed_param_count: u32,
}

/// Adam inside an existing command encoder (fused training).
#[allow(clippy::too_many_arguments)]
pub fn adam_encode_step(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    params_buf: &Buffer,
    grads_buf: &Buffer,
    moments_buf: &Buffer,
    settings_buf: &Buffer,
    hyper_buf: &Buffer,
    param_count: usize,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(params_buf), 0);
    encoder.set_buffer(1, Some(grads_buf), 0);
    encoder.set_buffer(2, Some(moments_buf), 0);
    encoder.set_buffer(3, Some(settings_buf), 0);
    encoder.set_buffer(4, Some(hyper_buf), 0);
    let tg = pipeline.thread_execution_width().max(1);
    let groups = (param_count as u64 + tg as u64 - 1) / tg as u64;
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(groups, 1, 1),
        metal::MTLSize::new(tg, 1, 1),
    );
}

pub fn adam_step_metal_packed(
    params: &mut [f32],
    grads: &[f32],
    moments: &mut [[f32; 2]],
    splat_count: u32,
    beta1: f32,
    beta2: f32,
    step_index: u32,
    adam_eps: f32,
    huge_value: f32,
    max_update: f32,
    settings: &[AdamParamSetting],
) {
    let k = kernels();
    adam_step_metal(
        &k.gaussian_splat_adam_step,
        params,
        grads,
        moments,
        splat_count,
        beta1,
        beta2,
        step_index,
        adam_eps,
        huge_value,
        max_update,
        settings,
    );
}
