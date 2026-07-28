// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::buffer::Arena;
use crate::kernels::{WelchPeaksGpuParams, welch_peaks_gpu_kernel};

fn dispatch_dims(n: u32, wg: u32) -> (u32, u32, u32) {
    (n.div_ceil(wg).max(1), 1, 1)
}

/// Dispatch native GPU WelchPeaks inside an existing compute pass.
pub fn dispatch_welch_peaks_gpu_in_pass(
    pass: &mut wgpu::ComputePass<'_>,
    device: &wgpu::Device,
    uniform: &wgpu::Buffer,
    bind_group: &wgpu::BindGroup,
    welch_batch: u32,
) {
    let k = welch_peaks_gpu_kernel(device);
    pass.set_pipeline(&k.pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    let _ = uniform;
    let (gx, gy, gz) = dispatch_dims(welch_batch, 64);
    pass.dispatch_workgroups(gx, gy, gz);
}

/// Standalone WelchPeaks GPU dispatch (submit + poll).
pub fn run_welch_peaks_gpu(
    _arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    uniform: &wgpu::Buffer,
    bind_group: &wgpu::BindGroup,
    welch_batch: u32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu welch_peaks_gpu"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("welch_peaks_gpu"),
            timestamp_writes: None,
        });
        dispatch_welch_peaks_gpu_in_pass(&mut pass, device, uniform, bind_group, welch_batch);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

pub fn welch_peaks_gpu_params(
    spec_off: u32,
    dst_off: u32,
    welch_batch: u32,
    n_fft: u32,
    n_segments: u32,
    k: u32,
    n_bins: u32,
) -> WelchPeaksGpuParams {
    WelchPeaksGpuParams {
        spec_off,
        dst_off,
        welch_batch,
        n_fft,
        n_segments,
        k,
        n_bins,
        _p0: 0,
        _p1: 0,
    }
}
