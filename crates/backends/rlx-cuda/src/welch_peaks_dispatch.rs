// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::kernels::{dispatch_grid_1d, welch_peaks_gpu_kernel};

/// Native GPU WelchPeaks on the device arena (f32 block-layout spectra).
pub fn run_welch_peaks_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    spec_off: u32,
    dst_off: u32,
    welch_batch: u32,
    n_fft: u32,
    n_segments: u32,
    k: u32,
    n_bins: u32,
) {
    let kernel = welch_peaks_gpu_kernel(ctx);
    let (grid, block) = dispatch_grid_1d(welch_batch, 64);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&spec_off)
        .arg(&dst_off)
        .arg(&welch_batch)
        .arg(&n_fft)
        .arg(&n_segments)
        .arg(&k)
        .arg(&n_bins);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: welch_peaks_gpu launch failed");
    }
}
