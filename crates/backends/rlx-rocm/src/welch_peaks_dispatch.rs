// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::device::RocmContext;
use crate::hip::HipDeviceptr;
use crate::hip::HipStream;
use crate::kernels::{dispatch_grid_1d, welch_peaks_gpu_kernel};

/// Native GPU WelchPeaks on the device arena (f32 block-layout spectra).
pub fn run_welch_peaks_gpu(
    ctx: &Arc<RocmContext>,
    stream: HipStream,
    arena_ptr: HipDeviceptr,
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
    let mut arena_ptr_mut = arena_ptr;
    crate::launch_kernel!(
        kernel,
        stream,
        (grid, 1, 1),
        (block, 1, 1),
        [
            &mut arena_ptr_mut,
            &spec_off,
            &dst_off,
            &welch_batch,
            &n_fft,
            &n_segments,
            &k,
            &n_bins
        ]
    );
}
