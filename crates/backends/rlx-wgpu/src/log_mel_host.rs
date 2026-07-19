// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Host-side `Op::LogMel` / backward for wgpu arenas.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;

#[allow(clippy::too_many_arguments)]
pub fn run_log_mel(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    spec_byte_off: usize,
    filt_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_log_mel(
        &mut a,
        spec_byte_off,
        filt_byte_off,
        dst_byte_off,
        outer,
        n_fft,
        n_bins,
        n_mels,
        /* pre_sync */ false,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_log_mel_backward(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    spec_byte_off: usize,
    filt_byte_off: usize,
    dy_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    rlx_gpu_host::run_log_mel_backward(
        &mut a,
        spec_byte_off,
        filt_byte_off,
        dy_byte_off,
        dst_byte_off,
        outer,
        n_fft,
        n_bins,
        n_mels,
        /* pre_sync */ false,
    );
}
