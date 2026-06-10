// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

use crate::buffer::Arena;

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
    let spec_len = outer * n_fft * 2;
    let filt_len = n_mels * n_bins;
    let dst_len = outer * n_mels;
    let span_off = spec_byte_off.min(filt_byte_off).min(dst_byte_off);
    let span_end = (spec_byte_off + spec_len * 4)
        .max(filt_byte_off + filt_len * 4)
        .max(dst_byte_off + dst_len * 4);
    let span_len = span_end - span_off;

    let mut host = arena.read_bytes_range(device, queue, span_off, span_len);
    unsafe {
        rlx_cpu::thunk::execute_log_mel_f32(
            spec_byte_off - span_off,
            filt_byte_off - span_off,
            dst_byte_off - span_off,
            outer,
            n_fft,
            n_bins,
            n_mels,
            host.as_mut_ptr(),
        );
    }
    arena.write_bytes_range(queue, span_off, &host);
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
    let spec_len = outer * n_fft * 2;
    let filt_len = n_mels * n_bins;
    let dy_len = outer * n_mels;
    let dst_len = outer * n_fft * 2;
    let span_off = spec_byte_off
        .min(filt_byte_off)
        .min(dy_byte_off)
        .min(dst_byte_off);
    let span_end = (spec_byte_off + spec_len * 4)
        .max(filt_byte_off + filt_len * 4)
        .max(dy_byte_off + dy_len * 4)
        .max(dst_byte_off + dst_len * 4);
    let span_len = span_end - span_off;

    let mut host = arena.read_bytes_range(device, queue, span_off, span_len);
    unsafe {
        rlx_cpu::thunk::execute_log_mel_backward_f32(
            spec_byte_off - span_off,
            filt_byte_off - span_off,
            dy_byte_off - span_off,
            dst_byte_off - span_off,
            outer,
            n_fft,
            n_bins,
            n_mels,
            host.as_mut_ptr(),
        );
    }
    arena.write_bytes_range(queue, span_off, &host);
}
