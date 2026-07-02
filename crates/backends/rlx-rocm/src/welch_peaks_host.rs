// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

use crate::device::RocmContext;
use crate::hip::HipBuffer;

#[allow(clippy::too_many_arguments)]
pub fn run_welch_peaks(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    spec_byte_off: usize,
    dst_byte_off: usize,
    welch_batch: usize,
    n_fft: usize,
    n_segments: usize,
    k: usize,
    pre_sync: bool,
) {
    let spec_len = welch_batch * n_segments * n_fft * 2;
    let dst_len = welch_batch * k * 2;
    let span_off = spec_byte_off.min(dst_byte_off);
    let span_end = (spec_byte_off + spec_len * 4).max(dst_byte_off + dst_len * 4);
    let span_len = span_end - span_off;

    let rt = &ctx.runtime;
    if pre_sync {
        unsafe {
            let _ = (rt.hip_stream_sync)(ctx.default_stream);
        }
    }

    let mut host = vec![0u8; span_len];
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(
            host.as_mut_ptr() as *mut _,
            buffer.ptr + span_off as u64,
            span_len,
        );
        rlx_cpu::thunk::execute_welch_peaks_f32(
            spec_byte_off - span_off,
            dst_byte_off - span_off,
            welch_batch,
            n_fft,
            n_segments,
            k,
            host.as_mut_ptr(),
        );
        let _ = (rt.hip_memcpy_htod)(
            buffer.ptr + span_off as u64,
            host.as_ptr() as *const _,
            span_len,
        );
    }
}
