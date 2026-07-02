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

// RLX — versatile ML compiler + runtime.

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub fn run_log_mel(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    spec_byte_off: usize,
    filt_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    pre_sync: bool,
) {
    let spec_len = outer * n_fft * 2;
    let filt_len = n_mels * n_bins;
    let dst_len = outer * n_mels;
    let span_off = spec_byte_off.min(filt_byte_off).min(dst_byte_off);
    let span_end = (spec_byte_off + spec_len * 4)
        .max(filt_byte_off + filt_len * 4)
        .max(dst_byte_off + dst_len * 4);
    let span_len = span_end - span_off;
    assert_eq!(
        span_off % 4,
        0,
        "log_mel_host: span_off must be f32-aligned"
    );
    assert_eq!(
        span_len % 4,
        0,
        "log_mel_host: span_len must be f32-aligned"
    );
    let span_f32 = span_off / 4;
    let span_n_f32 = span_len / 4;

    if pre_sync {
        stream
            .synchronize()
            .expect("rlx-cuda: log_mel pre-sync failed");
    }

    let mut host = vec![0u8; span_len];
    stream
        .memcpy_dtoh(
            &buffer.slice(span_f32..span_f32 + span_n_f32),
            bytemuck::cast_slice_mut(&mut host),
        )
        .expect("rlx-cuda: log_mel partial dtoh failed");

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

    stream
        .memcpy_htod(
            bytemuck::cast_slice(&host),
            &mut buffer.slice_mut(span_f32..span_f32 + span_n_f32),
        )
        .expect("rlx-cuda: log_mel partial htod failed");
}
