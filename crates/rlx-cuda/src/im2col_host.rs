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

#[allow(clippy::too_many_arguments)]
pub fn run_im2col(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    x_byte_off: usize,
    col_byte_off: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw_dil: u32,
) {
    let per_batch = (c_in as usize) * (h as usize) * (w as usize);
    let n_eff = if n == 0 { 0 } else { n as usize };
    let m = n_eff * h_out as usize * w_out as usize;
    let k = (c_in as usize) * (kh as usize) * (kw as usize);
    let x_len = if n == 0 {
        per_batch.max(1)
    } else {
        n_eff * per_batch
    };
    let col_len = if n == 0 { k.max(1) } else { m * k };
    let span_start = x_byte_off.min(col_byte_off);
    let span_end = (x_byte_off + x_len * 4).max(col_byte_off + col_len * 4);
    let span_len = span_end.saturating_sub(span_start);

    stream
        .synchronize()
        .expect("rlx-cuda: im2col pre-sync failed");

    let span_start_f32 = span_start / 4;
    let span_end_f32 = span_end.div_ceil(4);
    let mut host = vec![0u8; span_len];
    stream
        .memcpy_dtoh(
            &buffer.slice(span_start_f32..span_end_f32),
            bytemuck::cast_slice_mut(&mut host),
        )
        .expect("rlx-cuda: im2col partial dtoh failed");

    unsafe {
        rlx_cpu::im2col::execute_im2col_rows_layout(
            x_byte_off - span_start,
            col_byte_off - span_start,
            n,
            c_in,
            h,
            w,
            h_out,
            w_out,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw_dil,
            host.as_mut_ptr(),
        );
    }

    stream
        .memcpy_htod(
            bytemuck::cast_slice(&host),
            &mut buffer.slice_mut(span_start_f32..span_end_f32),
        )
        .expect("rlx-cuda: im2col partial htod failed");
}
