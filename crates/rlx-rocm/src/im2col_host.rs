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

use crate::device::RocmContext;
use crate::hip::HipBuffer;

#[allow(clippy::too_many_arguments)]
pub fn run_im2col(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
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

    let rt = &ctx.runtime;
    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
    }

    let mut host = vec![0u8; span_len];
    unsafe {
        let _ = (rt.hip_memcpy_dtoh)(
            host.as_mut_ptr() as *mut _,
            buffer.ptr + span_start as u64,
            span_len,
        );
    }

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

    unsafe {
        let _ = (rt.hip_memcpy_htod)(
            buffer.ptr + span_start as u64,
            host.as_ptr() as *const _,
            span_len,
        );
    }
}
