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

//! Host-side RNG fill for device arenas (D2H → fill → H2D).

use crate::device::RocmContext;
use crate::hip::HipBuffer;

pub fn run_rng_normal(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    dst_byte_off: usize,
    len: usize,
    mean: f32,
    scale: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    if len == 0 {
        return;
    }
    assert_eq!(
        dst_byte_off % 4,
        0,
        "rng_host: dst_byte_off must be f32-aligned"
    );
    let byte_len = len * 4;
    let rt = &ctx.runtime;
    let mut host = vec![0f32; len];
    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let src_ptr = (buffer.ptr as usize + dst_byte_off) as crate::hip::HipDeviceptr;
        let _ = (rt.hip_memcpy_dtoh)(host.as_mut_ptr() as *mut _, src_ptr, byte_len);
    }
    rlx_ir::fill_normal_like(&mut host, mean, scale, opts, key, op_seed);
    unsafe {
        let dst_ptr = (buffer.ptr as usize + dst_byte_off) as crate::hip::HipDeviceptr;
        let _ = (rt.hip_memcpy_htod)(dst_ptr, host.as_ptr() as *const _, byte_len);
    }
}

pub fn run_rng_uniform(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    dst_byte_off: usize,
    len: usize,
    low: f32,
    high: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    if len == 0 {
        return;
    }
    assert_eq!(
        dst_byte_off % 4,
        0,
        "rng_host: dst_byte_off must be f32-aligned"
    );
    let byte_len = len * 4;
    let rt = &ctx.runtime;
    let mut host = vec![0f32; len];
    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let src_ptr = (buffer.ptr as usize + dst_byte_off) as crate::hip::HipDeviceptr;
        let _ = (rt.hip_memcpy_dtoh)(host.as_mut_ptr() as *mut _, src_ptr, byte_len);
    }
    rlx_ir::fill_uniform_like(&mut host, low, high, opts, key, op_seed);
    unsafe {
        let dst_ptr = (buffer.ptr as usize + dst_byte_off) as crate::hip::HipDeviceptr;
        let _ = (rt.hip_memcpy_htod)(dst_ptr, host.as_ptr() as *const _, byte_len);
    }
}
