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

//! Host-side RNG fill for device arenas (fill on host → H2D).

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub fn run_rng_normal(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
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
    let off_f32 = dst_byte_off / 4;
    assert_eq!(
        dst_byte_off % 4,
        0,
        "rng_host: dst_byte_off must be f32-aligned"
    );
    let mut host = vec![0f32; len];
    rlx_ir::fill_normal_like(&mut host, mean, scale, opts, key, op_seed);
    stream
        .memcpy_htod(
            host.as_slice(),
            &mut buffer.slice_mut(off_f32..off_f32 + len),
        )
        .expect("rlx-cuda: rng htod failed");
}

pub fn run_rng_uniform(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
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
    let off_f32 = dst_byte_off / 4;
    assert_eq!(
        dst_byte_off % 4,
        0,
        "rng_host: dst_byte_off must be f32-aligned"
    );
    let mut host = vec![0f32; len];
    rlx_ir::fill_uniform_like(&mut host, low, high, opts, key, op_seed);
    stream
        .memcpy_htod(
            host.as_slice(),
            &mut buffer.slice_mut(off_f32..off_f32 + len),
        )
        .expect("rlx-cuda: rng htod failed");
}
