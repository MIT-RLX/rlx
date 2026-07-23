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

//! Native GPU Mamba-2 SSD scan for ROCm arenas (`Step::Mamba2`).
//!
//! Same coverage and semantics as [`rlx_cuda::mamba2_gpu`] / Metal `mamba2` /
//! wgpu `mamba2.wgsl` (`state_size ≤ 256`; host path otherwise).

use crate::device::RocmContext;
use crate::hip::{HipBuffer, HipStream};
use std::sync::Arc;

/// Max state size for the native kernel (matches wgpu `MAX_STATE`).
pub const MAMBA2_MAX_N: usize = 256;

/// True when the native `mamba2` kernel can run this geometry.
#[inline]
pub fn native_mamba2_ok(state_size: usize) -> bool {
    state_size > 0 && state_size <= MAMBA2_MAX_N
}

/// Launch native Mamba-2. All `*_byte` args are byte offsets into `buffer`.
#[allow(clippy::too_many_arguments)]
pub fn run_mamba2(
    ctx: &Arc<RocmContext>,
    stream: HipStream,
    buffer: &HipBuffer<f32>,
    x_byte: usize,
    dt_byte: usize,
    a_byte: usize,
    b_byte: usize,
    c_byte: usize,
    dst_byte: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    state_size: usize,
) {
    if batch == 0 || seq == 0 || heads == 0 || head_dim == 0 || state_size == 0 {
        return;
    }
    let kernel = crate::kernels::mamba2_kernel(ctx);
    let total = (batch * heads * head_dim) as u32;
    let (grid, block) = crate::kernels::dispatch_grid_1d(total, 64);
    let mut arena_ptr = buffer.ptr;
    let x_off = (x_byte / 4) as u32;
    let dt_off = (dt_byte / 4) as u32;
    let a_off = (a_byte / 4) as u32;
    let b_off = (b_byte / 4) as u32;
    let c_off = (c_byte / 4) as u32;
    let dst_off = (dst_byte / 4) as u32;
    let batch_u = batch as u32;
    let seq_u = seq as u32;
    let heads_u = heads as u32;
    let p_u = head_dim as u32;
    let n_u = state_size as u32;
    crate::launch_kernel!(
        kernel,
        stream,
        (grid, 1, 1),
        (block, 1, 1),
        [
            &mut arena_ptr,
            &x_off,
            &dt_off,
            &a_off,
            &b_off,
            &c_off,
            &dst_off,
            &batch_u,
            &seq_u,
            &heads_u,
            &p_u,
            &n_u
        ]
    );
}
