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

//! Native GPU Elman RNN for ROCm arenas (`Step::Rnn`).
//!
//! Same coverage and semantics as [`rlx_cuda::rnn_gpu`] / Metal `rnn` /
//! wgpu `rnn.wgsl` (single-layer / unidir / no-carry / `hidden ≤ 1024`).

use crate::device::RocmContext;
use crate::hip::{HipBuffer, HipStream};
use std::sync::Arc;

/// Max hidden size for the native kernel (matches Metal `RNN_MAX_H`).
pub const RNN_MAX_H: usize = 1024;

/// True when the native `rnn` kernel can run this geometry.
#[inline]
pub fn native_rnn_ok(num_layers: usize, bidirectional: bool, carry: bool, hidden: usize) -> bool {
    num_layers == 1 && !bidirectional && !carry && hidden > 0 && hidden <= RNN_MAX_H
}

/// Launch native Elman RNN. All `*_byte` args are byte offsets into `buffer`.
#[allow(clippy::too_many_arguments)]
pub fn run_rnn(
    ctx: &Arc<RocmContext>,
    stream: HipStream,
    buffer: &HipBuffer<f32>,
    x_byte: usize,
    w_ih_byte: usize,
    w_hh_byte: usize,
    bias_byte: usize,
    dst_byte: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    relu: bool,
) {
    if batch == 0 || seq == 0 || hidden == 0 {
        return;
    }
    let kernel = crate::kernels::rnn_kernel(ctx);
    let mut arena_ptr = buffer.ptr;
    let x_off = (x_byte / 4) as u32;
    let wih_off = (w_ih_byte / 4) as u32;
    let whh_off = (w_hh_byte / 4) as u32;
    let bias_off = (bias_byte / 4) as u32;
    let dst_off = (dst_byte / 4) as u32;
    let batch_u = batch as u32;
    let seq_u = seq as u32;
    let in_u = input_size as u32;
    let hidden_u = hidden as u32;
    let relu_u = u32::from(relu);
    crate::launch_kernel!(
        kernel,
        stream,
        (batch_u, 1, 1),
        (hidden_u, 1, 1),
        [
            &mut arena_ptr,
            &x_off,
            &wih_off,
            &whh_off,
            &bias_off,
            &dst_off,
            &batch_u,
            &seq_u,
            &in_u,
            &hidden_u,
            &relu_u
        ]
    );
}
