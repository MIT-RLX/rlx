// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native GPU GRU for ROCm arenas (`Step::Gru`).
//!
//! Same coverage and semantics as [`rlx_cuda::gru_gpu`] / Metal `gru` /
//! wgpu `gru.wgsl` (single-layer / unidir / no-carry / `hidden ≤ 1024`).

use crate::device::RocmContext;
use crate::hip::{HipBuffer, HipStream};
use std::sync::Arc;

/// Max hidden size for the native kernel (matches Metal `GRU_MAX_H`).
pub const GRU_MAX_H: usize = 1024;

/// True when the native `gru` kernel can run this geometry.
#[inline]
pub fn native_gru_ok(num_layers: usize, bidirectional: bool, carry: bool, hidden: usize) -> bool {
    num_layers == 1 && !bidirectional && !carry && hidden > 0 && hidden <= GRU_MAX_H
}

/// Launch native GRU. All `*_byte` args are byte offsets into `buffer`.
#[allow(clippy::too_many_arguments)]
pub fn run_gru(
    ctx: &Arc<RocmContext>,
    stream: HipStream,
    buffer: &HipBuffer<f32>,
    x_byte: usize,
    w_ih_byte: usize,
    w_hh_byte: usize,
    b_ih_byte: usize,
    b_hh_byte: usize,
    dst_byte: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
) {
    if batch == 0 || seq == 0 || hidden == 0 {
        return;
    }
    let kernel = crate::kernels::gru_kernel(ctx);
    let mut arena_ptr = buffer.ptr;
    let x_off = (x_byte / 4) as u32;
    let wih_off = (w_ih_byte / 4) as u32;
    let whh_off = (w_hh_byte / 4) as u32;
    let bih_off = (b_ih_byte / 4) as u32;
    let bhh_off = (b_hh_byte / 4) as u32;
    let dst_off = (dst_byte / 4) as u32;
    let batch_u = batch as u32;
    let seq_u = seq as u32;
    let in_u = input_size as u32;
    let hidden_u = hidden as u32;
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
            &bih_off,
            &bhh_off,
            &dst_off,
            &batch_u,
            &seq_u,
            &in_u,
            &hidden_u
        ]
    );
}
