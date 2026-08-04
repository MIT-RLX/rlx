// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native GPU Elman RNN for ROCm arenas (`Step::Rnn`).
//!
//! Multi-layer / bidirectional / carry, `hidden ≤ 1024` — mirror of
//! [`rlx_cuda::rnn_gpu`]: a per-(layer, direction) kernel launch with the
//! recurrence inside; intermediate layer outputs ping-pong in a scratch buffer.
//! `relu` selects ReLU vs tanh; single merged bias. Bit-for-bit mirror of
//! `execute_rnn_f32` (same packed weight layout).

use crate::device::RocmContext;
use crate::hip::{HipBuffer, HipStream};
use std::sync::{Arc, Mutex, OnceLock};

/// Max hidden size for the native kernel (matches Metal `RNN_MAX_H`).
pub const RNN_MAX_H: usize = 1024;

/// True when the native `rnn` kernel can run this geometry (any layers/dirs/carry).
#[inline]
pub fn native_rnn_ok(
    _num_layers: usize,
    _bidirectional: bool,
    _carry: bool,
    hidden: usize,
) -> bool {
    hidden > 0 && hidden <= RNN_MAX_H
}

fn scratch_pool() -> &'static Mutex<Option<(usize, HipBuffer<f32>)>> {
    static P: OnceLock<Mutex<Option<(usize, HipBuffer<f32>)>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(None))
}

/// Launch native Elman RNN. All `*_byte` args are byte offsets into `buffer`.
/// `h0_byte == 0` (or `!carry`) means no carry.
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
    num_layers: usize,
    bidirectional: bool,
    h0_byte: usize,
    relu: bool,
) {
    if batch == 0 || seq == 0 || hidden == 0 || num_layers == 0 {
        return;
    }
    let kernel = crate::kernels::rnn_kernel(ctx);
    let dirs = if bidirectional { 2 } else { 1 };
    let out_width = dirs * hidden;
    let layer_elems = batch * seq * out_width;

    let scratch_len = if num_layers > 1 { 2 * layer_elems } else { 1 };
    let mut pool = scratch_pool().lock().unwrap();
    let grow = match pool.as_ref() {
        Some((cap, _)) => *cap < scratch_len,
        None => true,
    };
    if grow {
        let buf = HipBuffer::<f32>::alloc_zeros(&ctx.runtime, scratch_len.max(1))
            .expect("rlx-rocm: rnn scratch alloc failed");
        *pool = Some((scratch_len, buf));
    }
    let mut scratch_ptr = pool.as_ref().unwrap().1.ptr;
    let mut arena_ptr = buffer.ptr;
    let relu_u = u32::from(relu);

    let mut in_l = input_size;
    let mut wih_cursor = 0usize;
    let mut in_off_f = x_byte / 4;
    let mut in_is_scratch = 0u32;

    for l in 0..num_layers {
        let last = l + 1 == num_layers;
        let (out_off_f, out_is_scratch) = if last {
            (dst_byte / 4, 0u32)
        } else {
            ((l % 2) * layer_elems, 1u32)
        };
        let wih_block = hidden * in_l;

        for dir in 0..dirs {
            let ld = l * dirs + dir;
            let wih_off = (w_ih_byte / 4 + wih_cursor + dir * wih_block) as u32;
            let whh_off = (w_hh_byte / 4 + ld * hidden * hidden) as u32;
            let bias_off = (bias_byte / 4 + ld * hidden) as u32;
            let h0_off = if h0_byte != 0 {
                (h0_byte / 4 + ld * batch * hidden) as u32
            } else {
                0u32
            };
            let x_off_u = in_off_f as u32;
            let x_is = in_is_scratch;
            let dst_off_u = out_off_f as u32;
            let dst_is = out_is_scratch;
            let batch_u = batch as u32;
            let seq_u = seq as u32;
            let in_l_u = in_l as u32;
            let hidden_u = hidden as u32;
            let out_width_u = out_width as u32;
            let dir_off_u = (dir * hidden) as u32;
            let reverse_u = u32::from(dir == 1);

            crate::launch_kernel!(
                kernel,
                stream,
                (batch_u, 1, 1),
                (hidden_u, 1, 1),
                [
                    &mut arena_ptr,
                    &mut scratch_ptr,
                    &x_off_u,
                    &x_is,
                    &wih_off,
                    &whh_off,
                    &bias_off,
                    &dst_off_u,
                    &dst_is,
                    &batch_u,
                    &seq_u,
                    &in_l_u,
                    &hidden_u,
                    &h0_off,
                    &out_width_u,
                    &dir_off_u,
                    &reverse_u,
                    &relu_u
                ]
            );
        }

        wih_cursor += dirs * wih_block;
        in_l = out_width;
        in_off_f = (l % 2) * layer_elems;
        in_is_scratch = 1;
    }
}
