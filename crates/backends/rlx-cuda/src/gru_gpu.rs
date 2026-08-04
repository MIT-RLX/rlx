// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native GPU GRU for CUDA arenas (`Step::Gru`).
//!
//! Multi-layer / bidirectional / carry, `hidden ≤ 1024` — the recurrence loops
//! inside a per-(layer, direction) kernel launch (no host round-trip). Gate
//! order r, z, n; separate b_ih/b_hh; `linear_before_reset=1`. Bit-for-bit
//! mirror of `execute_gru_f32` (same packed weight layout: `w_ih` cursor
//! advances `dirs·3h·in_l` per layer; `w_hh`/`b_ih`/`b_hh`/`h0` keyed by
//! `ld = l·dirs+dir`; output slice `[batch, seq, dirs·hidden]`).

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::{Arc, Mutex, OnceLock};

/// Max hidden size for the native kernel (matches Metal `GRU_MAX_H`).
pub const GRU_MAX_H: usize = 1024;

/// True when the native `gru` kernel can run this geometry (any layers/dirs/carry).
#[inline]
pub fn native_gru_ok(
    _num_layers: usize,
    _bidirectional: bool,
    _carry: bool,
    hidden: usize,
) -> bool {
    hidden > 0 && hidden <= GRU_MAX_H
}

/// Ping-pong scratch for the intermediate layer outputs (multi-layer only).
fn scratch_pool() -> &'static Mutex<Option<(usize, CudaSlice<f32>)>> {
    static P: OnceLock<Mutex<Option<(usize, CudaSlice<f32>)>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(None))
}

fn ensure_scratch<'a>(
    stream: &Arc<CudaStream>,
    need: usize,
    pool: &'a mut Option<(usize, CudaSlice<f32>)>,
) -> &'a mut CudaSlice<f32> {
    let need = need.max(1);
    let grow = match pool.as_ref() {
        Some((cap, _)) => *cap < need,
        None => true,
    };
    if grow {
        let buf = stream
            .alloc_zeros::<f32>(need)
            .expect("rlx-cuda: gru scratch alloc failed");
        *pool = Some((need, buf));
    }
    &mut pool.as_mut().unwrap().1
}

/// Launch native GRU. `*_byte` are byte offsets into `arena`. `h0_byte == 0`
/// (or `!carry`) means no carry.
#[allow(clippy::too_many_arguments)]
pub fn run_gru(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
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
    num_layers: usize,
    bidirectional: bool,
    h0_byte: usize,
) {
    if batch == 0 || seq == 0 || hidden == 0 || num_layers == 0 {
        return;
    }
    let kernel = crate::kernels::gru_kernel(ctx);
    let dirs = if bidirectional { 2 } else { 1 };
    let three_h = 3 * hidden;
    let out_width = dirs * hidden;
    let layer_elems = batch * seq * out_width;

    // Ping-pong scratch (word offsets 0 / layer_elems) for the intermediate
    // layer outputs; single-layer needs none.
    let scratch_len = if num_layers > 1 { 2 * layer_elems } else { 0 };
    let mut pool = scratch_pool().lock().unwrap();
    let scratch = ensure_scratch(stream, scratch_len, &mut pool);

    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (hidden as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut in_l = input_size;
    let mut wih_cursor = 0usize; // element cursor into w_ih
    // input source for layer 0: arena @ x_byte; later layers: scratch buffer.
    let mut in_off_f = x_byte / 4;
    let mut in_is_scratch = 0u32;

    for l in 0..num_layers {
        let last = l + 1 == num_layers;
        let (out_off_f, out_is_scratch) = if last {
            (dst_byte / 4, 0u32)
        } else {
            ((l % 2) * layer_elems, 1u32)
        };
        let wih_block = three_h * in_l;

        for dir in 0..dirs {
            let ld = l * dirs + dir;
            let wih_off = (w_ih_byte / 4 + wih_cursor + dir * wih_block) as u32;
            let whh_off = (w_hh_byte / 4 + ld * three_h * hidden) as u32;
            let bih_off = (b_ih_byte / 4 + ld * three_h) as u32;
            let bhh_off = (b_hh_byte / 4 + ld * three_h) as u32;
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

            let mut launcher = stream.launch_builder(&kernel.function);
            launcher
                .arg(&mut *arena)
                .arg(&mut *scratch)
                .arg(&x_off_u)
                .arg(&x_is)
                .arg(&wih_off)
                .arg(&whh_off)
                .arg(&bih_off)
                .arg(&bhh_off)
                .arg(&dst_off_u)
                .arg(&dst_is)
                .arg(&batch_u)
                .arg(&seq_u)
                .arg(&in_l_u)
                .arg(&hidden_u)
                .arg(&h0_off)
                .arg(&out_width_u)
                .arg(&dir_off_u)
                .arg(&reverse_u);
            unsafe {
                launcher.launch(cfg).expect("rlx-cuda: gru launch failed");
            }
        }

        wih_cursor += dirs * wih_block;
        in_l = out_width;
        in_off_f = (l % 2) * layer_elems;
        in_is_scratch = 1;
    }
}
