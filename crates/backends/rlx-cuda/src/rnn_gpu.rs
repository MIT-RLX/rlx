// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native GPU Elman RNN for CUDA arenas (`Step::Rnn`).
//!
//! Single-layer / unidirectional / no-carry / `hidden ≤ 1024` — same coverage
//! as Metal `rnn` and wgpu `rnn.wgsl`. `relu` selects ReLU vs tanh; h0 = 0.

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Max hidden size for the native kernel (matches Metal `RNN_MAX_H`).
pub const RNN_MAX_H: usize = 1024;

/// True when the native `rnn` kernel can run this geometry.
#[inline]
pub fn native_rnn_ok(num_layers: usize, bidirectional: bool, carry: bool, hidden: usize) -> bool {
    num_layers == 1 && !bidirectional && !carry && hidden > 0 && hidden <= RNN_MAX_H
}

/// Launch native Elman RNN. All `*_byte` args are byte offsets into `arena`.
#[allow(clippy::too_many_arguments)]
pub fn run_rnn(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
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
    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (hidden as u32, 1, 1),
        shared_mem_bytes: 0,
    };
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
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *arena)
        .arg(&x_off)
        .arg(&wih_off)
        .arg(&whh_off)
        .arg(&bias_off)
        .arg(&dst_off)
        .arg(&batch_u)
        .arg(&seq_u)
        .arg(&in_u)
        .arg(&hidden_u)
        .arg(&relu_u);
    unsafe {
        launcher.launch(cfg).expect("rlx-cuda: rnn launch failed");
    }
}
