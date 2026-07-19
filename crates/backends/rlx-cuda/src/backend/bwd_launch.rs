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

use std::sync::Arc;

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

use crate::kernels::{
    cumsum_backward_kernel, dispatch_grid_1d, gather_backward_kernel, rms_norm_backward_kernel,
    rms_norm_bwd_zero_kernel, rope_backward_kernel,
};

pub(crate) fn launch_cumsum_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    outer: u32,
    inner: u32,
    dy_off: u32,
    dx_off: u32,
    exclusive: u32,
) {
    let kernel = cumsum_backward_kernel(ctx);
    let (grid, block) = dispatch_grid_1d(outer, 256);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(buffer)
        .arg(&outer)
        .arg(&inner)
        .arg(&dy_off)
        .arg(&dx_off)
        .arg(&exclusive);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: cumsum_bwd launch failed");
    }
}

pub(crate) fn launch_rope_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    dy_off: u32,
    cos_off: u32,
    sin_off: u32,
    dx_off: u32,
    cos_len: u32,
) {
    let total = batch * seq * hidden;
    let kernel = rope_backward_kernel(ctx);
    let (grid, block) = dispatch_grid_1d(total, 256);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(buffer)
        .arg(&batch)
        .arg(&seq)
        .arg(&hidden)
        .arg(&head_dim)
        .arg(&n_rot)
        .arg(&dy_off)
        .arg(&cos_off)
        .arg(&sin_off)
        .arg(&dx_off)
        .arg(&cos_len);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: rope_bwd launch failed");
    }
}

pub(crate) fn launch_gather_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
    dy_off: u32,
    idx_off: u32,
    dst_off: u32,
) {
    let total = outer * axis_dim * trailing;
    if total > 0 {
        let zk = rms_norm_bwd_zero_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(total, 256);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut zl = stream.launch_builder(&zk.function);
        zl.arg(&mut *buffer).arg(&dst_off).arg(&total);
        unsafe {
            zl.launch(cfg)
                .expect("rlx-cuda: gather_bwd zero launch failed");
        }
    }
    let kernel = gather_backward_kernel(ctx);
    let cfg = LaunchConfig {
        grid_dim: (outer, (num_idx * trailing).div_ceil(256), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&outer)
        .arg(&axis_dim)
        .arg(&num_idx)
        .arg(&trailing)
        .arg(&dy_off)
        .arg(&idx_off)
        .arg(&dst_off);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: gather_bwd launch failed");
    }
}

pub(crate) fn launch_rms_norm_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    rows: u32,
    inner: u32,
    x_off: u32,
    gamma_off: u32,
    beta_off: u32,
    dy_off: u32,
    out_off: u32,
    eps_bits: u32,
    wrt: u32,
) {
    if wrt != 0 {
        let zk = rms_norm_bwd_zero_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(inner, 256);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut zl = stream.launch_builder(&zk.function);
        zl.arg(&mut *buffer).arg(&out_off).arg(&inner);
        unsafe {
            zl.launch(cfg)
                .expect("rlx-cuda: rms_norm_bwd zero launch failed");
        }
    }
    let kernel = rms_norm_backward_kernel(ctx);
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&rows)
        .arg(&inner)
        .arg(&x_off)
        .arg(&gamma_off)
        .arg(&beta_off)
        .arg(&dy_off)
        .arg(&out_off)
        .arg(&eps_bits)
        .arg(&wrt);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: rms_norm_bwd launch failed");
    }
}
