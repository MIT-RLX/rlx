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

//! Native GPU LSTM for CUDA arenas (`Step::Lstm`).
//!
//! Replaces the host D2H→CPU→H2D fallback with the `lstm_dir` kernel: one launch
//! per (layer, direction), the timestep recurrence looped inside the kernel. A
//! whole direction is a single launch, so a 2-layer bidirectional net is 4
//! launches instead of `4·seq` per-step kernels.
//!
//! Bit-for-bit mirror of `rlx_cpu::thunk::execute_lstm_f32` — same packed weight
//! layout (`w_ih` cursor advances `dirs·4h·in_l` per layer; `w_hh`/`bias`/`h0`/`c0`
//! blocks keyed by `ld = l·dirs+dir`), same i/f/g/o gate order, same output slice
//! `[batch, seq, dirs·hidden]`. Falls back (returns `false`) when a layer's
//! `x_t[in_l] + h + c + z[4h]` shared footprint exceeds the 48 KiB default budget.

use cudarc::cublas::sys as cublas_sys;
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use std::sync::{Arc, Mutex, OnceLock};

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
            .expect("rlx-cuda: lstm scratch alloc failed");
        *pool = Some((need, buf));
    }
    &mut pool.as_mut().unwrap().1
}

#[allow(clippy::too_many_arguments)]
pub fn run_lstm(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
    x_byte: usize,
    w_ih_byte: usize,
    w_hh_byte: usize,
    bias_byte: usize,
    h0_byte: usize,
    c0_byte: usize,
    dst_byte: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
) -> bool {
    if batch == 0 || seq == 0 || hidden == 0 || num_layers == 0 {
        return true;
    }
    if rlx_ir::env::flag("RLX_CUDA_LSTM_DEBUG") {
        eprintln!(
            "[lstm_gpu] batch={batch} seq={seq} in={input_size} hidden={hidden} layers={num_layers} bidir={bidirectional} carry={carry}"
        );
    }
    let dirs = if bidirectional { 2 } else { 1 };
    let four_h = 4 * hidden;
    let out_width = dirs * hidden;
    let layer_elems = batch * seq * out_width;

    let max_in_l = input_size.max(out_width);
    if (2 * hidden + four_h) * 4 > 48 * 1024 {
        return false;
    }

    let kernel = crate::kernels::lstm_dir_kernel(ctx);
    let pre_k = crate::kernels::lstm_pre_wih_kernel(ctx);
    let tk = crate::kernels::lstm_transpose_kernel(ctx);

    let wih_t_off = 0usize;
    let whh_t_off = max_in_l * four_h;
    let pre_off = whh_t_off + hidden * four_h;
    let pre_elems = batch * seq * four_h;
    let layer_base = pre_off + pre_elems;
    let scratch_len = layer_base + if num_layers > 1 { 2 * layer_elems } else { 0 };
    let mut pool = scratch_pool().lock().unwrap();
    let scratch = ensure_scratch(stream, scratch_len, &mut pool);

    let block = four_h.min(1024) as u32;
    let sh_floats = 2 * hidden + four_h;

    let mut in_l = input_size;
    let mut wih_cursor = 0usize;
    let mut in_off_f = x_byte / 4;
    let mut in_is_scratch = 0u32;

    for l in 0..num_layers {
        let last = l + 1 == num_layers;
        let out_is_scratch = if last { 0u32 } else { 1u32 };
        let out_off_f = if last {
            dst_byte / 4
        } else {
            layer_base + (l % 2) * layer_elems
        };
        let wih_block = four_h * in_l;
        let cfg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: (sh_floats * 4) as u32,
        };
        for dir in 0..dirs {
            let ld = l * dirs + dir;
            let wih_off = (w_ih_byte / 4 + wih_cursor + dir * wih_block) as u32;
            let whh_off = (w_hh_byte / 4 + ld * four_h * hidden) as u32;

            transpose(
                stream,
                tk,
                arena,
                &mut *scratch,
                wih_off,
                wih_t_off as u32,
                four_h,
                in_l,
            );
            transpose(
                stream,
                tk,
                arena,
                &mut *scratch,
                whh_off,
                whh_t_off as u32,
                four_h,
                hidden,
            );

            let in_off_u = in_off_f as u32;
            let in_is = in_is_scratch;
            let out_off_u = out_off_f as u32;
            let out_is = out_is_scratch;
            let wih_t_u = wih_t_off as u32;
            let whh_t_u = whh_t_off as u32;
            let pre_u = pre_off as u32;
            let bias_off = (bias_byte / 4 + ld * four_h) as u32;
            let h0_off = (h0_byte / 4 + ld * batch * hidden) as u32;
            let c0_off = (c0_byte / 4 + ld * batch * hidden) as u32;
            let carry_u = u32::from(carry);
            let batch_u = batch as u32;
            let seq_u = seq as u32;
            let in_l_u = in_l as u32;
            let four_h_u = four_h as u32;
            let hidden_u = hidden as u32;
            let out_width_u = out_width as u32;
            let dir_u = dir as u32;
            let reverse_u = u32::from(dir == 1);

            let used_cublas = try_pre_wih_cublas(
                stream,
                arena,
                scratch,
                in_off_f,
                in_is_scratch != 0,
                wih_t_off,
                bias_off as usize,
                pre_off,
                batch,
                seq,
                in_l,
                four_h,
            );
            if rlx_ir::env::flag("RLX_CUDA_LSTM_CUBLAS_TRACE") {
                eprintln!(
                    "[lstm_cublas] used={used_cublas} batch={batch} seq={seq} in={in_l} 4h={four_h}"
                );
            }
            if !used_cublas {
                let pre_n = (batch * seq * four_h) as u32;
                let (grid, block_p) = crate::kernels::dispatch_grid_1d(pre_n.max(1), 256);
                let pcfg = LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block_p, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut launcher = stream.launch_builder(&pre_k.function);
                launcher
                    .arg(&mut *arena)
                    .arg(&mut *scratch)
                    .arg(&in_off_u)
                    .arg(&in_is)
                    .arg(&wih_t_u)
                    .arg(&bias_off)
                    .arg(&pre_u)
                    .arg(&batch_u)
                    .arg(&seq_u)
                    .arg(&in_l_u)
                    .arg(&four_h_u);
                unsafe {
                    launcher
                        .launch(pcfg)
                        .expect("rlx-cuda: lstm_pre_wih launch failed");
                }
            }

            let mut launcher = stream.launch_builder(&kernel.function);
            launcher
                .arg(&mut *arena)
                .arg(&mut *scratch)
                .arg(&in_off_u)
                .arg(&in_is)
                .arg(&out_off_u)
                .arg(&out_is)
                .arg(&wih_t_u)
                .arg(&whh_t_u)
                .arg(&bias_off)
                .arg(&h0_off)
                .arg(&c0_off)
                .arg(&carry_u)
                .arg(&batch_u)
                .arg(&seq_u)
                .arg(&in_l_u)
                .arg(&hidden_u)
                .arg(&out_width_u)
                .arg(&dir_u)
                .arg(&reverse_u)
                .arg(&pre_u);
            unsafe {
                launcher
                    .launch(cfg)
                    .expect("rlx-cuda: lstm_dir launch failed");
            }
        }
        wih_cursor += dirs * wih_block;
        in_off_f = out_off_f;
        in_is_scratch = out_is_scratch;
        in_l = out_width;
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn try_pre_wih_cublas(
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
    scratch: &mut CudaSlice<f32>,
    in_off_f: usize,
    in_is_scratch: bool,
    wih_t_off: usize,
    bias_off: usize,
    pre_off: usize,
    batch: usize,
    seq: usize,
    in_l: usize,
    four_h: usize,
) -> bool {
    let Some(blas) = crate::device::cuda_blas() else {
        return false;
    };
    let blas = blas.lock().unwrap();
    if unsafe {
        cudarc::cublas::result::set_stream(*blas.handle(), stream.cu_stream() as _)
    }
    .is_err()
    {
        return false;
    }
    let (arena_ptr, arena_rec) = arena.device_ptr_mut(stream);
    let (scratch_ptr, scratch_rec) = scratch.device_ptr_mut(stream);
    let x_base = if in_is_scratch { scratch_ptr } else { arena_ptr };
    let x_dev = x_base + (in_off_f as u64) * 4;
    let wih_dev = scratch_ptr + (wih_t_off as u64) * 4;
    let pre_dev = scratch_ptr + (pre_off as u64) * 4;
    let m = (batch * seq) as i32;
    let k = in_l as i32;
    let n = four_h as i32;
    if m == 0 || k == 0 || n == 0 {
        return true;
    }
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let ok = unsafe {
        cudarc::cublas::result::sgemm(
            *blas.handle(),
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha as *const f32,
            wih_dev as *const f32,
            n,
            x_dev as *const f32,
            k,
            &beta as *const f32,
            pre_dev as *mut f32,
            n,
        )
    };
    drop(arena_rec);
    drop(scratch_rec);
    if ok.is_err() {
        return false;
    }
    let ctx = crate::device::cuda_context().expect("rlx-cuda: context for lstm bias");
    let add_k = crate::kernels::lstm_pre_add_bias_kernel(&ctx);
    let len = (batch * seq * four_h) as u32;
    let pre_u = pre_off as u32;
    let bias_u = bias_off as u32;
    let four_h_u = four_h as u32;
    let (grid, block) = crate::kernels::dispatch_grid_1d(len.max(1), 256);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&add_k.function);
    launcher
        .arg(&mut *scratch)
        .arg(&pre_u)
        .arg(&bias_u)
        .arg(&mut *arena)
        .arg(&len)
        .arg(&four_h_u);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: lstm_pre_add_bias launch failed");
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn transpose(
    stream: &Arc<CudaStream>,
    tk: &crate::kernels::CudaKernel,
    arena: &mut CudaSlice<f32>,
    scratch: &mut CudaSlice<f32>,
    src_off: u32,
    dst_off: u32,
    rows: usize,
    cols: usize,
) {
    let total = (rows * cols) as u32;
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let mut launcher = stream.launch_builder(&tk.function);
    launcher
        .arg(&mut *arena)
        .arg(&mut *scratch)
        .arg(&src_off)
        .arg(&dst_off)
        .arg(&rows_u)
        .arg(&cols_u);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: lstm transpose launch failed");
    }
}
