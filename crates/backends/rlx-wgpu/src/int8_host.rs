// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side Int8 block `Op::DequantMatMul` for discrete wgpu (Vulkan/DX12).
//!
//! The SPIR-V `dequant_matmul` WGSL path is wrong on discrete NVIDIA (Kitten
//! NSF → near-DC mush). Metal wgpu is fine; rlx-vulkan already hosts all
//! non-GGUF DequantMatMul. Mirror that for Int8Block{,Asym} here.
//!
//! Packed i8 weights live in the f32-uniform arena as raw bytes (4 elems per
//! f32 slot), matching `set_param_bytes` / [`rlx_gpu_host`] custom-op staging.
//! wgpu `copy_buffer_to_buffer` requires 4-byte sizes — pad like custom_host.

use crate::buffer::Arena;
use crate::host_stage::WgpuArena;
use rlx_gpu_host::DeviceArena;

#[inline]
fn round_up_4(n: usize) -> usize {
    n.div_ceil(4) * 4
}

fn dtoh_bytes(a: &mut WgpuArena<'_>, byte_off: usize, n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    let mut raw = vec![0u8; round_up_4(n)];
    a.dtoh(byte_off, &mut raw);
    raw.truncate(n);
    raw
}

fn dtoh_f32(a: &mut WgpuArena<'_>, byte_off: usize, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let mut raw = vec![0u8; n * 4];
    a.dtoh(byte_off, &mut raw);
    bytemuck::cast_slice(&raw).to_vec()
}

/// Host Int8Block{,Asym} `DequantMatMul`: D2H packed i8 + scales, CPU matmul, H2D.
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_int8(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    is_asymmetric: bool,
    x_byte_off: usize,
    w_byte_off: usize,
    scale_byte_off: usize,
    zp_byte_off: usize,
    out_byte_off: usize,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    let blocks = k.div_ceil(block_size);
    let scale_elems = blocks * n;

    let x = dtoh_f32(&mut a, x_byte_off, m * k);
    let w_raw = dtoh_bytes(&mut a, w_byte_off, k * n);
    let w_i8: Vec<i8> = w_raw.iter().map(|&b| b as i8).collect();
    let scales = dtoh_f32(&mut a, scale_byte_off, scale_elems);
    let zps = if is_asymmetric {
        dtoh_f32(&mut a, zp_byte_off, scale_elems)
    } else {
        Vec::new()
    };
    let mut out = vec![0f32; m * n];

    rlx_cpu::thunk::dequant_matmul_int8(
        &x,
        &w_i8,
        &scales,
        &zps,
        &mut out,
        m,
        k,
        n,
        block_size,
        is_asymmetric,
    );

    if rlx_ir::env::flag("RLX_WGPU_DBG_INT8_HOST") {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let x_abs: f32 = x.iter().map(|v| v.abs()).sum();
            let w_abs: i64 = w_i8.iter().map(|&v| i64::from(v.unsigned_abs())).sum();
            let s_abs: f32 = scales.iter().map(|v| v.abs()).sum();
            let o_abs: f32 = out.iter().map(|v| v.abs()).sum();
            let o_peak = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "[wgpu-int8-host] m={m} k={k} n={n} bs={block_size} asym={is_asymmetric} \
                 x_off={x_byte_off:#x} w_off={w_byte_off:#x} \
                 |x|={x_abs:.4} |w|={w_abs} |s|={s_abs:.4} |out|={o_abs:.4} out_peak={o_peak:.6}"
            );
        }
    }

    if !out.is_empty() {
        a.htod(out_byte_off, bytemuck::cast_slice(&out));
    }
}
