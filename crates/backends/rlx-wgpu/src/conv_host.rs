// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side NCHW Conv2d for discrete wgpu (Vulkan/DX12).
//!
//! Stages only the three touched tensors (not the whole multi-GiB arena).
//! Weight tensors are cached across calls (Kitten reuses the same kernels
//! many times per chunk).

use crate::buffer::{Arena, is_weight_off};
use crate::host_stage::WgpuArena;
use rlx_gpu_host::{DeviceArena, HostTensorCache};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn weight_cache() -> &'static Mutex<HashMap<(usize, usize), Vec<f32>>> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize), Vec<f32>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn dtoh_f32(a: &mut WgpuArena<'_>, byte_off: usize, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let mut raw = vec![0u8; n * 4];
    a.dtoh(byte_off, &mut raw);
    bytemuck::cast_slice(&raw).to_vec()
}

fn dtoh_f32_cached(
    a: &mut WgpuArena<'_>,
    byte_off: usize,
    n: usize,
    act_cache: Option<&mut HostTensorCache>,
) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if let Some(c) = act_cache.as_ref() {
        if let Some(hit) = c.get_arc_covering(byte_off, n) {
            return hit[..n].to_vec();
        }
    }
    if let Some(c) = act_cache {
        c.flush_offset(a, byte_off);
    }
    if is_weight_off(byte_off) {
        let key = (byte_off, n);
        if let Ok(guard) = weight_cache().lock() {
            if let Some(hit) = guard.get(&key) {
                return hit.clone();
            }
        }
        let v = dtoh_f32(a, byte_off, n);
        if let Ok(mut guard) = weight_cache().lock() {
            guard.insert(key, v.clone());
        }
        return v;
    }
    dtoh_f32(a, byte_off, n)
}

/// Host NCHW Conv2d: stage touched tensors, run CPU forward, write `out` back.
#[allow(clippy::too_many_arguments)]
pub fn run_conv2d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
    in_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    run_conv2d_cached(
        arena,
        device,
        queue,
        n,
        c_in,
        c_out,
        h,
        w,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        groups,
        in_byte_off,
        w_byte_off,
        out_byte_off,
        None,
    );
}

/// Like [`run_conv2d`], with optional activation [`HostTensorCache`] and a
/// process-wide weight D2H cache keyed by arena byte offset.
#[allow(clippy::too_many_arguments)]
pub fn run_conv2d_cached(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
    in_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
    mut act_cache: Option<&mut HostTensorCache>,
) {
    let mut a = WgpuArena {
        arena,
        device,
        queue,
        size_bytes: 0,
    };
    let c_in_per_g = c_in / groups.max(1);
    let inp = dtoh_f32_cached(
        &mut a,
        in_byte_off,
        n * c_in * h * w,
        act_cache.as_deref_mut(),
    );
    let wt = dtoh_f32_cached(
        &mut a,
        w_byte_off,
        c_out * c_in_per_g * kh * kw,
        act_cache.as_deref_mut(),
    );
    let mut out = vec![0f32; n * c_out * h_out * w_out];
    rlx_cpu::conv_fwd::conv2d_forward_nchw_f32(
        &inp, &wt, &mut out, n, c_in, h, w, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw,
        groups,
    );
    if !out.is_empty() {
        let defer = act_cache.is_some() && !rlx_ir::env::flag("RLX_WGPU_HOST_EAGER_H2D");
        if !defer {
            a.htod(out_byte_off, bytemuck::cast_slice(&out));
        }
        if let Some(c) = act_cache {
            c.insert(out_byte_off, out, defer);
        }
    }
}
