// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side RNG fill for device arenas (fill on host → H2D).

use crate::DeviceArena;

/// Fill `len` f32s at `dst_byte_off` with a normal draw (`mean`, `scale`).
pub fn run_rng_normal<A: DeviceArena>(
    a: &mut A,
    dst_byte_off: usize,
    len: usize,
    mean: f32,
    scale: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    if len == 0 {
        return;
    }
    assert_eq!(
        dst_byte_off % 4,
        0,
        "rng_host: dst_byte_off must be f32-aligned"
    );
    let mut host = vec![0f32; len];
    rlx_ir::fill_normal_like(&mut host, mean, scale, opts, key, op_seed);
    a.htod(dst_byte_off, bytemuck::cast_slice(&host));
}

/// Fill `len` f32s at `dst_byte_off` with a uniform draw on `[low, high)`.
pub fn run_rng_uniform<A: DeviceArena>(
    a: &mut A,
    dst_byte_off: usize,
    len: usize,
    low: f32,
    high: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
) {
    if len == 0 {
        return;
    }
    assert_eq!(
        dst_byte_off % 4,
        0,
        "rng_host: dst_byte_off must be f32-aligned"
    );
    let mut host = vec![0f32; len];
    rlx_ir::fill_uniform_like(&mut host, low, high, opts, key, op_seed);
    a.htod(dst_byte_off, bytemuck::cast_slice(&host));
}
