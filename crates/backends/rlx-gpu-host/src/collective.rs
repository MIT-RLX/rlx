// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side `Op::Custom("collective.*")` staging (no device kernel).
//!
//! Collectives are host/transport ops: stage the operand off-GPU, run the
//! registered `rlx-cpu` kernel, write the result back.

use crate::DeviceArena;
use rlx_ir::{DType, Shape};

/// Collective op names handled by GPU host delegates. Mirrors
/// `rlx_collectives::{ALL_REDUCE, ALL_GATHER, …}` — keep in sync.
pub const COLLECTIVE_OPS: &[&str] = &[
    "collective.all_reduce",
    "collective.all_gather",
    "collective.reduce_scatter",
    "collective.copy_to_parallel",
    "collective.reduce_from_parallel",
    "collective.broadcast",
    "collective.reduce",
    "collective.all_to_all",
    "collective.moe_dispatch",
    "collective.moe_combine",
    "collective.ppermute",
    "collective.send",
    "collective.recv",
];

/// Stage one f32 input span, run the registered collective kernel, write the
/// f32 output span. Offsets/lengths are **bytes** (must be f32-aligned).
pub fn run_collective_bytes<A: DeviceArena>(
    a: &mut A,
    name: &str,
    in_byte_off: usize,
    in_bytes: usize,
    out_byte_off: usize,
    out_bytes: usize,
    attrs: &[u8],
) {
    assert!(
        in_byte_off.is_multiple_of(4) && in_bytes.is_multiple_of(4),
        "collective: in span must be f32-aligned"
    );
    assert!(
        out_byte_off.is_multiple_of(4) && out_bytes.is_multiple_of(4),
        "collective: out span must be f32-aligned"
    );
    a.sync();
    let mut in_raw = vec![0u8; in_bytes];
    a.dtoh(in_byte_off, &mut in_raw);
    let in_f32: &[f32] = bytemuck::cast_slice(&in_raw);
    let mut out_f32 = vec![0f32; out_bytes / 4];
    let in_shape = Shape::new(&[in_f32.len()], DType::F32);
    let out_shape = Shape::new(&[out_f32.len()], DType::F32);
    rlx_cpu::op_registry::run_f32_custom_op_host(
        name,
        &[(bytemuck::cast_slice(in_f32), &in_shape)],
        (bytemuck::cast_slice_mut(&mut out_f32), &out_shape),
        attrs,
    )
    .unwrap_or_else(|e| panic!("rlx-gpu-host collective '{name}': {e}"));
    a.htod(out_byte_off, bytemuck::cast_slice(&out_f32));
}

/// Same as [`run_collective_bytes`] with **f32-element** offsets/lengths
/// (CUDA/ROCm historical layout).
pub fn run_collective_f32<A: DeviceArena>(
    a: &mut A,
    name: &str,
    in_f32_off: usize,
    in_len: usize,
    out_f32_off: usize,
    out_len: usize,
    attrs: &[u8],
) {
    run_collective_bytes(
        a,
        name,
        in_f32_off * 4,
        in_len * 4,
        out_f32_off * 4,
        out_len * 4,
        attrs,
    );
}
