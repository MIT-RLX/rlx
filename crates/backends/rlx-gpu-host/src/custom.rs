// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic host-delegate for `Op::Custom` on f32-uniform GPU arenas.
//!
//! Integer/float tensors occupy one f32 slot per element. Stage slots →
//! re-encode to the operand's declared dtype → run the registered
//! `rlx-cpu` kernel → cast the result back to f32 slots.

use crate::DeviceArena;
use rlx_ir::{DType, Shape};
use std::sync::Once;

/// Whether a generic host-delegate exists for `name` (a registered rlx-cpu
/// reference kernel). Registers the ONNX reference kernels once.
pub fn has_host_kernel(name: &str) -> bool {
    static REG: Once = Once::new();
    REG.call_once(rlx_cpu::onnx_ref::register_onnx_reference_kernels);
    rlx_cpu::op_registry::lookup_cpu_kernel(name).is_some()
}

/// Re-encode `n` f32 arena slots into the little-endian bytes of `dtype`.
pub fn f32_slots_to_dtype(f: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => f.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F64 => f.iter().flat_map(|&x| (x as f64).to_le_bytes()).collect(),
        DType::I64 => f.iter().flat_map(|&x| (x as i64).to_le_bytes()).collect(),
        DType::I32 => f.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect(),
        DType::I16 => f.iter().flat_map(|&x| (x as i16).to_le_bytes()).collect(),
        DType::I8 => f.iter().map(|&x| x as i8 as u8).collect(),
        DType::U32 => f.iter().flat_map(|&x| (x as u32).to_le_bytes()).collect(),
        DType::U8 => f.iter().map(|&x| x as u8).collect(),
        DType::Bool => f.iter().map(|&x| u8::from(x != 0.0)).collect(),
        _ => f.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// Decode the kernel's little-endian `dtype` output bytes back to f32 slots.
pub fn dtype_bytes_to_f32(b: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        DType::F64 => b
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I64 => b
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I32 => b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I16 => b
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I8 => b.iter().map(|&x| x as i8 as f32).collect(),
        DType::U32 => b
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::U8 | DType::Bool => b.iter().map(|&x| x as f32).collect(),
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}

/// Stage custom-op inputs/outputs using **byte** offsets into an f32-uniform arena.
pub fn run_custom_host_bytes<A: DeviceArena>(
    a: &mut A,
    name: &str,
    in_specs: &[(usize, Shape)],
    out_byte_off: usize,
    out_shape: &Shape,
    attrs: &[u8],
) {
    let _ = has_host_kernel(name);
    a.sync();

    let mut in_bufs: Vec<Vec<u8>> = Vec::with_capacity(in_specs.len());
    for &(byte_off, ref sh) in in_specs {
        let n = sh.num_elements().unwrap_or(0);
        if is_byte_packed(sh.dtype()) {
            // U8/I8 tensors are byte-packed in the f32-uniform arena (4 elems per
            // f32 slot — matches `set_param_bytes` and the backend arena's
            // `elems`-byte slot). Read the raw dtype bytes directly; do NOT treat
            // each element as its own f32 slot (that reads 4x too much and mangles
            // quantized activations/weights, e.g. Kitten `QMatMul` act_q / weights).
            //
            // Quantized weight params are immutable for the life of a compiled
            // graph — cache by arena offset so Kitten's DynamicQuantizeLSTM
            // (5×/wave) does not re-D2H the same int8 W/R every call.
            let cached = PARAM_CACHE.with(|c| c.borrow().get(&byte_off).cloned());
            let raw = if let Some(hit) = cached {
                hit
            } else {
                let mut raw = vec![0u8; round_up_4(n)];
                if n > 0 {
                    a.dtoh(byte_off, &mut raw);
                }
                raw.truncate(n);
                PARAM_CACHE.with(|c| {
                    c.borrow_mut().insert(byte_off, raw.clone());
                });
                raw
            };
            in_bufs.push(raw);
        } else {
            let mut raw = vec![0u8; n * 4];
            if n > 0 {
                a.dtoh(byte_off, &mut raw);
            }
            let f: &[f32] = bytemuck::cast_slice(&raw);
            in_bufs.push(f32_slots_to_dtype(f, sh.dtype()));
        }
    }
    let in_pairs: Vec<(&[u8], &Shape)> = in_bufs
        .iter()
        .zip(in_specs.iter())
        .map(|(buf, (_, sh))| (buf.as_slice(), sh))
        .collect();

    let out_n = out_shape.num_elements().unwrap_or(0);
    let mut out = vec![0u8; out_n * out_shape.dtype().size_bytes()];
    rlx_cpu::op_registry::run_custom_op_host(name, &in_pairs, (&mut out, out_shape), attrs)
        .unwrap_or_else(|e| panic!("rlx-gpu-host custom-op '{name}': {e}"));

    if std::env::var("RLX_DBG_CUSTOM").is_ok() {
        eprintln!(
            "[gpu-host-custom] {name} out_off={out_byte_off} out_dtype={:?} out_n={out_n} in={:?}",
            out_shape.dtype(),
            in_specs
                .iter()
                .map(|(o, s)| (*o, s.dtype(), s.num_elements()))
                .collect::<Vec<_>>(),
        );
    }
    if out_n == 0 {
        return;
    }
    if is_byte_packed(out_shape.dtype()) {
        // Byte-packed output: write the raw dtype bytes back, padded up to the
        // f32-slot boundary the arena allocated (never treated as f32 elements).
        let mut packed = out;
        packed.resize(round_up_4(out_n), 0);
        a.htod(out_byte_off, &packed);
    } else {
        let out_f32 = dtype_bytes_to_f32(&out, out_shape.dtype());
        if !out_f32.is_empty() {
            a.htod(out_byte_off, bytemuck::cast_slice(&out_f32));
        }
    }
}

thread_local! {
    static PARAM_CACHE: std::cell::RefCell<std::collections::HashMap<usize, Vec<u8>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Drop cached int8/u8 custom-op params (call after recompiling / swapping arenas).
pub fn clear_custom_param_cache() {
    PARAM_CACHE.with(|c| c.borrow_mut().clear());
}

/// U8/I8 tensors live byte-packed (1 native byte/elem) in the f32-uniform arena,
/// unlike Bool (written as 1.0/0.0 f32 lanes) and every wider dtype (one f32 lane
/// per element). Keep in sync with the backend arenas' slot sizing.
#[inline]
fn is_byte_packed(dtype: DType) -> bool {
    matches!(dtype, DType::U8 | DType::I8)
}

/// Round a byte length up to the f32-slot (4-byte) boundary the arena aligns to.
#[inline]
fn round_up_4(n: usize) -> usize {
    n.div_ceil(4) * 4
}

/// Same as [`run_custom_host_bytes`] with **f32-element** offsets (CUDA layout).
pub fn run_custom_host_f32<A: DeviceArena>(
    a: &mut A,
    name: &str,
    in_specs: &[(usize, Shape)],
    out_f32_off: usize,
    out_shape: &Shape,
    attrs: &[u8],
) {
    let byte_specs: Vec<(usize, Shape)> = in_specs
        .iter()
        .map(|(off, sh)| (*off * 4, sh.clone()))
        .collect();
    run_custom_host_bytes(a, name, &byte_specs, out_f32_off * 4, out_shape, attrs);
}
