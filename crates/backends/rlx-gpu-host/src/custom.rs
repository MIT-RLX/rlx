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
        let mut raw = vec![0u8; n * 4];
        if n > 0 {
            a.dtoh(byte_off, &mut raw);
        }
        let f: &[f32] = bytemuck::cast_slice(&raw);
        in_bufs.push(f32_slots_to_dtype(f, sh.dtype()));
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

    let out_f32 = dtype_bytes_to_f32(&out, out_shape.dtype());
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
    if !out_f32.is_empty() {
        a.htod(out_byte_off, bytemuck::cast_slice(&out_f32));
    }
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
