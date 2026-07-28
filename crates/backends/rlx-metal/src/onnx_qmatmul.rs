// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! In-graph GPU path for `onnx.QMatMul` custom ops (Kitten / ORT QDQ matmul).

use std::collections::HashMap;

use metal::{Buffer, ComputeCommandEncoderRef, MTLResourceOptions};
use rlx_ir::{DType, Shape};

use crate::blas::metal_sgemm_bufs;
use crate::device::metal_device;

pub const KERNEL_NAME: &str = "onnx.QMatMul";
pub const BAKED_KERNEL_NAME: &str = "onnx.QMatMulBaked";

pub fn is_ingraph_qmatmul_kernel(name: &str) -> bool {
    name == KERNEL_NAME || name == BAKED_KERNEL_NAME
}

pub fn ingraph_enabled() -> bool {
    if let Some(v) = rlx_ir::env::var("KITTEN_RLX_QMATMUL_INGRAPH") {
        if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no") {
            return false;
        }
        return v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes");
    }
    false
}

/// Opt-in GPU f32 GEMM on the active Metal encoder (requires [`ingraph_enabled`]).
pub fn ingraph_gpu_enabled() -> bool {
    if let Some(v) = rlx_ir::env::var("RLX_METAL_ONNX_QMATMUL_GPU") {
        if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no") {
            return false;
        }
    }
    ingraph_enabled() && rlx_ir::env::flag("RLX_METAL_ONNX_QMATMUL_GPU")
}

pub fn gpu_min_flops() -> usize {
    rlx_ir::env::var("RLX_METAL_ONNX_QMATMUL_MIN_FLOPS")
        .or_else(|| rlx_ir::env::var("KITTEN_RLX_QMATMUL_GPU_MIN_FLOPS"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_097_152)
}

pub fn act_scratch_bytes(graph: &rlx_ir::Graph) -> usize {
    let mut max_mk = 0usize;
    for node in graph.nodes() {
        if let rlx_ir::Op::Custom { name, .. } = &node.op {
            if name != KERNEL_NAME && name != BAKED_KERNEL_NAME {
                continue;
            }
            let (m, k, _) = matmul_dims_from_node(graph, node);
            max_mk = max_mk.max(m.saturating_mul(k).saturating_mul(4));
        }
    }
    max_mk
}

fn matmul_dims_from_node(graph: &rlx_ir::Graph, node: &rlx_ir::Node) -> (usize, usize, usize) {
    let n = node
        .shape
        .dim(node.shape.rank().saturating_sub(1))
        .unwrap_static()
        .max(1);
    let act = graph.node(node.inputs[0]);
    let w = graph.node(node.inputs[3]);
    let k = w.shape.dim(0).unwrap_static().max(1);
    let m = if act.shape.rank() >= 3 {
        act.shape.dim(act.shape.rank() - 2).unwrap_static().max(1)
    } else if act.shape.rank() == 2 {
        act.shape.dim(0).unwrap_static().max(1)
    } else {
        1
    };
    (m.max(1), k.max(1), n.max(1))
}

/// Cached f32 dequant weights keyed by `(w_q arena offset, k, n, w_zp, w_scale bits)`.
pub struct QMatMulWeightCache {
    buffer: Buffer,
    cap_bytes: usize,
    tail: usize,
    map: HashMap<(usize, u32, u32, i32, u32), u64>,
}

impl QMatMulWeightCache {
    pub fn new() -> Self {
        let dev = metal_device().expect("metal device for QMatMul weight cache");
        let cap = 64 * 1024 * 1024;
        Self {
            buffer: dev
                .device
                .new_buffer(cap as u64, MTLResourceOptions::StorageModeShared),
            cap_bytes: cap,
            tail: 0,
            map: HashMap::new(),
        }
    }

    fn ensure(&mut self, need: usize) {
        if self.tail + need <= self.cap_bytes {
            return;
        }
        let dev = metal_device().expect("metal device");
        let new_cap = (self.cap_bytes.max(need) * 2).max(64 * 1024 * 1024);
        self.buffer = dev
            .device
            .new_buffer(new_cap as u64, MTLResourceOptions::StorageModeShared);
        self.cap_bytes = new_cap;
        self.tail = 0;
        self.map.clear();
    }

    fn w_f32_offset(
        &mut self,
        w_q: &[i8],
        w_q_off: usize,
        k: usize,
        n: usize,
        w_zp: i32,
        w_scale: f32,
    ) -> u64 {
        let key = (w_q_off, k as u32, n as u32, w_zp, w_scale.to_bits());
        if let Some(&off) = self.map.get(&key) {
            return off;
        }
        let bytes = k * n * 4;
        self.ensure(self.tail + bytes);
        let off = self.tail as u64;
        self.tail += bytes;
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents().add(off as usize) as *mut f32,
                k * n,
            )
        };
        let wz = w_zp as f32;
        for i in 0..k * n {
            dst[i] = (w_q[i] as f32 - wz) * w_scale;
        }
        self.map.insert(key, off);
        off
    }

    /// Pre-dequantize one weight tile into the cache (no-op if already present).
    pub fn preload_weight(
        &mut self,
        w_q_off: usize,
        w_bytes: &[u8],
        w_dtype: DType,
        k: usize,
        n: usize,
        w_zp: i32,
        w_scale: f32,
    ) {
        let w: &[i8] = match w_dtype {
            DType::I8 => unsafe {
                std::slice::from_raw_parts(w_bytes.as_ptr() as *const i8, w_bytes.len())
            },
            DType::U8 => unsafe {
                std::slice::from_raw_parts(w_bytes.as_ptr() as *const i8, w_bytes.len())
            },
            _ => return,
        };
        let _ = self.w_f32_offset(w, w_q_off, k, n, w_zp, w_scale);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl Default for QMatMulWeightCache {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn read_f32_scalar(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes[..4].try_into().unwrap())
}

pub(crate) fn read_zp_i32(bytes: &[u8], dt: DType) -> i32 {
    match dt {
        DType::I64 => i64::from_le_bytes(bytes[..8].try_into().unwrap()) as i32,
        DType::I32 => i32::from_le_bytes(bytes[..4].try_into().unwrap()),
        DType::U8 => bytes.first().copied().unwrap_or(0) as i32,
        _ => 0,
    }
}

fn read_zp_u8(bytes: &[u8], dt: DType) -> u8 {
    match dt {
        DType::U8 => bytes.first().copied().unwrap_or(0),
        DType::I64 => i64::from_le_bytes(bytes[..8].try_into().unwrap()) as u8,
        DType::I32 => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as u8,
        _ => 0,
    }
}

/// FLOP count for thresholding GPU vs host inline execution.
pub fn matmul_flops(inputs: &[(usize, u32, Shape)], output: &(usize, u32, Shape)) -> usize {
    if inputs.len() < 4 {
        return 0;
    }
    let act_sh = &inputs[0].2;
    let w_sh = &inputs[3].2;
    let n = output
        .2
        .dim(output.2.rank().saturating_sub(1))
        .unwrap_static()
        .max(1);
    let k = w_sh.dim(0).unwrap_static().max(1);
    let m = if act_sh.rank() >= 3 {
        act_sh.dim(act_sh.rank() - 2).unwrap_static().max(1)
    } else if act_sh.rank() == 2 {
        act_sh.dim(0).unwrap_static().max(1)
    } else {
        1
    };
    m.max(1) * k.max(1) * n.max(1)
}

/// Encode one `onnx.QMatMul` on the active compute encoder (no queue sync).
pub fn encode_onnx_qmatmul_f32_gpu(
    enc: &ComputeCommandEncoderRef,
    arena: &Buffer,
    act_scratch_off: usize,
    weight_cache: &mut QMatMulWeightCache,
    inputs: &[(usize, u32, Shape)],
    output: &(usize, u32, Shape),
) {
    if inputs.len() < 6 {
        panic!("onnx.QMatMul expected 6 inputs, got {}", inputs.len());
    }
    let arena_ptr = arena.contents() as *mut u8;
    let read_input = |idx: usize| -> (&[u8], &Shape) {
        let (off, len, shape) = &inputs[idx];
        let nbytes = (*len as usize) * shape.dtype().size_bytes();
        let data = unsafe { std::slice::from_raw_parts(arena_ptr.add(*off), nbytes) };
        (data, shape)
    };

    let (act_q_b, act_q_sh) = read_input(0);
    let (act_scale_b, _) = read_input(1);
    let (act_zp_b, act_zp_sh) = read_input(2);
    let (w_b, w_sh) = read_input(3);
    let (w_scale_b, _) = read_input(4);
    let (w_zp_b, w_zp_sh) = read_input(5);

    let act_q: &[u8] = match act_q_sh.dtype() {
        DType::U8 => act_q_b,
        DType::I8 => act_q_b,
        dt => panic!("onnx.QMatMul act_q expected U8/I8, got {dt:?}"),
    };
    let w: &[i8] = match w_sh.dtype() {
        DType::I8 => unsafe { std::slice::from_raw_parts(w_b.as_ptr() as *const i8, w_b.len()) },
        DType::U8 => unsafe { std::slice::from_raw_parts(w_b.as_ptr() as *const i8, w_b.len()) },
        dt => panic!("onnx.QMatMul w expected I8/U8, got {dt:?}"),
    };

    let act_scale = read_f32_scalar(act_scale_b);
    let act_zp = read_zp_u8(act_zp_b, act_zp_sh.dtype());
    let w_scale = read_f32_scalar(w_scale_b);
    let w_zp = read_zp_i32(w_zp_b, w_zp_sh.dtype());

    let n = output
        .2
        .dim(output.2.rank().saturating_sub(1))
        .unwrap_static()
        .max(1);
    let m = if act_q_sh.rank() >= 3 {
        act_q_sh.dim(act_q_sh.rank() - 2).unwrap_static().max(1)
    } else if act_q_sh.rank() == 2 {
        act_q_sh.dim(0).unwrap_static().max(1)
    } else {
        1
    };
    let k = w_sh.dim(0).unwrap_static().max(1);
    let (m, k, n) = (m.max(1), k.max(1), n.max(1));

    let act_dst = unsafe {
        std::slice::from_raw_parts_mut(arena_ptr.add(act_scratch_off) as *mut f32, m * k)
    };
    let az = act_zp as f32;
    for i in 0..m * k {
        act_dst[i] = (act_q[i] as f32 - az) * act_scale;
    }

    let w_off = weight_cache.w_f32_offset(w, inputs[3].0, k, n, w_zp, w_scale);
    let w_buf = weight_cache.buffer.clone();
    let c_off = output.0;

    metal_sgemm_bufs(
        enc,
        arena,
        act_scratch_off,
        &w_buf,
        w_off as usize,
        arena,
        c_off,
        m,
        k,
        n,
    );
}

/// Baked-weight variant: act QDQ on CPU/scratch, f32 weight already in arena.
pub fn encode_onnx_qmatmul_baked_f32_gpu(
    enc: &ComputeCommandEncoderRef,
    arena: &Buffer,
    act_scratch_off: usize,
    inputs: &[(usize, u32, Shape)],
    output: &(usize, u32, Shape),
) {
    if inputs.len() < 4 {
        panic!("onnx.QMatMulBaked expected 4 inputs, got {}", inputs.len());
    }
    let arena_ptr = arena.contents() as *mut u8;
    let read_input = |idx: usize| -> (&[u8], &Shape) {
        let (off, len, shape) = &inputs[idx];
        let nbytes = (*len as usize) * shape.dtype().size_bytes();
        let data = unsafe { std::slice::from_raw_parts(arena_ptr.add(*off), nbytes) };
        (data, shape)
    };

    let (act_q_b, act_q_sh) = read_input(0);
    let (act_scale_b, _) = read_input(1);
    let (act_zp_b, act_zp_sh) = read_input(2);
    let w_off = inputs[3].0;

    let act_q: &[u8] = match act_q_sh.dtype() {
        DType::U8 | DType::I8 => act_q_b,
        dt => panic!("onnx.QMatMulBaked act_q expected U8/I8, got {dt:?}"),
    };
    let act_scale = read_f32_scalar(act_scale_b);
    let act_zp = read_zp_u8(act_zp_b, act_zp_sh.dtype());

    let n = output
        .2
        .dim(output.2.rank().saturating_sub(1))
        .unwrap_static()
        .max(1);
    let m = if act_q_sh.rank() >= 3 {
        act_q_sh.dim(act_q_sh.rank() - 2).unwrap_static().max(1)
    } else if act_q_sh.rank() == 2 {
        act_q_sh.dim(0).unwrap_static().max(1)
    } else {
        1
    };
    let k = if m > 0 && act_q.len() >= m * n {
        act_q.len() / m
    } else {
        act_q_sh
            .dim(act_q_sh.rank().saturating_sub(1))
            .unwrap_static()
            .max(1)
    };
    let (m, k, n) = (m.max(1), k.max(1), n.max(1));

    let act_dst = unsafe {
        std::slice::from_raw_parts_mut(arena_ptr.add(act_scratch_off) as *mut f32, m * k)
    };
    let az = act_zp as f32;
    for i in 0..m * k {
        act_dst[i] = (act_q[i] as f32 - az) * act_scale;
    }

    metal_sgemm_bufs(
        enc,
        arena,
        act_scratch_off,
        arena,
        w_off,
        arena,
        output.0,
        m,
        k,
        n,
    );
}
