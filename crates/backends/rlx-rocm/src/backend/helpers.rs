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

use std::sync::{Arc, Mutex};

use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{Graph, NodeId};

use crate::arena::{Arena, HalfDtype};
use crate::device::RocmContext;
use crate::hip::{HipBuffer, HipDeviceptr};
use crate::hipblas::{
    HipblasComputeType, HipblasContext, HipblasDatatype, HipblasOperation, hipblas_gemm_default,
};

use super::ExecMode;

// ── log_fallback (port from rlx-cuda) ────────────────────────────────

pub(crate) fn log_fallback(tier: &str, err: impl std::fmt::Debug) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        rlx_ir::env::var("RLX_ROCM_LOG_FALLBACK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    if enabled {
        eprintln!("rlx-rocm: tier '{tier}' fell back: {err:?}");
    }
}

// ── step_name (port from rlx-cuda) ────────────────────────────────────

pub(crate) fn rocm_fft_dtype_tag(dtype: rlx_ir::DType) -> u32 {
    match dtype {
        rlx_ir::DType::F32 => 0,
        rlx_ir::DType::F64 => 1,
        rlx_ir::DType::C64 => 2,
        other => panic!("rlx-rocm Op::Fft: unsupported dtype {other:?}"),
    }
}

pub(crate) fn rocm_fft_dtype_from_tag(tag: u32) -> rlx_ir::DType {
    match tag {
        0 => rlx_ir::DType::F32,
        1 => rlx_ir::DType::F64,
        2 => rlx_ir::DType::C64,
        other => panic!("rlx-rocm Op::Fft: bad dtype tag {other}"),
    }
}

// ── Op-id encoders + matmul shape (port from rlx-cuda) ───────────────

pub(crate) fn matmul_shape(
    graph: &Graph,
    node: &rlx_ir::Node,
    op_label: &str,
) -> (u32, u32, u32, u32, u32, u32, u32, NodeId, NodeId) {
    let a_id = node.inputs[0];
    let b_id = node.inputs[1];
    let a_shape = graph.node(a_id).shape.dims();
    let b_shape = graph.node(b_id).shape.dims();
    let out_shape = node.shape.dims();
    if a_shape.len() == 2 && b_shape.len() == 2 && out_shape.len() == 2 {
        let m = a_shape[0].unwrap_static() as u32;
        let k = a_shape[1].unwrap_static() as u32;
        let n = b_shape[1].unwrap_static() as u32;
        (m, k, n, 1, 0, 0, 0, a_id, b_id)
    } else if a_shape.len() >= 2 && b_shape.len() == 2 && out_shape.len() == a_shape.len() {
        let leading: usize = a_shape[..a_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .product();
        let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
        let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
        let n_inner = b_shape[1].unwrap_static();
        (
            (leading * m_inner) as u32,
            k_inner as u32,
            n_inner as u32,
            1,
            0,
            0,
            0,
            a_id,
            b_id,
        )
    } else if a_shape.len() == b_shape.len() && a_shape.len() >= 3 {
        let leading_a: Vec<usize> = a_shape[..a_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let leading_b: Vec<usize> = b_shape[..b_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        if leading_a != leading_b {
            panic!(
                "rlx-rocm {op_label}: batched shape mismatch \
                    a_leading={leading_a:?} b_leading={leading_b:?}"
            );
        }
        let b_count: usize = leading_a.iter().product();
        let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
        let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
        let n_inner = b_shape[b_shape.len() - 1].unwrap_static();
        (
            m_inner as u32,
            k_inner as u32,
            n_inner as u32,
            b_count as u32,
            (m_inner * k_inner) as u32,
            (k_inner * n_inner) as u32,
            (m_inner * n_inner) as u32,
            a_id,
            b_id,
        )
    } else {
        panic!(
            "rlx-rocm {op_label}: unsupported shapes a={a_shape:?} b={b_shape:?} out={out_shape:?}"
        );
    }
}

pub(crate) fn binary_op_id(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Max => 4,
        BinaryOp::Min => 5,
        BinaryOp::Pow => 6,
    }
}

pub(crate) fn compare_op_id(op: CmpOp) -> u32 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

pub(crate) fn reduce_op_id(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    }
}

pub(crate) fn activation_op_id(act: Activation) -> u32 {
    match act {
        Activation::Relu => 0,
        Activation::Sigmoid => 1,
        Activation::Tanh => 2,
        Activation::Exp => 3,
        Activation::Log => 4,
        Activation::Sqrt => 5,
        Activation::Rsqrt => 6,
        Activation::Neg => 7,
        Activation::Abs => 8,
        Activation::Gelu => 9,
        Activation::Silu => 10,
        Activation::GeluApprox => 11,
        Activation::Round => 12,
        Activation::Sin => 13,
        Activation::Cos => 14,
        Activation::Tan => 15,
        Activation::Atan => 16,
        Activation::Recip => 17,
    }
}

/// Upload a `&[u32]` to a freshly-allocated device buffer (analogue of
/// cudarc's `stream.clone_htod`). Used for transpose / expand meta
/// buffers.
pub(crate) fn upload_meta(ctx: &Arc<RocmContext>, data: &[u32]) -> HipBuffer<u32> {
    let mut buf = HipBuffer::<u32>::alloc_zeros(&ctx.runtime, data.len().max(1))
        .expect("rlx-rocm: meta upload alloc failed");
    buf.copy_from_host(data)
        .expect("rlx-rocm: meta upload htod failed");
    buf
}

/// Upload an arbitrary `&[f32]` slice to a specific arena offset
/// (used for Constant nodes during compile).
pub(crate) fn upload_to_arena(
    ctx: &Arc<RocmContext>,
    arena_ptr: HipDeviceptr,
    off_f32: usize,
    data: &[f32],
) {
    let dst = arena_ptr + (off_f32 as u64) * 4;
    let bytes = std::mem::size_of_val(data);
    unsafe {
        let _ = (ctx.runtime.hip_memcpy_htod)(dst, data.as_ptr() as *const _, bytes);
    }
}

/// Opt-in MFMA / WMMA matrix-core kernel via rocWMMA. Reads
/// `RLX_ROCM_MFMA=1` once at process start. When true and the higher
/// tiers (mixed-precision, hipBLASLt, hipBLAS) all decline, the
/// matmul dispatch picks the matrix-core kernel instead of the
/// scalar fallback. The kernel will fail to compile under hipRTC on
/// archs without rocWMMA support; the cache miss surfaces as a
/// clean fallback through the normal panic path here, so we keep
/// this opt-in.
pub(crate) fn use_mfma() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        rlx_ir::env::var("RLX_ROCM_MFMA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Mixed-precision matmul tier: when the weight (B input) is stored
/// in the half-arena, cast f32 activations to f16/bf16 in the scratch
/// buffer and run `hipblasGemmEx` with both inputs half + f32
/// accumulator. Returns `true` on success. Same shape as
/// `rlx-cuda::backend::try_mixed_precision_gemm` (free function so the
/// caller can hold `&self.schedule` across the call without violating
/// disjoint-field borrow checks).
pub(crate) fn try_mixed_precision_gemm_rocm(
    ctx: &Arc<RocmContext>,
    arena: &mut Arena,
    half_act_scratch: &mut Option<HipBuffer<u16>>,
    blas: Option<&Arc<Mutex<HipblasContext>>>,
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    a_off_f32: u32,
    b_off_f32: u32,
    c_off_f32: u32,
) -> bool {
    let (half_off, half_dtype) = match arena.half_by_f32_off.get(&b_off_f32).copied() {
        Some(v) => v,
        None => return false,
    };
    let blas = match blas {
        Some(b) => b,
        None => return false,
    };

    let act_elems = (m * k * batch.max(1)) as usize;
    let need_resize = half_act_scratch.as_ref().is_none_or(|s| s.len < act_elems);
    if need_resize {
        *half_act_scratch = HipBuffer::<u16>::alloc_zeros(&ctx.runtime, act_elems.max(4)).ok();
    }
    if half_act_scratch.is_none() {
        return false;
    }

    // Phase 1: cast activations f32 → f16/bf16 into the scratch.
    let n_total = m * k * batch.max(1);
    let dtype_id: u32 = match half_dtype {
        HalfDtype::F16 => 0,
        HalfDtype::Bf16 => 1,
    };
    let stream = ctx.default_stream;
    let kernel = crate::kernels::cast_f32_to_half_kernel(ctx);
    let arena_base = arena.buffer.ptr;
    let scratch_ptr = half_act_scratch.as_ref().unwrap().ptr;
    // The cast kernel takes a `float*` source pointer (already at the
    // input offset) and a `unsigned short*` dest. We use raw pointer
    // values so the kernel reads from a_off + i.
    let src_dev = arena_base + (a_off_f32 as u64) * 4;
    let mut src_pp = src_dev;
    let mut dst_pp = scratch_ptr;
    crate::launch_kernel!(
        kernel,
        stream,
        (n_total.div_ceil(256), 1, 1),
        (256, 1, 1),
        [&mut src_pp, &mut dst_pp, &n_total, &dtype_id]
    );

    // Phase 2: hipblasGemmEx with both inputs half + f32 output.
    let blas = blas.lock().unwrap();
    let half_buf_ptr = match arena.half_buffer.as_ref() {
        Some(b) => b.ptr,
        None => return false,
    };
    let weight_dev = half_buf_ptr + (half_off as u64) * 2; // u16 = 2 bytes
    let act_dev = scratch_ptr;
    let c_dev = arena_base + (c_off_f32 as u64) * 4;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let cuda_dt = match half_dtype {
        HalfDtype::F16 => HipblasDatatype::R16F,
        HalfDtype::Bf16 => HipblasDatatype::R16BF,
    };
    let compute_ty = match half_dtype {
        HalfDtype::F16 => HipblasComputeType::F32Fast16F,
        HalfDtype::Bf16 => HipblasComputeType::F32Fast16BF,
    };
    let result = unsafe {
        (blas.runtime.gemm_ex)(
            blas.handle,
            HipblasOperation::N,
            HipblasOperation::N,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const f32 as *const _,
            weight_dev as *const _,
            cuda_dt,
            n as i32,
            act_dev as *const _,
            cuda_dt,
            k as i32,
            &beta as *const f32 as *const _,
            c_dev as *mut _,
            HipblasDatatype::R32F,
            n as i32,
            compute_ty,
            hipblas_gemm_default(),
        )
    };
    if let Err(e) = result.ok() {
        log_fallback("matmul.hipblasGemmEx (mixed)", e);
        return false;
    }
    true
}

pub(crate) fn im2col_use_gpu(n: u32, exec_mode: ExecMode) -> bool {
    if rlx_ir::env::var("RLX_ROCM_IM2COL_HOST").is_some() {
        return false;
    }
    if matches!(exec_mode, ExecMode::Graph) {
        return n > 0;
    }
    n > 0
}

pub(crate) fn pinned_io_enabled(exec_mode: ExecMode) -> bool {
    if matches!(exec_mode, ExecMode::Graph) {
        return true;
    }
    rlx_ir::env::var("RLX_ROCM_PINNED_IO").is_some_and(|v| !v.eq_ignore_ascii_case("0"))
}
