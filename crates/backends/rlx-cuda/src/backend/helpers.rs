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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublas::{CudaBlas, sys as cublas_sys};
use cudarc::cublaslt::{result as cublaslt_result, sys as cublaslt_sys};
use cudarc::cudnn::{result as cudnn_result, sys as cudnn_sys};
use cudarc::driver::{CudaContext, DevicePtrMut, LaunchConfig, PushKernelArg};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};

use crate::device::{
    CUBLASLT_WORKSPACE_BYTES, CUDNN_WORKSPACE_BYTES, cuda_blas, cuda_blas_lt_handle,
    cuda_blas_lt_workspace, cuda_context, cuda_dnn_handle, cuda_dnn_workspace,
};
use crate::kernels::{
    ada_layer_norm_backward_kernel, ada_layer_norm_kernel, argmax_kernel, attention_bwd_kernel,
    attention_kernel, attention_row_kernel, batch_elementwise_region_kernel, binary_kernel,
    compare_kernel, concat_kernel, conv_transpose2d_kernel, conv1d_kernel, conv2d_kernel,
    conv3d_kernel, copy_kernel, cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel,
    dispatch_grid_1d, dispatch_grid_prologue_nchw, elementwise_region_kernel, expand_kernel,
    fused_attn_kernel, fused_binary_unary_kernel, fused_residual_ln_kernel,
    fused_residual_rms_norm_kernel, gated_delta_net_kernel, gated_residual_backward_kernel,
    gated_residual_kernel, gather_axis_kernel, gather_backward_kernel, gather_kernel,
    group_norm_kernel, grouped_matmul_kernel, im2col_kernel, layer_norm2d_kernel, layernorm_kernel,
    matmul_epilogue_kernel, matmul_kernel, matmul_wmma_kernel, maxpool2d_backward_kernel,
    narrow_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel, reduce_kernel,
    resize_nearest_2x_kernel, rms_norm_backward_kernel, rms_norm_bwd_zero_kernel,
    rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_acc_kernel,
    scatter_add_zero_kernel, selective_scan_kernel, softmax_kernel, topk_kernel, transpose_kernel,
    unary_kernel, where_kernel,
};

use super::{CompileMode, ExecMode, Step};

/// Opt-in WMMA Tensor Core matmul. Reads `RLX_CUDA_WMMA=1` from env at
/// process start (cached behind a `OnceLock`). When true and cuBLAS is
/// unavailable, the scalar matmul kernel is replaced by the WMMA kernel
/// for plain (non-fused) matmul. Tensor Cores require SM 70+; on older
/// hardware NVRTC's `load_module` will fail and we fall back to scalar.
pub(crate) fn use_wmma() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        rlx_ir::env::var("RLX_CUDA_WMMA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Strict f32 matmul for encoder parity: tiled `matmul.cu` kernel (same
/// family as wgpu), not cuBLASLt / cuBLAS heuristics.
pub(crate) fn matmul_parity_mode() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        rlx_ir::env::flag("RLX_CUDA_NO_TF32")
            || rlx_ir::env::flag("RLX_CUDA_PARITY")
            || rlx_ir::env::flag("RLX_CUDA_NO_CUBLASLT")
    })
}

/// cuDNN conv math type. FMA (strict FP32) by default. Unlike the matmul path
/// (where `CUBLAS_COMPUTE_32F_FAST_TF32` is the stable default), TF32
/// tensor-core convs destabilize large-batch Adam training here — a loss
/// blow-up reproduces at batch ≥ 1024, and the forward pass alone is enough to
/// trigger it — so TF32 conv is opt-in: `RLX_CUDA_CONV_TF32=1` enables it
/// (≈1.4× on conv, safe for inference / forward-only or verified-stable runs).
/// `RLX_CUDA_NO_TF32` / `RLX_CUDA_PARITY` always force FMA.
pub(crate) fn conv_math_type() -> cudnn_sys::cudnnMathType_t {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    let tf32 = *ON.get_or_init(|| {
        rlx_ir::env::flag("RLX_CUDA_CONV_TF32")
            && !(rlx_ir::env::flag("RLX_CUDA_NO_TF32") || rlx_ir::env::flag("RLX_CUDA_PARITY"))
    });
    if tf32 {
        cudnn_sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH
    } else {
        cudnn_sys::cudnnMathType_t::CUDNN_FMA_MATH
    }
}

pub(crate) fn schedule_needs_blas_lt(schedule: &[Step]) -> bool {
    schedule.iter().any(|s| {
        matches!(
            s,
            Step::Matmul { act_id, .. } if cublaslt_act_supported(*act_id)
        )
    })
}

pub(crate) fn schedule_needs_dnn(schedule: &[Step]) -> bool {
    schedule.iter().any(|s| {
        matches!(
            s,
            Step::Conv1d { .. } | Step::Conv2d { .. } | Step::Conv3d { .. }
        )
    })
}

/// Map our internal activation id (matches the `unary` kernel table)
/// to a cuBLASLt epilogue activation, if it's natively fusable.
/// cuBLASLt only supports Relu and Gelu in the epilogue — anything else
/// (sigmoid, tanh, silu, abs, neg, sqrt) returns None and the caller
/// falls back to plain sgemm + the matmul_epilogue kernel.
pub(crate) fn cublaslt_act_for(act_id: u32) -> Option<cublaslt_sys::cublasLtEpilogue_t> {
    None.or(match act_id {
        // Identity
        0xFFFFu32 => Some(None),
        // Relu = 0; Gelu = 9; GeluApprox = 11 (treat as Gelu).
        0 => Some(Some(
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
        )),
        9 | 11 => Some(Some(
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
        )),
        _ => Some(None),
    })
    .flatten()
}

/// True when `act_id` is fusable in cuBLASLt's epilogue (or absent).
pub(crate) fn cublaslt_act_supported(act_id: u32) -> bool {
    matches!(act_id, 0xFFFFu32 | 0 | 9 | 11)
}

/// Single cuBLASLt fused matmul. Consumes one descriptor + three matrix
/// layouts + one preference object per call (descriptors are cheap to
/// create; future optimization could cache them by shape). Returns
/// `Err` on any setup failure so the caller can fall back to plain
/// cuBLAS sgemm + epilogue kernel.
pub(crate) unsafe fn cublaslt_matmul_fused(
    handle: cublaslt_sys::cublasLtHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    m: u32,
    k: u32,
    n: u32,
    a_off_f32: u32,
    b_off_f32: u32,
    c_off_f32: u32,
    has_bias: bool,
    bias_off_f32: u32,
    epilogue_act: Option<cublaslt_sys::cublasLtEpilogue_t>,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    cu_stream: cudarc::driver::sys::CUstream,
) -> Result<(), cublaslt_result::CublasError> {
    use core::ffi::c_void;
    use core::mem;

    // cuBLASLt is column-major. We swap A↔B so that "computing C^T =
    // B^T·A^T in column-major" matches "C = A·B in row-major".
    let a_ptr = (arena_dev_ptr + (b_off_f32 as u64) * 4) as *const c_void; // = our B
    let b_ptr = (arena_dev_ptr + (a_off_f32 as u64) * 4) as *const c_void; // = our A
    let c_ptr = (arena_dev_ptr + (c_off_f32 as u64) * 4) as *const c_void;
    let d_ptr = c_ptr as *mut c_void;

    let dt = cublaslt_sys::cudaDataType_t::CUDA_R_32F;

    // Layouts. After A↔B swap: cuBLASLt sees a [n,k] · [k,m] = [n,m].
    let a_layout = cublaslt_result::create_matrix_layout(dt, n as u64, k as u64, n as i64)?;
    let b_layout = cublaslt_result::create_matrix_layout(dt, k as u64, m as u64, k as i64)?;
    let c_layout = cublaslt_result::create_matrix_layout(dt, n as u64, m as u64, n as i64)?;

    if batch > 1 {
        unsafe {
            let bsz = batch as i32;
            for &layout in &[a_layout, b_layout, c_layout] {
                cublaslt_result::set_matrix_layout_attribute(
                layout,
                cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &bsz as *const _ as *const _,
                mem::size_of::<i32>(),
            )?;
            }
            let stride_b = b_batch_stride as i64;
            let stride_a = a_batch_stride as i64;
            let stride_c = c_batch_stride as i64;
            cublaslt_result::set_matrix_layout_attribute(
            a_layout,
            cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_b as *const _ as *const _, mem::size_of::<i64>())?;
            cublaslt_result::set_matrix_layout_attribute(
            b_layout,
            cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_a as *const _ as *const _, mem::size_of::<i64>())?;
            cublaslt_result::set_matrix_layout_attribute(
            c_layout,
            cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_c as *const _ as *const _, mem::size_of::<i64>())?;
        }
    }

    // CUBLAS_COMPUTE_32F_FAST_TF32 enables Tensor-Core paths on Ampere+.
    // Set RLX_CUDA_NO_TF32=1 (or RLX_CUDA_PARITY=1) for strict f32 parity
    // vs CPU / wgpu reference paths.
    let compute_type =
        if rlx_ir::env::flag("RLX_CUDA_NO_TF32") || rlx_ir::env::flag("RLX_CUDA_PARITY") {
            cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        } else {
            cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        };
    let matmul_desc = cublaslt_result::create_matmul_desc(compute_type, dt)?;

    // Pick the epilogue mode. cuBLASLt fuses bias broadcast over the
    // M dimension (in cuBLASLt's view). With our A↔B swap, cuBLASLt's
    // M = our row-major N, so a bias[N] vector broadcasts across M
    // rows of row-major C — exactly what we want.
    let epilogue = match (has_bias, epilogue_act) {
        (true, Some(cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU)) => {
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS
        }
        (true, Some(cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU)) => {
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS
        }
        (true, None) => cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        (false, Some(act)) => act,
        (false, None) => cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        _ => cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
    };
    unsafe {
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epilogue as *const _ as *const _,
            mem::size_of::<cublaslt_sys::cublasLtEpilogue_t>(),
        )?;
    }

    if has_bias {
        let bias_dev_ptr = arena_dev_ptr + (bias_off_f32 as u64) * 4;
        unsafe {
            cublaslt_result::set_matmul_desc_attribute(
                matmul_desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_dev_ptr as *const _ as *const _,
                mem::size_of::<u64>(),
            )?;
        }
    }

    let matmul_pref = cublaslt_result::create_matmul_pref()?;
    unsafe {
        cublaslt_result::set_matmul_pref_attribute(
            matmul_pref,
            cublaslt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size as *const _ as *const _,
            mem::size_of::<usize>(),
        )?;
    }

    let heuristic = unsafe {
        cublaslt_result::get_matmul_algo_heuristic(
            handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            matmul_pref,
        )
    }?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let workspace_ptr = workspace_dev_ptr as *mut c_void;

    let result = unsafe {
        cublaslt_result::matmul(
            handle,
            matmul_desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            a_ptr,
            a_layout,
            b_ptr,
            b_layout,
            c_ptr,
            c_layout,
            d_ptr,
            c_layout,
            &heuristic.algo as *const _,
            workspace_ptr,
            workspace_size,
            cu_stream as cublaslt_sys::cudaStream_t,
        )
    };

    // Always destroy descriptors (success or fail).
    unsafe {
        let _ = cublaslt_result::destroy_matmul_pref(matmul_pref);
        let _ = cublaslt_result::destroy_matmul_desc(matmul_desc);
        let _ = cublaslt_result::destroy_matrix_layout(c_layout);
        let _ = cublaslt_result::destroy_matrix_layout(b_layout);
        let _ = cublaslt_result::destroy_matrix_layout(a_layout);
    }

    result
}

/// Native **FP8 tensor-core GEMM** via cuBLASLt (Hopper/Ada sm_89+).
/// Computes row-major `D[m,n] = (lhs[m,k] · rhs[n,k]ᵀ) · lhs_scale · rhs_scale`
/// where `lhs`/`rhs` are FP8 (E4M3/E5M2) codes and the scales are device f32
/// scalars. This is RLX's `Op::ScaledMatMul` (TN layout) — the operands are fed
/// straight into the tensor cores with f32 accumulation, the real low-precision
/// throughput win that the decode-then-sgemm storage path leaves on the table.
///
/// Mapping to cuBLASLt's column-major `D = op(A)·op(B)`: we compute the
/// transpose `Dᵀ[n,m]` in column-major (= our row-major `D[m,n]`) with
///   A = rhs  (col-major `[k,n]`, op = **T**)   — FP8 requires transa=T
///   B = lhs  (col-major `[k,m]`, op = **N**)   — FP8 requires transb=N
/// so A↔scale: A_SCALE = rhs_scale, B_SCALE = lhs_scale. Offsets are **bytes**
/// (FP8 codes are 1 byte; scales/out/bias are f32).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cublaslt_matmul_fp8(
    handle: cublaslt_sys::cublasLtHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    m: u32,
    k: u32,
    n: u32,
    lhs_byte_off: u64,
    rhs_byte_off: u64,
    lhs_scale_byte_off: u64,
    rhs_scale_byte_off: u64,
    out_byte_off: u64,
    has_bias: bool,
    bias_byte_off: u64,
    lhs_e5m2: bool,
    rhs_e5m2: bool,
    cu_stream: cudarc::driver::sys::CUstream,
) -> Result<(), cublaslt_result::CublasError> {
    use core::ffi::c_void;
    use core::mem;

    let fp8 = |e5m2: bool| {
        if e5m2 {
            cublaslt_sys::cudaDataType_t::CUDA_R_8F_E5M2
        } else {
            cublaslt_sys::cudaDataType_t::CUDA_R_8F_E4M3
        }
    };
    let a_dt = fp8(rhs_e5m2); // A = rhs
    let b_dt = fp8(lhs_e5m2); // B = lhs
    let out_dt = cublaslt_sys::cudaDataType_t::CUDA_R_32F;

    let a_ptr = (arena_dev_ptr + rhs_byte_off) as *const c_void;
    let b_ptr = (arena_dev_ptr + lhs_byte_off) as *const c_void;
    let c_ptr = (arena_dev_ptr + out_byte_off) as *const c_void;
    let d_ptr = c_ptr as *mut c_void;

    // A = rhs col-major [k,n] ld=k; B = lhs col-major [k,m] ld=k;
    // D = col-major [n,m] ld=n  (== row-major [m,n]).
    let a_layout = cublaslt_result::create_matrix_layout(a_dt, k as u64, n as u64, k as i64)?;
    let b_layout = cublaslt_result::create_matrix_layout(b_dt, k as u64, m as u64, k as i64)?;
    let cd_layout = cublaslt_result::create_matrix_layout(out_dt, n as u64, m as u64, n as i64)?;

    // FP8 accumulation is f32; scale type (alpha/beta) f32.
    let compute_type = cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
    let matmul_desc = cublaslt_result::create_matmul_desc(
        compute_type,
        cublaslt_sys::cudaDataType_t::CUDA_R_32F,
    )?;

    // cuBLASLt FP8 requires transa = T, transb = N (cublasOperation_t as i32).
    let op_t: i32 = 1; // CUBLAS_OP_T
    let op_n: i32 = 0; // CUBLAS_OP_N
    unsafe {
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &op_t as *const i32 as *const _,
            mem::size_of::<i32>(),
        )?;
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &op_n as *const i32 as *const _,
            mem::size_of::<i32>(),
        )?;

        // Per-tensor dequant scales: D = a_scale · b_scale · (A·B).
        let a_scale_ptr = arena_dev_ptr + rhs_scale_byte_off;
        let b_scale_ptr = arena_dev_ptr + lhs_scale_byte_off;
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale_ptr as *const u64 as *const _,
            mem::size_of::<u64>(),
        )?;
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale_ptr as *const u64 as *const _,
            mem::size_of::<u64>(),
        )?;

        if has_bias {
            let epi = cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            cublaslt_result::set_matmul_desc_attribute(
                matmul_desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epi as *const _ as *const _,
                mem::size_of::<cublaslt_sys::cublasLtEpilogue_t>(),
            )?;
            let bias_ptr = arena_dev_ptr + bias_byte_off;
            cublaslt_result::set_matmul_desc_attribute(
                matmul_desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_ptr as *const u64 as *const _,
                mem::size_of::<u64>(),
            )?;
        }
    }

    let matmul_pref = cublaslt_result::create_matmul_pref()?;
    unsafe {
        cublaslt_result::set_matmul_pref_attribute(
            matmul_pref,
            cublaslt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size as *const _ as *const _,
            mem::size_of::<usize>(),
        )?;
    }

    let heuristic = unsafe {
        cublaslt_result::get_matmul_algo_heuristic(
            handle,
            matmul_desc,
            a_layout,
            b_layout,
            cd_layout,
            cd_layout,
            matmul_pref,
        )
    }?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let result = unsafe {
        cublaslt_result::matmul(
            handle,
            matmul_desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            a_ptr,
            a_layout,
            b_ptr,
            b_layout,
            c_ptr,
            cd_layout,
            d_ptr,
            cd_layout,
            &heuristic.algo as *const _,
            workspace_dev_ptr as *mut c_void,
            workspace_size,
            cu_stream as cublaslt_sys::cudaStream_t,
        )
    };

    unsafe {
        let _ = cublaslt_result::destroy_matmul_pref(matmul_pref);
        let _ = cublaslt_result::destroy_matmul_desc(matmul_desc);
        let _ = cublaslt_result::destroy_matrix_layout(cd_layout);
        let _ = cublaslt_result::destroy_matrix_layout(b_layout);
        let _ = cublaslt_result::destroy_matrix_layout(a_layout);
    }
    result
}

/// Conv algos to request from the cuDNN v7 heuristic. It returns them
/// ranked fastest-first; we scan for the fastest whose workspace fits the
/// budget. Taking only the top-1 and rejecting it when its workspace
/// overflows would fall back to the ~10× slower im2col path at large batch
/// (the top algo's workspace grows with N), which is exactly the cliff this
/// avoids — `IMPLICIT_GEMM` needs ~0 workspace, so a fitting algo always exists.
const CUDNN_ALGO_CANDIDATES: usize = 8;

/// Fastest forward-conv algo whose workspace fits `workspace_size`.
pub(crate) unsafe fn pick_conv_fwd_algo(
    handle: cudnn_sys::cudnnHandle_t,
    x_desc: cudnn_sys::cudnnTensorDescriptor_t,
    w_desc: cudnn_sys::cudnnFilterDescriptor_t,
    conv_desc: cudnn_sys::cudnnConvolutionDescriptor_t,
    y_desc: cudnn_sys::cudnnTensorDescriptor_t,
    workspace_size: usize,
) -> Result<cudnn_sys::cudnnConvolutionFwdAlgo_t, cudnn_result::CudnnError> {
    let mut returned: i32 = 0;
    let mut perfs = std::mem::MaybeUninit::<
        [cudnn_sys::cudnnConvolutionFwdAlgoPerf_t; CUDNN_ALGO_CANDIDATES],
    >::uninit();
    unsafe {
        cudnn_result::get_convolution_forward_algorithm(
            handle,
            x_desc,
            w_desc,
            conv_desc,
            y_desc,
            CUDNN_ALGO_CANDIDATES as i32,
            &mut returned,
            perfs.as_mut_ptr() as *mut cudnn_sys::cudnnConvolutionFwdAlgoPerf_t,
        )?;
        let base = perfs.as_ptr() as *const cudnn_sys::cudnnConvolutionFwdAlgoPerf_t;
        for i in 0..(returned.max(0) as usize).min(CUDNN_ALGO_CANDIDATES) {
            let p = base.add(i).read();
            if matches!(p.status, cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
                && p.memory <= workspace_size
            {
                return Ok(p.algo);
            }
        }
    }
    Err(cudnn_result::CudnnError(
        cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
    ))
}

/// Fastest backward-data conv algo whose workspace fits `workspace_size`.
/// Prefer deterministic cuDNN backward-conv algorithms (default on). cuDNN's
/// fastest backward-data/-filter algos accumulate partial sums via `atomicAdd`
/// and are NON-deterministic run-to-run. That per-step gradient noise
/// intermittently collapses training to chance (loss pinned at ln k) —
/// especially with folded/frozen BatchNorm, whose fixed scale/shift amplify
/// gradient noise that adaptive norms tolerate — while CPU/MLX and rlx's own
/// im2col conv path are bit-exact. Empirically: EEGNet fold-BN training on CUDA
/// collapsed ~5/8 runs with cuDNN, 0/8 (bit-identical) without. Set
/// `RLX_CUDA_NONDET_CONV=1` to opt back into the raw-fastest (possibly
/// non-deterministic) algo.
pub(crate) fn deterministic_conv() -> bool {
    !rlx_ir::env::flag("RLX_CUDA_NONDET_CONV")
}

pub(crate) unsafe fn pick_conv_bwd_data_algo(
    handle: cudnn_sys::cudnnHandle_t,
    w_desc: cudnn_sys::cudnnFilterDescriptor_t,
    dy_desc: cudnn_sys::cudnnTensorDescriptor_t,
    conv_desc: cudnn_sys::cudnnConvolutionDescriptor_t,
    dx_desc: cudnn_sys::cudnnTensorDescriptor_t,
    workspace_size: usize,
) -> Result<cudnn_sys::cudnnConvolutionBwdDataAlgo_t, cudnn_result::CudnnError> {
    let mut returned: i32 = 0;
    let mut perfs = std::mem::MaybeUninit::<
        [cudnn_sys::cudnnConvolutionBwdDataAlgoPerf_t; CUDNN_ALGO_CANDIDATES],
    >::uninit();
    unsafe {
        cudnn_result::get_convolution_backward_data_algorithm(
            handle,
            w_desc,
            dy_desc,
            conv_desc,
            dx_desc,
            CUDNN_ALGO_CANDIDATES as i32,
            &mut returned,
            perfs.as_mut_ptr() as *mut cudnn_sys::cudnnConvolutionBwdDataAlgoPerf_t,
        )?;
        let base = perfs.as_ptr() as *const cudnn_sys::cudnnConvolutionBwdDataAlgoPerf_t;
        let n = (returned.max(0) as usize).min(CUDNN_ALGO_CANDIDATES);
        // Pass 1 (default): fastest DETERMINISTIC algo that fits the workspace.
        if deterministic_conv() {
            for i in 0..n {
                let p = base.add(i).read();
                if matches!(p.status, cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
                    && p.memory <= workspace_size
                    && matches!(
                        p.determinism,
                        cudnn_sys::cudnnDeterminism_t::CUDNN_DETERMINISTIC
                    )
                {
                    return Ok(p.algo);
                }
            }
        }
        // Pass 2: fastest fitting algo regardless of determinism (fallback when no
        // deterministic algo fits, or when RLX_CUDA_NONDET_CONV opts into speed).
        for i in 0..n {
            let p = base.add(i).read();
            if matches!(p.status, cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
                && p.memory <= workspace_size
            {
                return Ok(p.algo);
            }
        }
    }
    Err(cudnn_result::CudnnError(
        cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
    ))
}

/// Fastest backward-filter conv algo whose workspace fits `workspace_size`.
pub(crate) unsafe fn pick_conv_bwd_filter_algo(
    handle: cudnn_sys::cudnnHandle_t,
    x_desc: cudnn_sys::cudnnTensorDescriptor_t,
    dy_desc: cudnn_sys::cudnnTensorDescriptor_t,
    conv_desc: cudnn_sys::cudnnConvolutionDescriptor_t,
    dw_desc: cudnn_sys::cudnnFilterDescriptor_t,
    workspace_size: usize,
) -> Result<cudnn_sys::cudnnConvolutionBwdFilterAlgo_t, cudnn_result::CudnnError> {
    let mut returned: i32 = 0;
    let mut perfs = std::mem::MaybeUninit::<
        [cudnn_sys::cudnnConvolutionBwdFilterAlgoPerf_t; CUDNN_ALGO_CANDIDATES],
    >::uninit();
    unsafe {
        cudnn_result::get_convolution_backward_filter_algorithm(
            handle,
            x_desc,
            dy_desc,
            conv_desc,
            dw_desc,
            CUDNN_ALGO_CANDIDATES as i32,
            &mut returned,
            perfs.as_mut_ptr() as *mut cudnn_sys::cudnnConvolutionBwdFilterAlgoPerf_t,
        )?;
        let base = perfs.as_ptr() as *const cudnn_sys::cudnnConvolutionBwdFilterAlgoPerf_t;
        let n = (returned.max(0) as usize).min(CUDNN_ALGO_CANDIDATES);
        // Pass 1 (default): fastest DETERMINISTIC algo that fits. The backward-
        // FILTER `ALGO_0`/`ALGO_3` accumulate via atomics and are the main source
        // of the training-collapse nondeterminism — see `deterministic_conv`.
        if deterministic_conv() {
            for i in 0..n {
                let p = base.add(i).read();
                if matches!(p.status, cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
                    && p.memory <= workspace_size
                    && matches!(
                        p.determinism,
                        cudnn_sys::cudnnDeterminism_t::CUDNN_DETERMINISTIC
                    )
                {
                    return Ok(p.algo);
                }
            }
        }
        // Pass 2: fastest fitting algo regardless of determinism (fallback).
        for i in 0..n {
            let p = base.add(i).read();
            if matches!(p.status, cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
                && p.memory <= workspace_size
            {
                return Ok(p.algo);
            }
        }
    }
    Err(cudnn_result::CudnnError(
        cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
    ))
}

/// cuDNN forward 2D convolution against arena offsets. NCHW input,
/// KCRS filter, NCHW output. Uses the v7 algorithm heuristic to pick
/// the fastest algo that fits in the supplied workspace. Returns
/// `Err` on any setup failure so the caller can fall back to the
/// direct-convolution kernel.
pub(crate) unsafe fn cudnn_conv2d_forward(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    in_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;

    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

    let x_desc = cudnn_result::create_tensor_descriptor()?;
    let y_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;

    let w_desc = unsafe {
        let mut w_desc_uninit = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(w_desc_uninit.as_mut_ptr()).result()?;
        w_desc_uninit.assume_init()
    };

    let setup = unsafe {
        cudnn_result::set_tensor4d_descriptor(
            x_desc,
            fmt,
            dt,
            [n as i32, c_in as i32, h as i32, w as i32],
        )?;
        cudnn_result::set_tensor4d_descriptor(
            y_desc,
            fmt,
            dt,
            [n as i32, c_out as i32, h_out as i32, w_out as i32],
        )?;
        cudnn_result::set_filter4d_descriptor(
            w_desc,
            dt,
            fmt,
            [
                c_out as i32,
                (c_in / groups.max(1)) as i32,
                kh as i32,
                kw as i32,
            ],
        )?;
        cudnn_result::set_convolution2d_descriptor(
            conv_desc,
            ph as i32,
            pw as i32,
            sh as i32,
            sw as i32,
            dh as i32,
            dw as i32,
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        // FMA by default; TF32 tensor cores only under RLX_CUDA_CONV_TF32 (they
        // destabilize large-batch training — see `conv_math_type`).
        cudnn_sys::cudnnSetConvolutionMathType(conv_desc, conv_math_type()).result()?;
        Ok::<(), cudnn_result::CudnnError>(())
    };

    let result = setup.and_then(|()| unsafe {
        // Fastest fwd algo whose workspace fits (never bails to im2col when a
        // fitting algo — e.g. IMPLICIT_GEMM — exists).
        let algo = pick_conv_fwd_algo(handle, x_desc, w_desc, conv_desc, y_desc, workspace_size)?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let x_ptr = (arena_dev_ptr + (in_off_f32 as u64) * 4) as *const c_void;
        let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
        let y_ptr = (arena_dev_ptr + (out_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;

        cudnn_result::convolution_forward(
            handle,
            &alpha as *const _ as *const c_void,
            x_desc,
            x_ptr,
            w_desc,
            w_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            y_desc,
            y_ptr,
        )
    });

    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(w_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(y_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(x_desc);
    }

    result
}

/// Fused cuDNN forward conv + bias + activation
/// (`cudnnConvolutionBiasActivationForward`): `y = act(conv(x, w) + bias)`.
///
/// `bias` is a rank-1 `[C_out]` vector described as `[1, C_out, 1, 1]` (channel
/// broadcast). Uses `IMPLICIT_PRECOMP_GEMM`, the algorithm cuDNN documents as
/// compatible with the fused bias-activation call (and the only one accepted
/// for `CUDNN_ACTIVATION_IDENTITY`, i.e. bias-only).
///
/// cuDNN's fused call ONLY supports IDENTITY and RELU epilogues — sigmoid/tanh
/// and every other activation return `CUDNN_STATUS_NOT_SUPPORTED`, so `act_id`
/// here is expected to be 0 (relu) or 0xFFFF (identity); callers gate on that
/// (`cudnn_epilogue_ok`) and route the rest through the `conv_bias_act_epilogue`
/// kernel. Any other id returns Err (defensive). On any cuDNN error the caller
/// falls back to the direct-conv kernel + epilogue, so this never has to
/// succeed.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cudnn_conv2d_bias_act_forward(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    in_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
    bias_off_f32: u32,
    act_id: u32,
    residual_off_f32: u32,
    has_residual: bool,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;

    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

    // rlx activation id → cuDNN mode. cuDNN's fused conv-bias-activation ONLY
    // supports IDENTITY and RELU; anything else bails so the caller's epilogue
    // applies the correct activation (callers already gate on this).
    let act_mode = match act_id {
        0xFFFFu32 => cudnn_sys::cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
        0 => cudnn_sys::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
        _ => {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
    };

    let x_desc = cudnn_result::create_tensor_descriptor()?;
    let y_desc = cudnn_result::create_tensor_descriptor()?;
    let bias_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;
    let act_desc = cudnn_result::create_activation_descriptor()?;

    let w_desc = unsafe {
        let mut w_desc_uninit = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(w_desc_uninit.as_mut_ptr()).result()?;
        w_desc_uninit.assume_init()
    };

    let setup = unsafe {
        cudnn_result::set_tensor4d_descriptor(
            x_desc,
            fmt,
            dt,
            [n as i32, c_in as i32, h as i32, w as i32],
        )?;
        cudnn_result::set_tensor4d_descriptor(
            y_desc,
            fmt,
            dt,
            [n as i32, c_out as i32, h_out as i32, w_out as i32],
        )?;
        // Bias broadcasts over N/H/W → [1, C_out, 1, 1].
        cudnn_result::set_tensor4d_descriptor(bias_desc, fmt, dt, [1, c_out as i32, 1, 1])?;
        cudnn_result::set_filter4d_descriptor(
            w_desc,
            dt,
            fmt,
            [
                c_out as i32,
                (c_in / groups.max(1)) as i32,
                kh as i32,
                kw as i32,
            ],
        )?;
        cudnn_result::set_convolution2d_descriptor(
            conv_desc,
            ph as i32,
            pw as i32,
            sh as i32,
            sw as i32,
            dh as i32,
            dw as i32,
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        cudnn_sys::cudnnSetConvolutionMathType(conv_desc, conv_math_type()).result()?;
        cudnn_result::set_activation_descriptor(
            act_desc,
            act_mode,
            cudnn_sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            0.0,
        )?;
        Ok::<(), cudnn_result::CudnnError>(())
    };

    let result = setup.and_then(|()| unsafe {
        // IMPLICIT_PRECOMP_GEMM is the fused-call-compatible algo (and the only
        // one accepted with an IDENTITY activation). Insufficient workspace →
        // Err → caller falls back to the direct kernel + epilogue.
        let algo =
            cudnn_sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        let alpha1: f32 = 1.0;
        // ResNet residual: y = act(conv + bias + alpha2·z). With a residual,
        // alpha2 = 1 and z points at it; otherwise alpha2 = 0 and z = y (unread,
        // zDesc must still be a valid = yDesc descriptor).
        let alpha2: f32 = if has_residual { 1.0 } else { 0.0 };
        let x_ptr = (arena_dev_ptr + (in_off_f32 as u64) * 4) as *const c_void;
        let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
        let y_ptr = (arena_dev_ptr + (out_off_f32 as u64) * 4) as *mut c_void;
        let bias_ptr = (arena_dev_ptr + (bias_off_f32 as u64) * 4) as *const c_void;
        let z_ptr = if has_residual {
            (arena_dev_ptr + (residual_off_f32 as u64) * 4) as *const c_void
        } else {
            y_ptr as *const c_void
        };
        let workspace_ptr = workspace_dev_ptr as *mut c_void;

        cudnn_sys::cudnnConvolutionBiasActivationForward(
            handle,
            &alpha1 as *const _ as *const c_void,
            x_desc,
            x_ptr,
            w_desc,
            w_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &alpha2 as *const _ as *const c_void,
            y_desc,
            z_ptr,
            bias_desc,
            bias_ptr,
            act_desc,
            y_desc,
            y_ptr,
        )
        .result()
    });

    unsafe {
        let _ = cudnn_sys::cudnnDestroyActivationDescriptor(act_desc);
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(w_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(bias_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(y_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(x_desc);
    }

    result
}

/// cuDNN backward-data 2-D convolution: dx (input grad) from dy and w.
/// Mirrors `cudnn_conv2d_forward`; returns Err so the caller can fall back
/// to the host reference.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cudnn_conv2d_backward_data(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    dy_off_f32: u32,
    w_off_f32: u32,
    dx_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;
    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
    let dx_desc = cudnn_result::create_tensor_descriptor()?;
    let dy_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;
    let w_desc = unsafe {
        let mut u = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(u.as_mut_ptr()).result()?;
        u.assume_init()
    };
    let setup = unsafe {
        cudnn_result::set_tensor4d_descriptor(
            dx_desc,
            fmt,
            dt,
            [n as i32, c_in as i32, h as i32, w as i32],
        )?;
        cudnn_result::set_tensor4d_descriptor(
            dy_desc,
            fmt,
            dt,
            [n as i32, c_out as i32, h_out as i32, w_out as i32],
        )?;
        cudnn_result::set_filter4d_descriptor(
            w_desc,
            dt,
            fmt,
            [
                c_out as i32,
                (c_in / groups.max(1)) as i32,
                kh as i32,
                kw as i32,
            ],
        )?;
        cudnn_result::set_convolution2d_descriptor(
            conv_desc,
            ph as i32,
            pw as i32,
            sh as i32,
            sw as i32,
            dh as i32,
            dw as i32,
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        // FMA by default; TF32 tensor cores only under RLX_CUDA_CONV_TF32 (they
        // destabilize large-batch training — see `conv_math_type`).
        cudnn_sys::cudnnSetConvolutionMathType(conv_desc, conv_math_type()).result()?;
        Ok::<(), cudnn_result::CudnnError>(())
    };
    let result = setup.and_then(|()| unsafe {
        let algo =
            pick_conv_bwd_data_algo(handle, w_desc, dy_desc, conv_desc, dx_desc, workspace_size)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
        let dy_ptr = (arena_dev_ptr + (dy_off_f32 as u64) * 4) as *const c_void;
        let dx_ptr = (arena_dev_ptr + (dx_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;
        cudnn_result::convolution_backward_data(
            handle,
            &alpha as *const _ as *const c_void,
            w_desc,
            w_ptr,
            dy_desc,
            dy_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            dx_desc,
            dx_ptr,
        )
    });
    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(w_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(dy_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(dx_desc);
    }
    result
}

/// cuDNN backward-filter 2-D convolution: dw (weight grad) from x and dy.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cudnn_conv2d_backward_filter(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    x_off_f32: u32,
    dy_off_f32: u32,
    dw_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;
    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
    let x_desc = cudnn_result::create_tensor_descriptor()?;
    let dy_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;
    let dw_desc = unsafe {
        let mut u = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(u.as_mut_ptr()).result()?;
        u.assume_init()
    };
    let setup = unsafe {
        cudnn_result::set_tensor4d_descriptor(
            x_desc,
            fmt,
            dt,
            [n as i32, c_in as i32, h as i32, w as i32],
        )?;
        cudnn_result::set_tensor4d_descriptor(
            dy_desc,
            fmt,
            dt,
            [n as i32, c_out as i32, h_out as i32, w_out as i32],
        )?;
        cudnn_result::set_filter4d_descriptor(
            dw_desc,
            dt,
            fmt,
            [
                c_out as i32,
                (c_in / groups.max(1)) as i32,
                kh as i32,
                kw as i32,
            ],
        )?;
        cudnn_result::set_convolution2d_descriptor(
            conv_desc,
            ph as i32,
            pw as i32,
            sh as i32,
            sw as i32,
            dh as i32,
            dw as i32,
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        // FMA by default; TF32 tensor cores only under RLX_CUDA_CONV_TF32 (they
        // destabilize large-batch training — see `conv_math_type`).
        cudnn_sys::cudnnSetConvolutionMathType(conv_desc, conv_math_type()).result()?;
        Ok::<(), cudnn_result::CudnnError>(())
    };
    let result = setup.and_then(|()| unsafe {
        let algo =
            pick_conv_bwd_filter_algo(handle, x_desc, dy_desc, conv_desc, dw_desc, workspace_size)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let x_ptr = (arena_dev_ptr + (x_off_f32 as u64) * 4) as *const c_void;
        let dy_ptr = (arena_dev_ptr + (dy_off_f32 as u64) * 4) as *const c_void;
        let dw_ptr = (arena_dev_ptr + (dw_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;
        cudnn_result::convolution_backward_filter(
            handle,
            &alpha as *const _ as *const c_void,
            x_desc,
            x_ptr,
            dy_desc,
            dy_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            dw_desc,
            dw_ptr,
        )
    });
    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(dw_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(dy_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(x_desc);
    }
    result
}

/// cuDNN forward 3-D convolution. NCDHW input, KCDRS filter, NCDHW
/// output. Uses cuDNN's nd-descriptor APIs (set_tensornd / set_filternd
/// / set_convolutionnd) since the 4D versions only cover up to 2D conv.
pub(crate) unsafe fn cudnn_conv3d_forward(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    d: u32,
    h: u32,
    w: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    in_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;

    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

    let x_desc = cudnn_result::create_tensor_descriptor()?;
    let y_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;
    let w_desc = unsafe {
        let mut w_desc_uninit = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(w_desc_uninit.as_mut_ptr()).result()?;
        w_desc_uninit.assume_init()
    };

    // 5-D tensor: [N, C, D, H, W] with row-major contiguous strides.
    let x_dims: [i32; 5] = [n as i32, c_in as i32, d as i32, h as i32, w as i32];
    let x_strides: [i32; 5] = [
        (c_in * d * h * w) as i32,
        (d * h * w) as i32,
        (h * w) as i32,
        w as i32,
        1,
    ];
    let y_dims: [i32; 5] = [
        n as i32,
        c_out as i32,
        d_out as i32,
        h_out as i32,
        w_out as i32,
    ];
    let y_strides: [i32; 5] = [
        (c_out * d_out * h_out * w_out) as i32,
        (d_out * h_out * w_out) as i32,
        (h_out * w_out) as i32,
        w_out as i32,
        1,
    ];
    let f_dims: [i32; 5] = [
        c_out as i32,
        (c_in / groups.max(1)) as i32,
        kd as i32,
        kh as i32,
        kw as i32,
    ];
    let pads: [i32; 3] = [pd as i32, ph as i32, pw as i32];
    let strides: [i32; 3] = [sd as i32, sh as i32, sw as i32];
    let dilations: [i32; 3] = [dd as i32, dh as i32, dw as i32];

    let setup = unsafe {
        cudnn_result::set_tensornd_descriptor(x_desc, dt, 5, x_dims.as_ptr(), x_strides.as_ptr())?;
        cudnn_result::set_tensornd_descriptor(y_desc, dt, 5, y_dims.as_ptr(), y_strides.as_ptr())?;
        cudnn_result::set_filternd_descriptor(w_desc, dt, fmt, 5, f_dims.as_ptr())?;
        cudnn_result::set_convolutionnd_descriptor(
            conv_desc,
            3,
            pads.as_ptr(),
            strides.as_ptr(),
            dilations.as_ptr(),
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        // FMA by default; TF32 tensor cores only under RLX_CUDA_CONV_TF32 (they
        // destabilize large-batch training — see `conv_math_type`).
        cudnn_sys::cudnnSetConvolutionMathType(conv_desc, conv_math_type()).result()?;
        Ok::<(), cudnn_result::CudnnError>(())
    };

    let result = setup.and_then(|()| unsafe {
        let mut returned_count: i32 = 0;
        let mut perf = std::mem::MaybeUninit::<cudnn_sys::cudnnConvolutionFwdAlgoPerf_t>::uninit();
        cudnn_result::get_convolution_forward_algorithm(
            handle,
            x_desc,
            w_desc,
            conv_desc,
            y_desc,
            1,
            &mut returned_count,
            perf.as_mut_ptr(),
        )?;
        if returned_count == 0 {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
        let algo = perf.assume_init().algo;

        let needed = cudnn_result::get_convolution_forward_workspace_size(
            handle, x_desc, w_desc, conv_desc, y_desc, algo,
        )?;
        if needed > workspace_size {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let x_ptr = (arena_dev_ptr + (in_off_f32 as u64) * 4) as *const c_void;
        let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
        let y_ptr = (arena_dev_ptr + (out_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;

        cudnn_result::convolution_forward(
            handle,
            &alpha as *const _ as *const c_void,
            x_desc,
            x_ptr,
            w_desc,
            w_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            y_desc,
            y_ptr,
        )
    });

    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(w_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(y_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(x_desc);
    }

    result
}

/// Per-`Op::FusedAttentionBlock` scratch: packed QKV `[B,S,3*inner]` followed
/// by the attention output `[B,S,inner]`, both f32, 16-byte aligned per block.
/// Returns the total scratch size in BYTES and a map from each surviving FAB
/// node to its `(qkv, attn)` f32-element offsets *relative to the scratch
/// base*. Empty when the unfuse pass decomposed every FAB to primitives.
pub(crate) fn fab_scratch_plan(graph: &Graph) -> (usize, HashMap<rlx_ir::NodeId, (u32, u32)>) {
    let mut map = HashMap::new();
    let mut cur: usize = 0; // f32 elements
    for node in graph.nodes() {
        if let Op::FusedAttentionBlock {
            num_heads,
            head_dim,
            ..
        } = &node.op
        {
            let dims = node.shape.dims();
            let b = dims[0].unwrap_static();
            let s = dims[1].unwrap_static();
            let inner = num_heads * head_dim;
            let qkv_rel = cur as u32;
            cur += b * s * 3 * inner;
            let attn_rel = cur as u32;
            cur += b * s * inner;
            cur = (cur + 3) & !3; // 16-byte align the next block's region
            map.insert(node.id, (qkv_rel, attn_rel));
        }
    }
    (cur * 4, map)
}

/// Shared ephemeral state for `Op::GatedDeltaNet` with `carry_state=false`
/// (Metal mirrors this). Sequential GDN ops reuse one scratch region.
pub(crate) fn gdn_ephemeral_state_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::GatedDeltaNet {
            carry_state,
            state_size,
            ..
        } = &node.op
            && !*carry_state
        {
            let q_shape = &graph.node(node.inputs[0]).shape;
            let elems = q_shape.dim(0).unwrap_static()
                * q_shape.dim(2).unwrap_static()
                * state_size
                * state_size;
            max = max.max(elems * std::mem::size_of::<f32>());
        }
    }
    max
}

/// Decode a Matmul/FusedMatMulBiasAct node's input shapes into the
/// (m, k, n, batch, a_stride, b_stride, c_stride, a_id, b_id) tuple
/// the kernel expects. Four patterns:
///   • 2D × 2D                       → batch=1, all strides 0
///   • [..,M,K] × [K,N] (broadcast)  → batch=1, leading dims flattened into M
///   • [M,K] × [..,K,N] (broadcast)  → batch=prod(B leading), A stride 0
///   • [..,M,K] × [..,K,N] (matched) → batch=prod(leading), per-batch strides
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
    } else if a_shape.len() == 2 && b_shape.len() >= 3 && out_shape.len() == b_shape.len() {
        // [M,K] × [..,K,N] — broadcast A across B's leading dims (a_stride 0).
        let lead_b: usize = b_shape[..b_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .product();
        let lead_out: usize = out_shape[..out_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .product();
        let m_inner = a_shape[0].unwrap_static();
        let k_a = a_shape[1].unwrap_static();
        let k_b = b_shape[b_shape.len() - 2].unwrap_static();
        let n_inner = b_shape[b_shape.len() - 1].unwrap_static();
        let m_out = out_shape[out_shape.len() - 2].unwrap_static();
        let n_out = out_shape[out_shape.len() - 1].unwrap_static();
        if k_a != k_b || m_inner != m_out || n_inner != n_out || lead_b != lead_out {
            panic!(
                "rlx-cuda {op_label}: [M,K]×[..,K,N] shape mismatch \
                    a={a_shape:?} b={b_shape:?} out={out_shape:?}"
            );
        }
        (
            m_inner as u32,
            k_a as u32,
            n_inner as u32,
            lead_out as u32,
            0,
            (k_b * n_inner) as u32,
            (m_inner * n_inner) as u32,
            a_id,
            b_id,
        )
    } else if a_shape.len() == b_shape.len() && a_shape.len() >= 3 {
        // Leading (batch) PRODUCTS must match the output, OR be 1 and BROADCAST
        // across the batch (per-matrix stride 0 — reuse matrix 0).
        let lead_a: usize = a_shape[..a_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .product();
        let lead_b: usize = b_shape[..b_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .product();
        let lead_out: usize = out_shape[..out_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .product();
        let ok = (lead_a == lead_out || lead_a == 1)
            && (lead_b == lead_out || lead_b == 1)
            && lead_out == lead_a.max(lead_b);
        if !ok {
            panic!(
                "rlx-cuda {op_label}: batched shape mismatch \
                    a={a_shape:?} b={b_shape:?} out={out_shape:?}"
            );
        }
        let b_count: usize = lead_out;
        let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
        let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
        let n_inner = b_shape[b_shape.len() - 1].unwrap_static();
        let a_str = if lead_a == 1 && lead_out > 1 {
            0
        } else {
            m_inner * k_inner
        };
        let b_str = if lead_b == 1 && lead_out > 1 {
            0
        } else {
            k_inner * n_inner
        };
        (
            m_inner as u32,
            k_inner as u32,
            n_inner as u32,
            b_count as u32,
            a_str as u32,
            b_str as u32,
            (m_inner * n_inner) as u32,
            a_id,
            b_id,
        )
    } else {
        panic!(
            "rlx-cuda {op_label}: unsupported shapes a={a_shape:?} b={b_shape:?} out={out_shape:?}"
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

/// Op code for the `pool{1,2,3}d.cu` kernels, whose legend is `0=max, 1=mean,
/// 2=sum, 3=min, 4=prod` — this differs from [`reduce_op_id`] (which swaps Max
/// and Sum). Using `reduce_op_id` here made max-pooling compute the window sum.
pub(crate) fn pool_op_id(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Max => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Sum => 2,
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
    }
}

/// Mixed-precision matmul tier-0: when the weight (B input) is stored
/// in the half-arena, cast f32 activations to f16/bf16 in the scratch
/// buffer and run `cublasGemmEx` with both inputs half + f32
/// accumulator. Returns `true` on success.
///
/// Free function (rather than `&mut self` method) so the caller can
/// hold `&self.schedule` across the call without violating disjoint-
/// field borrow checks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_mixed_precision_gemm(
    ctx: &Arc<CudaContext>,
    arena: &mut crate::arena::Arena,
    half_act_scratch: &mut Option<cudarc::driver::CudaSlice<u16>>,
    blas: Option<&Arc<Mutex<CudaBlas>>>,
    stream: &Arc<cudarc::driver::CudaStream>,
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
    let need_resize = half_act_scratch
        .as_ref()
        .is_none_or(|s| s.len() < act_elems);
    if need_resize {
        *half_act_scratch = stream.alloc_zeros::<u16>(act_elems.max(4)).ok();
    }
    if half_act_scratch.is_none() {
        return false;
    }

    // Phase 1: cast activations f32 → f16/bf16 into the scratch.
    let n_total = m * k * batch.max(1);
    let dtype_id: u32 = match half_dtype {
        crate::arena::HalfDtype::F16 => 0,
        crate::arena::HalfDtype::Bf16 => 1,
    };
    {
        let kernel = crate::kernels::cast_f32_to_half_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(n_total, 256);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let src_view = arena
            .f32_buf()
            .slice(a_off_f32 as usize..a_off_f32 as usize + n_total as usize);
        let scratch_mut = half_act_scratch.as_mut().unwrap();
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&src_view)
            .arg(scratch_mut)
            .arg(&n_total)
            .arg(&dtype_id);
        if unsafe { launcher.launch(cfg) }.is_err() {
            return false;
        }
    }

    // Phase 2: cublasGemmEx with both inputs half + f32 output.
    let blas = blas.lock().unwrap();
    let arena_ptr_u64 = {
        let (p, _ar) = arena.buffer.device_ptr_mut(stream);
        p
    };
    let (half_buf_ptr, _hb) = arena.half_buffer.as_mut().unwrap().device_ptr_mut(stream);
    let scratch_ptr_u64 = {
        let s = half_act_scratch.as_mut().unwrap();
        let (p, _r) = s.device_ptr_mut(stream);
        p
    };
    let weight_dev = half_buf_ptr + (half_off as u64) * 2; // u16 = 2 bytes
    let act_dev = scratch_ptr_u64;
    let c_dev = arena_ptr_u64 + (c_off_f32 as u64) * 4;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let cuda_dt = match half_dtype {
        crate::arena::HalfDtype::F16 => cublas_sys::cudaDataType_t::CUDA_R_16F,
        crate::arena::HalfDtype::Bf16 => cublas_sys::cudaDataType_t::CUDA_R_16BF,
    };
    let compute_ty = match half_dtype {
        crate::arena::HalfDtype::F16 => {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F
        }
        crate::arena::HalfDtype::Bf16 => {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
        }
    };
    let result = unsafe {
        cudarc::cublas::result::gemm_ex(
            *blas.handle(),
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
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
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            n as i32,
            compute_ty,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    if let Err(ref e) = result {
        log_fallback("matmul.gemmEx (mixed-precision)", e);
    }
    result.is_ok()
}

/// One-time-per-tier log when a fast-path dispatch silently falls
/// back. Helps cloud-GPU debugging see *why* the slow path took over —
/// otherwise the only signal is unexpectedly low throughput.
/// Gated behind `RLX_CUDA_LOG_FALLBACK=1` so production isn't spammed.
pub(crate) fn log_fallback(tier: &str, err: impl std::fmt::Debug) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        rlx_ir::env::var("RLX_CUDA_LOG_FALLBACK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    if enabled {
        eprintln!("rlx-cuda: tier '{tier}' fell back: {err:?}");
    }
}

/// Stable, profiler-friendly name for an NVTX range covering a Step
/// dispatch. Matches the variant name; nsight-systems / nvprof show
/// these as range boundaries in the timeline.
pub(crate) fn fft_dtype_tag(dtype: rlx_ir::DType) -> u32 {
    match dtype {
        rlx_ir::DType::F32 => 0,
        rlx_ir::DType::F64 => 1,
        rlx_ir::DType::C64 => 2,
        other => panic!("rlx-cuda Op::Fft: unsupported dtype {other:?}"),
    }
}

pub(crate) fn fft_dtype_from_tag(tag: u32) -> rlx_ir::DType {
    match tag {
        0 => rlx_ir::DType::F32,
        1 => rlx_ir::DType::F64,
        2 => rlx_ir::DType::C64,
        other => panic!("rlx-cuda Op::Fft: bad dtype tag {other}"),
    }
}

/// Pre-compile every NVRTC kernel against `ctx`. Used by AOT mode to
/// move JIT compile cost out of the first-run critical path. Runs at
/// most once per process — later `CompileMode::Aot` compiles skip it.
pub(crate) static AOT_PREWARM_ONCE: std::sync::Once = std::sync::Once::new();

pub(crate) fn prewarm_all(ctx: &Arc<CudaContext>) {
    AOT_PREWARM_ONCE.call_once(|| prewarm_all_kernels(ctx));
}

pub(crate) fn prewarm_all_kernels(ctx: &Arc<CudaContext>) {
    use crate::kernels::*;
    let _ = binary_kernel(ctx);
    let _ = fused_binary_unary_kernel(ctx);
    let _ = unary_kernel(ctx);
    let _ = copy_kernel(ctx);
    let _ = matmul_kernel(ctx);
    let _ = matmul_epilogue_kernel(ctx);
    let _ = compare_kernel(ctx);
    let _ = where_kernel(ctx);
    let _ = reduce_kernel(ctx);
    let _ = softmax_kernel(ctx);
    let _ = layernorm_kernel(ctx);
    let _ = fused_residual_ln_kernel(ctx);
    let _ = fused_residual_rms_norm_kernel(ctx);
    let _ = ada_layer_norm_kernel(ctx);
    let _ = gated_residual_kernel(ctx);
    let _ = ada_layer_norm_backward_kernel(ctx);
    let _ = gated_residual_backward_kernel(ctx);
    let _ = gather_kernel(ctx);
    let _ = gather_axis_kernel(ctx);
    let _ = narrow_kernel(ctx);
    let _ = concat_kernel(ctx);
    let _ = transpose_kernel(ctx);
    let _ = expand_kernel(ctx);
    let _ = attention_kernel(ctx);
    let _ = attention_row_kernel(ctx);
    let _ = attention_bwd_kernel(ctx);
    let _ = argmax_kernel(ctx);
    let _ = rope_kernel(ctx);
    let _ = cumsum_kernel(ctx);
    let _ = topk_kernel(ctx);
    let _ = grouped_matmul_kernel(ctx);
    let _ = scatter_add_zero_kernel(ctx);
    let _ = scatter_add_acc_kernel(ctx);
    let _ = dequant_matmul_kernel(ctx);
    let _ = dequant_matmul_gguf_kernel(ctx);
    let _ = dequant_gguf_kernel(ctx);
    let _ = sample_kernel(ctx);
    let _ = selective_scan_kernel(ctx);
    let _ = gated_delta_net_kernel(ctx);
    let _ = pool1d_kernel(ctx);
    let _ = pool2d_kernel(ctx);
    let _ = pool3d_kernel(ctx);
    let _ = conv1d_kernel(ctx);
    let _ = conv2d_kernel(ctx);
    let _ = im2col_kernel(ctx);
    let _ = conv3d_kernel(ctx);
    let _ = layer_norm2d_kernel(ctx);
    let _ = conv_transpose2d_kernel(ctx);
    let _ = group_norm_kernel(ctx);
    let _ = resize_nearest_2x_kernel(ctx);
    let _ = elementwise_region_kernel(ctx);
    let _ = batch_elementwise_region_kernel(ctx);
    // matmul_wmma deliberately excluded: requires SM 70+ and may fail
    // load_module on older GPUs. Compile lazily on first opt-in dispatch.
}

pub(crate) fn im2col_use_gpu(n: u32, exec_mode: ExecMode) -> bool {
    if rlx_ir::env::var("RLX_CUDA_IM2COL_HOST").is_some() {
        return false;
    }
    if matches!(exec_mode, ExecMode::Graph) {
        return n > 0;
    }
    n > 0
}

pub(crate) fn pinned_host_io_disabled() -> bool {
    rlx_ir::env::var("RLX_CUDA_PINNED_IO").is_some_and(|v| v.eq_ignore_ascii_case("0"))
}

/// Pinned host output staging (faster D2H). On by default; set `RLX_CUDA_PINNED_IO=0` to disable.
pub(crate) fn pinned_output_staging_enabled() -> bool {
    !pinned_host_io_disabled()
}

/// Pinned host input staging for H2D. Graph mode always; stream mode when `RLX_CUDA_PINNED_IO=1`.
pub(crate) fn pinned_input_staging_enabled(exec_mode: ExecMode) -> bool {
    if pinned_host_io_disabled() {
        return false;
    }
    matches!(exec_mode, ExecMode::Graph)
        || rlx_ir::env::var("RLX_CUDA_PINNED_IO").is_some_and(|v| !v.eq_ignore_ascii_case("0"))
}

pub(crate) fn normalize_read_indices(buf: &mut Vec<usize>) {
    if buf.len() > 1 {
        buf.sort_unstable();
        buf.dedup();
    }
}

pub(crate) fn compile_mode_from_env() -> CompileMode {
    match rlx_ir::env::var("RLX_CUDA_COMPILE_MODE").as_deref() {
        Some(mode) if mode.eq_ignore_ascii_case("aot") => CompileMode::Aot,
        _ => CompileMode::Jit,
    }
}

pub(crate) fn exec_mode_from_env() -> ExecMode {
    match rlx_ir::env::var("RLX_CUDA_EXEC_MODE").as_deref() {
        Some(mode) if mode.eq_ignore_ascii_case("graph") => ExecMode::Graph,
        Some(mode) => {
            let lower = mode.to_ascii_lowercase();
            if let Some(rest) = lower.strip_prefix("multistream") {
                let n = rest.trim_start_matches([':', '=']).parse().unwrap_or(2);
                ExecMode::MultiStream(n.max(1))
            } else {
                ExecMode::Stream
            }
        }
        _ => ExecMode::Stream,
    }
}
