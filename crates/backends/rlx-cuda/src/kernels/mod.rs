// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA C++ kernel sources + NVRTC compilation cache.
//!
//! Sources live in [`rlx_gpu_kernels`]; this module JIT-compiles them to
//! PTX via NVRTC on first use, then caches `cuModule` handles for the
//! rest of the process. Pure NVRTC — no nvcc at workspace build time.

pub use rlx_gpu_kernels::*;

use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};

/// One compiled NVRTC module + the function handle we use from it.
pub struct CudaKernel {
    pub module: Arc<CudaModule>,
    pub function: CudaFunction,
}

/// Persistent PTX disk cache directory. Resolved once at startup from
/// `RLX_CUDA_PTX_CACHE` (explicit override) or `XDG_CACHE_HOME` /
/// `~/.cache`, namespaced by the cuda toolkit version baked into the
/// crate. Returning `None` disables caching (still works, just slower
/// cold-start).
fn ptx_cache_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Some(p) = rlx_ir::env::var("RLX_CUDA_PTX_CACHE") {
        return Some(PathBuf::from(p));
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache"))
        })?;
    Some(base.join("rlx-cuda").join("ptx-cuda-12060"))
}

/// FNV-1a 64-bit. Cheap and deterministic; collision-resistance is
/// good enough for filename hashing where source mismatch is the only
/// failure mode (we re-compile on cache miss, so no correctness risk).
fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Dump one NVRTC translation unit for offline kernel inspection. Writes
/// `<dir>/cu/<entry>.cu` (the exact assembled source NVRTC compiled) and
/// `<dir>/ptx/<entry>.ptx` (rlx's own compiled PTX), and appends one JSON
/// line to `<dir>/manifest.jsonl`. `src_hash` is the TU identity so the
/// analyzer can dedup entry-points that share a `.cu` file. Best-effort:
/// dump failures never disturb the compile. Appends are O_APPEND with sub-
/// PIPE_BUF lines, so concurrent test threads interleave cleanly.
fn dump_kernel(dir: &str, entry: &str, src: &str, ptx_src: &str) {
    use std::io::Write;
    let base = std::path::Path::new(dir);
    let cu_dir = base.join("cu");
    let ptx_dir = base.join("ptx");
    let _ = std::fs::create_dir_all(&cu_dir);
    let _ = std::fs::create_dir_all(&ptx_dir);
    let _ = std::fs::write(cu_dir.join(format!("{entry}.cu")), src);
    let _ = std::fs::write(ptx_dir.join(format!("{entry}.ptx")), ptx_src);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(base.join("manifest.jsonl"))
    {
        let _ = writeln!(
            f,
            "{{\"entry\":\"{entry}\",\"src_hash\":\"{:016x}\",\"src_bytes\":{},\"ptx_bytes\":{}}}",
            fnv1a64(src),
            src.len(),
            ptx_src.len()
        );
    }
}

/// Portable compile: NVRTC emits PTX for its baseline virtual arch and the
/// driver JITs to whatever GPU is present. Correct and forward-compatible for
/// every non-arch-specific kernel — which is all of them today.
/// CUDA toolkit include dir(s) for NVRTC (which has no default header search
/// path). Probes `CUDA_PATH`/`CUDA_HOME`/`CUDA_ROOT` then the common install
/// prefixes; keeps only dirs that actually exist. Lets kernels `#include
/// <mma.h>` (wmma) / `<cuda_fp16.h>`.
fn cuda_include_dirs() -> Vec<String> {
    let mut out = Vec::new();
    for k in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(p) = std::env::var(k) {
            out.push(format!("{p}/include"));
        }
    }
    for p in [
        "/usr/local/cuda/include",
        "/opt/cuda/include",
        "/usr/lib/cuda/include",
    ] {
        out.push(p.to_string());
    }
    out.retain(|p| std::path::Path::new(p).is_dir());
    out.dedup();
    out
}

pub(crate) fn compile(ctx: &Arc<CudaContext>, src: &str, entry: &str) -> CudaKernel {
    compile_with_arch(ctx, src, entry, None)
}

/// Compile pinning an optional NVRTC target arch. `Some("compute_90a")` unlocks
/// Hopper TMA/wgmma PTX (needs `sm_90` at runtime); `None` is the portable path.
/// The arch tag folds into the disk-cache key so a `compute_90a` build and the
/// portable build of the same entry never collide.
pub(crate) fn compile_with_arch(
    ctx: &Arc<CudaContext>,
    src: &str,
    entry: &str,
    arch: Option<&'static str>,
) -> CudaKernel {
    // Try the disk cache first. The cache key folds the kernel entry
    // name and target arch into the source hash so different entry-points
    // sharing a .cu file (scatter_add_zero / scatter_add_acc), and portable
    // vs arch-pinned builds of the same entry, get distinct cache slots.
    let arch_tag = arch.unwrap_or("portable");
    let cache_path = ptx_cache_dir()
        .map(|d| d.join(format!("{}-{}-{:016x}.ptx", entry, arch_tag, fnv1a64(src))));

    let compile_fresh = || {
        let opts = cudarc::nvrtc::CompileOptions {
            arch,
            // NVRTC has no default header search path, so kernels that
            // `#include <mma.h>` (wmma tensor cores) / `<cuda_fp16.h>` fail to
            // find them. Add the toolkit include dir(s). Harmless for the
            // kernels that include nothing.
            include_paths: cuda_include_dirs(),
            ..Default::default()
        };
        cudarc::nvrtc::compile_ptx_with_opts(src, opts).unwrap_or_else(|e| {
            panic!("rlx-cuda: NVRTC compile failed for {entry} ({arch_tag}): {e}")
        })
    };

    let ptx = if let Some(ref p) = cache_path {
        if let Ok(cached) = std::fs::read_to_string(p) {
            cudarc::nvrtc::Ptx::from_src(cached)
        } else {
            let fresh = compile_fresh();
            // Best-effort write to the cache. Atomic via tmp + rename
            // so a crash mid-write doesn't poison the cache.
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let tmp = p.with_extension("ptx.tmp");
            if std::fs::write(&tmp, fresh.to_src()).is_ok() {
                let _ = std::fs::rename(&tmp, p);
            }
            fresh
        }
    } else {
        compile_fresh()
    };

    // Kernel-inspection hook: with `RLX_DUMP_KERNELS=<dir>` set, snapshot
    // the exact translation unit NVRTC saw (post gelu/codegen assembly) plus
    // rlx's own compiled PTX, so the offline tooling in `tools/kernel-inspect/`
    // can produce SASS / occupancy / register reports on the target GPU.
    // Off by default; captures both the static `kernel_cache!` kernels and
    // the dynamic `CudaGpuKernel` registry (both route through here).
    if let Some(dir) = rlx_ir::env::var("RLX_DUMP_KERNELS") {
        dump_kernel(&dir, entry, src, &ptx.to_src());
    }

    let module = ctx
        .load_module(ptx)
        .unwrap_or_else(|e| panic!("rlx-cuda: load_module failed for {entry}: {e}"));
    let function = module
        .load_function(entry)
        .unwrap_or_else(|e| panic!("rlx-cuda: load_function {entry}: {e}"));
    CudaKernel { module, function }
}

macro_rules! kernel_cache {
    ($static_name:ident, $fn_name:ident, $src:expr, $entry:expr) => {
        static $static_name: OnceLock<CudaKernel> = OnceLock::new();
        pub fn $fn_name(ctx: &Arc<CudaContext>) -> &'static CudaKernel {
            $static_name.get_or_init(|| compile(ctx, $src, $entry))
        }
    };
}

/// Like `kernel_cache!` but pins the Hopper target when the running device
/// supports it (else portable). `$arch_fn` is a
/// `fn(&Arc<CudaContext>) -> Option<&'static str>`, e.g.
/// [`crate::backend::tma_arch`]. The `OnceLock` binds the first
/// device's variant — the same single-device assumption `kernel_cache!`
/// already makes.
macro_rules! kernel_cache_arch {
    ($static_name:ident, $fn_name:ident, $src:expr, $entry:expr, $arch_fn:path) => {
        static $static_name: OnceLock<CudaKernel> = OnceLock::new();
        pub fn $fn_name(ctx: &Arc<CudaContext>) -> &'static CudaKernel {
            $static_name.get_or_init(|| compile_with_arch(ctx, $src, $entry, $arch_fn(ctx)))
        }
    };
}

kernel_cache!(BINARY, binary_kernel, BINARY_CU, "binary");
// On-device complex simulation (f32-lane): standalone complex cast + C64
// binary. Shared with rlx-wgpu's `complex_cast.wgsl` / `binary_c64.wgsl`.
kernel_cache!(
    COMPLEX_CAST,
    complex_cast_kernel,
    COMPLEX_CAST_CU,
    "complex_cast"
);
kernel_cache!(BINARY_C64, binary_c64_kernel, BINARY_C64_CU, "binary_c64");
// C64 Wirtinger surface (ComplexNormSq / Backward / Conjugate). Three entry
// points share `complex_wirtinger.cu`; each gets its own OnceLock cache.
kernel_cache!(
    COMPLEX_NORM_SQ,
    complex_norm_sq_kernel,
    COMPLEX_WIRINGER_CU,
    "complex_norm_sq"
);
kernel_cache!(
    COMPLEX_NORM_SQ_BWD,
    complex_norm_sq_backward_kernel,
    COMPLEX_WIRINGER_CU,
    "complex_norm_sq_backward"
);
kernel_cache!(
    CONJUGATE_C64,
    conjugate_c64_kernel,
    COMPLEX_WIRINGER_CU,
    "conjugate_c64"
);
kernel_cache!(
    BINARY_BROADCAST,
    binary_broadcast_kernel,
    BINARY_BROADCAST_CU,
    "binary_broadcast"
);
kernel_cache!(LSTM_DIR, lstm_dir_kernel, LSTM_CU, "lstm_dir");
kernel_cache!(LSTM_PRE_WIH, lstm_pre_wih_kernel, LSTM_CU, "lstm_pre_wih");
kernel_cache!(
    LSTM_PRE_ADD_BIAS,
    lstm_pre_add_bias_kernel,
    LSTM_CU,
    "lstm_pre_add_bias"
);
kernel_cache!(
    LSTM_TRANSPOSE,
    lstm_transpose_kernel,
    LSTM_CU,
    "transpose_rc"
);
kernel_cache!(
    RNG_NORMAL_PHILOX,
    rng_normal_philox_kernel,
    RNG_PHILOX_CU,
    "rng_normal_philox"
);
kernel_cache!(
    RNG_UNIFORM_PHILOX,
    rng_uniform_philox_kernel,
    RNG_PHILOX_CU,
    "rng_uniform_philox"
);
kernel_cache!(
    RNG_FILL_ZERO,
    rng_fill_zero_kernel,
    RNG_PHILOX_CU,
    "rng_fill_zero"
);
kernel_cache!(GRU, gru_kernel, GRU_CU, "gru");
kernel_cache!(RNN, rnn_kernel, RNN_CU, "rnn");
kernel_cache!(MAMBA2, mamba2_kernel, MAMBA2_CU, "mamba2");
kernel_cache!(
    FUSED_BINARY_UNARY,
    fused_binary_unary_kernel,
    rlx_gpu_kernels::fused_binary_unary_cuda_src(),
    "fused_binary_unary"
);
kernel_cache!(
    CAST_F32_TO_HALF,
    cast_f32_to_half_kernel,
    CAST_F32_TO_HALF_CU,
    "cast_f32_to_half"
);
kernel_cache!(
    SCALED_QUANT_SCALE,
    scaled_quant_scale_kernel,
    SCALED_LOWP_CU,
    "scaled_quant_scale_per_tensor"
);
kernel_cache!(
    SCALED_QUANTIZE_FP8,
    scaled_quantize_fp8_kernel,
    SCALED_LOWP_CU,
    "scaled_quantize_fp8_per_tensor"
);
kernel_cache!(
    SCALED_QUANT_SCALE_GENERAL,
    scaled_quant_scale_general_kernel,
    SCALED_LOWP_GENERAL_CU,
    "scaled_quant_scale_general"
);
kernel_cache!(
    SCALED_QUANTIZE_GENERAL,
    scaled_quantize_general_kernel,
    SCALED_LOWP_GENERAL_CU,
    "scaled_quantize_general"
);
kernel_cache!(
    SCALED_DEQUANTIZE_GENERAL,
    scaled_dequantize_general_kernel,
    SCALED_LOWP_GENERAL_CU,
    "scaled_dequantize_general"
);
kernel_cache!(
    SCALED_MATMUL_DECODE,
    scaled_matmul_decode_kernel,
    SCALED_LOWP_GENERAL_CU,
    "scaled_matmul_decode"
);
kernel_cache!(
    SCALED_GROUPED_MATMUL_DECODE,
    scaled_grouped_matmul_decode_kernel,
    SCALED_LOWP_GENERAL_CU,
    "scaled_grouped_matmul_decode"
);
kernel_cache!(
    MXFP4X2_DEQUANT_NK,
    mxfp4x2_dequant_nk_kernel,
    SCALED_LOWP_GENERAL_CU,
    "mxfp4x2_dequant_nk"
);
kernel_cache!(
    UNARY,
    unary_kernel,
    rlx_gpu_kernels::unary_cuda_src(),
    "unary"
);
kernel_cache!(COPY, copy_kernel, COPY_CU, "copy");
kernel_cache!(PAD, pad_kernel, PAD_CU, "pad");
kernel_cache!(SLICE, slice_kernel, SLICE_CU, "slice");
kernel_cache!(
    MATMUL,
    matmul_kernel,
    rlx_gpu_kernels::matmul_cuda_src(),
    "matmul"
);
kernel_cache!(
    MATMUL_BT,
    matmul_bt_kernel,
    rlx_gpu_kernels::MATMUL_BT_CU,
    "matmul_bt"
);
kernel_cache!(
    MATMUL_EPILOGUE,
    matmul_epilogue_kernel,
    rlx_gpu_kernels::matmul_epilogue_cuda_src(),
    "matmul_epilogue"
);
kernel_cache!(
    CONV_BIAS_ACT_EPILOGUE,
    conv_bias_act_epilogue_kernel,
    rlx_gpu_kernels::conv_bias_act_epilogue_cuda_src(),
    "conv_bias_act_epilogue"
);
kernel_cache!(
    MATMUL_WMMA,
    matmul_wmma_kernel,
    MATMUL_WMMA_CU,
    "matmul_wmma"
);
// Hopper TMA-staged GEMM: compiled for `compute_90a` when the running device
// is sm_90 and `RLX_CUDA_TMA` is set, else the portable fallback (which traps
// on non-Hopper — dispatch never routes here off sm_90, so it can't fire).
kernel_cache_arch!(
    MATMUL_TMA,
    matmul_tma_kernel,
    MATMUL_TMA_CU,
    "matmul_tma",
    crate::backend::tma_arch
);
// TMA NT GEMM (C = A·Wᵀ) for the GGUF prefill post-dequant matmul. Same gate.
kernel_cache_arch!(
    MATMUL_BT_TMA,
    matmul_bt_tma_kernel,
    MATMUL_BT_TMA_CU,
    "matmul_bt_tma",
    crate::backend::tma_arch
);
kernel_cache!(COMPARE, compare_kernel, COMPARE_CU, "compare");
kernel_cache!(WHEREK, where_kernel, WHERE_CU, "where_select");
kernel_cache!(FMA, fma_kernel, FMA_CU, "fma_elem");
kernel_cache!(REDUCE, reduce_kernel, REDUCE_CU, "reduce");
kernel_cache!(SOFTMAX, softmax_kernel, SOFTMAX_CU, "softmax");
kernel_cache!(
    RELU_BACKWARD,
    relu_backward_kernel,
    ACTIVATION_BACKWARD_CU,
    "relu_backward"
);
kernel_cache!(
    ACTIVATION_BACKWARD,
    activation_backward_kernel,
    ACTIVATION_BACKWARD_CU,
    "activation_backward"
);
kernel_cache!(
    SOFTMAX_CROSS_ENTROPY,
    softmax_cross_entropy_kernel,
    SOFTMAX_CROSS_ENTROPY_CU,
    "softmax_cross_entropy"
);
kernel_cache!(
    SOFTMAX_CROSS_ENTROPY_WITH_LOGITS,
    softmax_cross_entropy_with_logits_kernel,
    SOFTMAX_CROSS_ENTROPY_CU,
    "softmax_cross_entropy_with_logits"
);
kernel_cache!(
    SOFTMAX_CROSS_ENTROPY_BACKWARD,
    softmax_cross_entropy_backward_kernel,
    SOFTMAX_CROSS_ENTROPY_CU,
    "softmax_cross_entropy_backward"
);
kernel_cache!(LAYERNORM, layernorm_kernel, LAYERNORM_CU, "rlx_norm");
kernel_cache!(
    LAYER_NORM_BWD_INPUT,
    layer_norm_bwd_input_kernel,
    LAYER_NORM_BWD_CU,
    "layer_norm_bwd_input"
);
kernel_cache!(
    LAYER_NORM_BWD_GAMMA,
    layer_norm_bwd_gamma_kernel,
    LAYER_NORM_BWD_CU,
    "layer_norm_bwd_gamma"
);
kernel_cache!(
    FAKE_QUANTIZE_FIXED,
    fake_quantize_fixed_kernel,
    FAKE_QUANTIZE_CU,
    "fake_quantize_fixed"
);
kernel_cache!(
    FAKE_QUANTIZE_PERBATCH,
    fake_quantize_perbatch_kernel,
    FAKE_QUANTIZE_CU,
    "fake_quantize_perbatch"
);
kernel_cache!(
    FAKE_QUANTIZE_EMA,
    fake_quantize_ema_kernel,
    FAKE_QUANTIZE_CU,
    "fake_quantize_ema"
);
kernel_cache!(
    FAKE_QUANTIZE_LSQ_BWD_X,
    fake_quantize_lsq_bwd_x_kernel,
    FAKE_QUANTIZE_CU,
    "fake_quantize_lsq_bwd_x"
);
kernel_cache!(
    FAKE_QUANTIZE_LSQ_BWD_SCALE,
    fake_quantize_lsq_bwd_scale_kernel,
    FAKE_QUANTIZE_CU,
    "fake_quantize_lsq_bwd_scale"
);
kernel_cache!(
    FAKE_QUANTIZE_BACKWARD,
    fake_quantize_backward_kernel,
    FAKE_QUANTIZE_CU,
    "fake_quantize_backward"
);
kernel_cache!(QUANTIZE_I8, quantize_i8_kernel, QUANTIZE_CU, "quantize_i8");
kernel_cache!(
    DEQUANTIZE_I8,
    dequantize_i8_kernel,
    QUANTIZE_CU,
    "dequantize_i8"
);
kernel_cache!(Q_MATMUL, q_matmul_kernel, Q_MATMUL_CU, "q_matmul");
kernel_cache!(Q_CONV2D, q_conv2d_kernel, Q_CONV2D_CU, "q_conv2d");
kernel_cache!(
    RMS_NORM_BWD,
    rms_norm_backward_kernel,
    RMS_NORM_BWD_CU,
    "rlx_rms_norm_bwd"
);
kernel_cache!(
    RMS_NORM_BWD_ZERO,
    rms_norm_bwd_zero_kernel,
    RMS_NORM_BWD_CU,
    "rlx_zero_f32"
);
kernel_cache!(
    CUMSUM_BWD,
    cumsum_backward_kernel,
    CUMSUM_BWD_CU,
    "rlx_cumsum_bwd"
);
kernel_cache!(ROPE_BWD, rope_backward_kernel, ROPE_BWD_CU, "rlx_rope_bwd");
kernel_cache!(
    GATHER_BWD,
    gather_backward_kernel,
    GATHER_BWD_CU,
    "rlx_gather_axis_bwd"
);
kernel_cache!(
    FUSED_RESIDUAL_LN,
    fused_residual_ln_kernel,
    FUSED_RESIDUAL_LN_CU,
    "fused_residual_ln"
);
kernel_cache!(
    FUSED_RESIDUAL_RMS_NORM,
    fused_residual_rms_norm_kernel,
    FUSED_RESIDUAL_RMS_NORM_CU,
    "fused_residual_rms_norm"
);
kernel_cache!(
    ADA_LAYER_NORM,
    ada_layer_norm_kernel,
    ADA_LAYER_NORM_CU,
    "ada_layer_norm"
);
kernel_cache!(
    GATED_RESIDUAL,
    gated_residual_kernel,
    GATED_RESIDUAL_CU,
    "gated_residual"
);
kernel_cache!(
    ADA_LAYER_NORM_BACKWARD,
    ada_layer_norm_backward_kernel,
    ADA_LAYER_NORM_BACKWARD_CU,
    "ada_layer_norm_backward"
);
kernel_cache!(
    GATED_RESIDUAL_BACKWARD,
    gated_residual_backward_kernel,
    GATED_RESIDUAL_BACKWARD_CU,
    "gated_residual_backward"
);
kernel_cache!(GATHER, gather_kernel, GATHER_CU, "gather");
kernel_cache!(
    GATHER_AXIS,
    gather_axis_kernel,
    GATHER_AXIS_CU,
    "gather_axis"
);
kernel_cache!(NARROW, narrow_kernel, NARROW_CU, "narrow");
kernel_cache!(KV_APPEND, kv_append_kernel, KV_APPEND_CU, "kv_append");
kernel_cache!(CONCAT, concat_kernel, CONCAT_CU, "concat");
kernel_cache!(TRANSPOSE, transpose_kernel, TRANSPOSE_CU, "transpose");
kernel_cache!(EXPAND, expand_kernel, EXPAND_CU, "expand");
kernel_cache!(ATTENTION, attention_kernel, ATTENTION_CU, "attention");
// Tensor-Core (fp16 WMMA) attention — CUDA-only drop-in, opt-in via
// `RLX_CUDA_ATTENTION_WMMA` dispatch in backend/run.rs.
kernel_cache!(
    ATTENTION_WMMA,
    attention_wmma_kernel,
    ATTENTION_WMMA_CU,
    "attention_wmma"
);
// head_dim<=128 variant (2 warps, 32-query tile) — same source file.
kernel_cache!(
    ATTENTION_WMMA_D128,
    attention_wmma_d128_kernel,
    ATTENTION_WMMA_CU,
    "attention_wmma_d128"
);
kernel_cache!(
    FUSED_ATTN,
    fused_attn_kernel,
    FUSED_ATTN_CU,
    "fused_attn_block"
);
kernel_cache!(
    ATTENTION_ROW,
    attention_row_kernel,
    ATTENTION_ROW_CU,
    "attention_row"
);
kernel_cache!(
    ATTENTION_WARP,
    attention_warp_kernel,
    ATTENTION_WARP_CU,
    "attention_warp"
);
kernel_cache!(
    ATTENTION_BWD,
    attention_bwd_kernel,
    ATTENTION_BWD_CU,
    "attention_bwd"
);
kernel_cache!(ARGMAX, argmax_kernel, ARGMAX_CU, "argmax");
kernel_cache!(ROPE, rope_kernel, ROPE_CU, "rope");
kernel_cache!(CUMSUM, cumsum_kernel, CUMSUM_CU, "cumsum");
kernel_cache!(CUM_SCAN, cum_scan_kernel, CUM_SCAN_CU, "cum_scan");
kernel_cache!(TOPK, topk_kernel, TOPK_CU, "topk");
kernel_cache!(
    GROUPED_MATMUL,
    grouped_matmul_kernel,
    GROUPED_MATMUL_CU,
    "grouped_matmul"
);
kernel_cache!(
    GROUPED_GEMV_SPLITK,
    grouped_gemv_splitk_kernel,
    GROUPED_MATMUL_CU,
    "grouped_gemv_splitk"
);
kernel_cache!(
    SCATTER_ADD_ZERO,
    scatter_add_zero_kernel,
    SCATTER_ADD_CU,
    "scatter_add_zero"
);
kernel_cache!(
    SCATTER_ADD_ACC,
    scatter_add_acc_kernel,
    SCATTER_ADD_CU,
    "scatter_add_acc"
);
kernel_cache!(
    SCATTER_ND,
    scatter_nd_kernel,
    SCATTER_ND_CU,
    "scatter_nd_f32"
);
kernel_cache!(
    DEQUANT_MATMUL,
    dequant_matmul_kernel,
    DEQUANT_MATMUL_CU,
    "dequant_matmul"
);
kernel_cache!(
    DEQUANT_MATMUL_MLX,
    dequant_matmul_mlx_kernel,
    DEQUANT_MATMUL_MLX_CU,
    "dequant_matmul_mlx"
);
kernel_cache!(
    DEQUANT_MATMUL_MLX_GEMV,
    dequant_matmul_mlx_gemv_kernel,
    DEQUANT_MATMUL_MLX_CU,
    "dequant_matmul_mlx_gemv"
);
kernel_cache!(
    DEQUANT_MATMUL_MLX_GEMM,
    dequant_matmul_mlx_gemm_kernel,
    DEQUANT_MATMUL_MLX_CU,
    "dequant_matmul_mlx_gemm"
);
kernel_cache!(
    DEQUANT_GROUPED_MATMUL_MLX_MXFP4,
    dequant_grouped_matmul_mlx_mxfp4_kernel,
    DEQUANT_MATMUL_MLX_CU,
    "dequant_grouped_matmul_mlx_mxfp4"
);
kernel_cache!(
    DEQUANT_GROUPED_MATMUL_MLX_MXFP4_V3,
    dequant_grouped_matmul_mlx_mxfp4_v3_kernel,
    DEQUANT_MATMUL_MLX_CU,
    "dequant_grouped_matmul_mlx_mxfp4_v3"
);
kernel_cache!(
    DEQUANT_GROUPED_MATMUL_MLX_MXFP4_SPLITK,
    dequant_grouped_matmul_mlx_mxfp4_splitk_kernel,
    DEQUANT_MATMUL_MLX_CU,
    "dequant_grouped_matmul_mlx_mxfp4_splitk"
);
kernel_cache!(
    DEQUANT_GROUPED_MATMUL_MLX_MXFP4_AMORT,
    dequant_grouped_matmul_mlx_mxfp4_amort_kernel,
    DEQUANT_MATMUL_MLX_CU,
    "dequant_grouped_matmul_mlx_mxfp4_amort"
);
kernel_cache!(
    DEQUANT_GGUF,
    dequant_gguf_kernel,
    DEQUANT_GGUF_CU,
    "dequant_gguf"
);
kernel_cache!(
    DEQUANT_MATMUL_GGUF,
    dequant_matmul_gguf_kernel,
    DEQUANT_MATMUL_GGUF_CU,
    "dequant_matmul_gguf"
);
kernel_cache!(
    DEQUANT_MATMUL_GGUF_Q1_GEMV,
    dequant_matmul_gguf_q1_gemv_kernel,
    DEQUANT_MATMUL_GGUF_CU,
    "dequant_matmul_gguf_q1_gemv"
);
kernel_cache!(
    DEQUANT_MATMUL_GGUF_Q4K_GEMV,
    dequant_matmul_gguf_q4k_gemv_kernel,
    DEQUANT_MATMUL_GGUF_CU,
    "dequant_matmul_gguf_q4k_gemv"
);
kernel_cache!(
    DEQUANT_MATMUL_GGUF_Q4K_GEMV_WARP,
    dequant_matmul_gguf_q4k_gemv_warp_kernel,
    DEQUANT_MATMUL_GGUF_CU,
    "dequant_matmul_gguf_q4k_gemv_warp"
);
kernel_cache!(SAMPLE, sample_kernel, SAMPLE_CU, "sample");
kernel_cache!(
    SELECTIVE_SCAN,
    selective_scan_kernel,
    SELECTIVE_SCAN_CU,
    "selective_scan"
);
kernel_cache!(
    GATED_DELTA_NET,
    gated_delta_net_kernel,
    GATED_DELTA_NET_CU,
    "gated_delta_net"
);
kernel_cache!(
    KIMI_DELTA_CHUNK,
    kimi_delta_chunk_kernel,
    KIMI_DELTA_CHUNK_CU,
    "kimi_delta_chunk"
);
kernel_cache!(POOL1D, pool1d_kernel, POOL1D_CU, "pool1d");
kernel_cache!(POOL2D, pool2d_kernel, POOL2D_CU, "pool2d");
kernel_cache!(
    MAXPOOL2D_BWD,
    maxpool2d_backward_kernel,
    MAXPOOL2D_BACKWARD_CU,
    "maxpool2d_backward"
);
kernel_cache!(
    MAXPOOL3D_BWD,
    maxpool3d_backward_kernel,
    MAXPOOL3D_BACKWARD_CU,
    "maxpool3d_backward"
);
kernel_cache!(POOL3D, pool3d_kernel, POOL3D_CU, "pool3d");
kernel_cache!(CONV1D, conv1d_kernel, CONV1D_CU, "conv1d");
kernel_cache!(CONV2D, conv2d_kernel, CONV2D_CU, "conv2d");
kernel_cache!(
    CONV2D_BACKWARD_INPUT,
    conv2d_backward_input_kernel,
    CONV2D_BACKWARD_INPUT_CU,
    "conv2d_backward_input"
);
kernel_cache!(
    CONV2D_BACKWARD_WEIGHT,
    conv2d_backward_weight_kernel,
    CONV2D_BACKWARD_WEIGHT_CU,
    "conv2d_backward_weight"
);
kernel_cache!(IM2COL, im2col_kernel, IM2COL_CU, "im2col");
kernel_cache!(CONV3D, conv3d_kernel, CONV3D_CU, "conv3d");
kernel_cache!(
    CONV3D_BACKWARD_INPUT,
    conv3d_backward_input_kernel,
    CONV3D_BACKWARD_INPUT_CU,
    "conv3d_backward_input"
);
kernel_cache!(
    CONV3D_BACKWARD_WEIGHT,
    conv3d_backward_weight_kernel,
    CONV3D_BACKWARD_WEIGHT_CU,
    "conv3d_backward_weight"
);
kernel_cache!(
    CONV_TRANSPOSE3D,
    conv_transpose3d_kernel,
    CONV_TRANSPOSE3D_CU,
    "conv_transpose3d"
);
kernel_cache!(
    LAYER_NORM2D,
    layer_norm2d_kernel,
    LAYER_NORM2D_CU,
    "layer_norm2d"
);
kernel_cache!(
    CONV_TRANSPOSE2D,
    conv_transpose2d_kernel,
    CONV_TRANSPOSE2D_CU,
    "conv_transpose2d"
);
kernel_cache!(
    FUSED_SWIGLU,
    fused_swiglu_kernel,
    FUSED_SWIGLU_CU,
    "fused_swiglu"
);
kernel_cache!(
    AXIAL_ROPE2D,
    axial_rope2d_kernel,
    AXIAL_ROPE2D_CU,
    "axial_rope2d"
);
kernel_cache!(GROUP_NORM, group_norm_kernel, GROUP_NORM_CU, "group_norm");
kernel_cache!(
    GROUP_NORM_BWD_INPUT,
    group_norm_bwd_input_kernel,
    GROUP_NORM_BWD_CU,
    "group_norm_bwd_input"
);
kernel_cache!(
    GROUP_NORM_BWD_GAMMA,
    group_norm_bwd_gamma_kernel,
    GROUP_NORM_BWD_CU,
    "group_norm_bwd_gamma"
);
kernel_cache!(
    GROUP_NORM_BWD_BETA,
    group_norm_bwd_beta_kernel,
    GROUP_NORM_BWD_CU,
    "group_norm_bwd_beta"
);
kernel_cache!(
    BATCH_NORM_INFERENCE,
    batch_norm_inference_kernel,
    BATCH_NORM_INFERENCE_CU,
    "batch_norm_inference"
);
kernel_cache!(
    BATCH_NORM_INFERENCE_BWD_INPUT,
    batch_norm_inference_bwd_input_kernel,
    BATCH_NORM_INFERENCE_CU,
    "batch_norm_inference_bwd_input"
);
kernel_cache!(
    BATCH_NORM_INFERENCE_BWD_GAMMA,
    batch_norm_inference_bwd_gamma_kernel,
    BATCH_NORM_INFERENCE_CU,
    "batch_norm_inference_bwd_gamma"
);
kernel_cache!(
    BATCH_NORM_INFERENCE_BWD_BETA,
    batch_norm_inference_bwd_beta_kernel,
    BATCH_NORM_INFERENCE_CU,
    "batch_norm_inference_bwd_beta"
);
kernel_cache!(
    RESIZE_NEAREST_2X,
    resize_nearest_2x_kernel,
    RESIZE_NEAREST_2X_CU,
    "resize_nearest_2x"
);
kernel_cache!(
    INTERPOLATE3D,
    interpolate3d_kernel,
    INTERPOLATE3D_CU,
    "interpolate3d"
);
kernel_cache!(
    ELEMENTWISE_REGION,
    elementwise_region_kernel,
    rlx_gpu_kernels::elementwise_region_cuda_src(),
    "elementwise_region"
);
kernel_cache!(
    BATCH_ELEMENTWISE_REGION,
    batch_elementwise_region_kernel,
    rlx_gpu_kernels::batch_elementwise_region_cuda_src(),
    "batch_elementwise_region"
);
kernel_cache!(
    GAUSSIAN_SPLAT_RASTERIZE,
    gaussian_splat_rasterize_kernel,
    GAUSSIAN_SPLAT_RASTERIZE_CU,
    "gaussian_splat_rasterize"
);
kernel_cache!(
    FFT_RADIX2_FULL,
    fft_radix2_full_kernel,
    FFT_CU,
    "fft_radix2_full"
);
kernel_cache!(
    FFT_BIT_REVERSE,
    fft_bit_reverse_kernel,
    FFT_CU,
    "fft_bit_reverse"
);
kernel_cache!(FFT_INNER, fft_inner_kernel, FFT_CU, "fft_inner");
kernel_cache!(FFT_OUTER_R4, fft_outer_r4_kernel, FFT_CU, "fft_outer_r4");
kernel_cache!(FFT_OUTER_R2, fft_outer_r2_kernel, FFT_CU, "fft_outer_r2");
// cuFFT planar⇄interleaved bridge (only dispatched under the `cufft` feature;
// lazy NVRTC compile means these cost nothing unless actually used).
kernel_cache!(
    FFT_PACK_INTERLEAVE,
    fft_pack_interleave_kernel,
    FFT_CU,
    "fft_pack_interleave"
);
kernel_cache!(
    FFT_UNPACK_PLANAR,
    fft_unpack_planar_kernel,
    FFT_CU,
    "fft_unpack_planar"
);
// native-cuda-fft: Stockham single-kernel FFT (cuFFT-parity for n<=1024).
kernel_cache!(
    FFT_STOCKHAM_R4,
    fft_stockham_r4_kernel,
    FFT_CU,
    "fft_stockham_r4"
);
kernel_cache!(
    FFT_STOCKHAM_R2,
    fft_stockham_r2_kernel,
    FFT_CU,
    "fft_stockham_r2"
);
kernel_cache!(
    FFT_STOCKHAM_R8,
    fft_stockham_r8_kernel,
    FFT_CU,
    "fft_stockham_r8"
);
kernel_cache!(
    FFT_STOCKHAM_R16,
    fft_stockham_r16_kernel,
    FFT_CU,
    "fft_stockham_r16"
);
kernel_cache!(
    FFT_STOCKHAM_MIXED,
    fft_stockham_mixed_kernel,
    FFT_CU,
    "fft_stockham_mixed"
);
kernel_cache!(
    WELCH_PEAKS_GPU,
    welch_peaks_gpu_kernel,
    WELCH_PEAKS_CU,
    "welch_peaks_gpu"
);
kernel_cache!(
    FFT_BUTTERFLY_STAGE,
    fft_butterfly_stage_kernel,
    FFT_BUTTERFLY_STAGE_CU,
    "fft_butterfly_stage"
);

/// Dispatch grid for a 1-D workload of `n` threads with workgroup
/// size `block_x`. CUDA's per-grid-dim limit is 2^31-1 on the X axis,
/// so the 2-D fallback wgpu requires isn't needed here.
///
/// `n == 0` would yield `grid_dim.x == 0`, which CUDA rejects
/// (`CUDA_ERROR_INVALID_VALUE`). Empty tensors still get an arena slot
/// (see `arena_slot_bytes`); skip the launch at the call site, or use
/// this helper which returns a no-op `(1, block)` grid — kernels must
/// guard `idx < n`.
pub fn dispatch_grid_1d(n: u32, block_x: u32) -> (u32, u32) {
    let block_x = block_x.max(1);
    if n == 0 {
        return (1, block_x);
    }
    (n.div_ceil(block_x), block_x)
}

/// 2-D grid for pixel kernels (`block_x` × `block_y` threads per block).
pub fn dispatch_grid_2d(
    width: u32,
    height: u32,
    block_x: u32,
    block_y: u32,
) -> ((u32, u32, u32), (u32, u32, u32)) {
    (
        (width.div_ceil(block_x), height.div_ceil(block_y), 1),
        (block_x, block_y, 1),
    )
}

/// 3-D grid for NCHW resize-prologue region kernels (W × H × N·C).
pub fn dispatch_grid_prologue_nchw(w: u32, h: u32, nc: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
    const BX: u32 = 16;
    const BY: u32 = 16;
    (
        (w.div_ceil(BX), h.div_ceil(BY), nc),
        (BX.min(w.max(1)), BY.min(h.max(1)), 1),
    )
}
