// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HIP kernel sources + hipRTC compile cache.
//!
//! Mirror of `rlx-cuda::kernels`. Each `.cu` source compiles via
//! hipRTC the first time it's needed, then the resulting `.hsaco`
//! binary lives in a `OnceLock<HipKernel>` for the rest of the
//! process. Persistent disk cache under
//! `$RLX_ROCM_HSACO_CACHE` / `$XDG_CACHE_HOME/rlx-rocm/hsaco-rocm`
//! follows the same shape as rlx-cuda's PTX cache.

mod sources;
pub use sources::*;

use std::sync::Arc;
use std::sync::OnceLock;

use crate::device::RocmContext;
use crate::hip::HipKernel;

/// Disk cache directory for compiled `.hsaco` blobs. Returns `None`
/// to disable caching.
fn hsaco_cache_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Some(p) = rlx_ir::env::var("RLX_ROCM_HSACO_CACHE") {
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
    Some(base.join("rlx-rocm").join("hsaco-rocm"))
}

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub(crate) fn compile(ctx: &Arc<RocmContext>, src: &str, entry: &str) -> HipKernel {
    // Include the target arch in the cache key: a `.hsaco` compiled for one
    // gfx arch must never be reused for another (it would fail to load, or
    // load a wrong-arch binary). `default` when no arch is resolved.
    let arch = crate::hip::rocm_target_arch().unwrap_or_else(|| "default".to_string());
    let cache_path = hsaco_cache_dir()
        .map(|d| d.join(format!("{}-{}-{:016x}.hsaco", entry, arch, fnv1a64(src))));

    let hsaco: Vec<u8> = if let Some(ref p) = cache_path {
        if let Ok(bytes) = std::fs::read(p) {
            bytes
        } else {
            let fresh = ctx
                .runtime
                .hiprtc_compile_to_hsaco(src, entry)
                .unwrap_or_else(|e| panic!("rlx-rocm: hipRTC compile failed for {entry}: {e}"));
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let tmp = p.with_extension("hsaco.tmp");
            if std::fs::write(&tmp, &fresh).is_ok() {
                let _ = std::fs::rename(&tmp, p);
            }
            fresh
        }
    } else {
        ctx.runtime
            .hiprtc_compile_to_hsaco(src, entry)
            .unwrap_or_else(|e| panic!("rlx-rocm: hipRTC compile failed for {entry}: {e}"))
    };

    // Kernel-inspection hook: with `RLX_DUMP_KERNELS=<dir>` set, snapshot the
    // shared `.cu` source hipRTC compiled plus the arch-specific `.hsaco` code
    // object, so `tools/kernel-inspect/kinspect.py` can produce GCN ISA /
    // VGPR-SGPR-LDS / occupancy reports on the AMD rig. Off by default; the
    // twin of the rlx-cuda hook (NVRTC PTX there, hipRTC code object here).
    if let Some(dir) = rlx_ir::env::var("RLX_DUMP_KERNELS") {
        dump_kernel(&dir, entry, &arch, src, &hsaco);
    }

    match HipKernel::from_hsaco(&ctx.runtime, &hsaco, entry) {
        Ok(k) => {
            crate::hip::note_proven_arch(&arch);
            // Record the signature's arity here, where the source is still in
            // hand, so `launch_checked` can catch an argument-count mismatch
            // that the HIP driver cannot see.
            k.with_expected_params(rlx_gpu_kernels::declared_param_count(src, entry))
        }
        // `hipErrorNoBinaryForGpu` (209) means this device rejects a code
        // object built for `arch`. Arch resolution reads
        // `rocm_agent_enumerator` — an HSA tool listing GPUs in *physical*
        // order — while the object loads onto a HIP device, and on a
        // mixed-arch box those orderings need not agree. Rather than fail,
        // recompile for the other architectures the tools report and keep the
        // one the device accepts: a load that succeeds is proof of the arch,
        // which no amount of enumeration ordering can give us.
        Err(e) if e.to_string().contains("209") => {
            let mut tried = vec![arch.clone()];
            for cand in crate::hip::candidate_archs() {
                if tried.contains(&cand) {
                    continue;
                }
                tried.push(cand.clone());
                let Ok(blob) = ctx
                    .runtime
                    .hiprtc_compile_to_hsaco_for(src, entry, Some(&cand))
                else {
                    continue;
                };
                if let Ok(k) = HipKernel::from_hsaco(&ctx.runtime, &blob, entry) {
                    // Remember it so the remaining kernels compile correctly
                    // the first time — only this one pays the retry.
                    crate::hip::note_proven_arch(&cand);
                    eprintln!(
                        "rlx-rocm: code object for `{arch}` was rejected by this device \
                         (hipErrorNoBinaryForGpu); recompiled `{entry}` for `{cand}` and \
                         will use it for subsequent kernels. Set RLX_ROCM_ARCH={cand} to \
                         skip this probe."
                    );
                    if let Some(ref p) = cache_path {
                        // The cached blob is keyed by the wrong arch; drop it
                        // so the next process does not reload it.
                        let _ = std::fs::remove_file(p);
                    }
                    return k;
                }
            }
            panic!("{}", load_failure_message(entry, &arch, &e, &tried))
        }
        Err(e) => panic!(
            "{}",
            load_failure_message(entry, &arch, &e, std::slice::from_ref(&arch))
        ),
    }
}

/// Explain a `hipModuleLoadData` failure in terms of the arch mismatch that
/// almost always causes it.
///
/// `hipError(209)` is `hipErrorNoBinaryForGpu`: the code object was built for
/// one GPU architecture and handed to a device of another. On a mixed-arch box
/// (this project's AMD rig pairs a gfx908 MI100 with a gfx1103 780M) that is
/// easy to hit — set `RLX_ROCM_ARCH` to the wrong value, or point
/// `HIP_VISIBLE_DEVICES` at one GPU while the arch resolves from another — and
/// the bare `hipError(209)` gives no hint which knob is wrong. Standalone HIP
/// is even less forgiving here: a mismatched code object segfaults rather than
/// returning an error, so a clear message at this boundary is the only warning
/// anyone gets.
fn load_failure_message(
    entry: &str,
    arch: &str,
    err: &impl std::fmt::Display,
    tried: &[String],
) -> String {
    // `arch` is already resolved to "default" when hipRTC was given no
    // `--offload-arch`; say so rather than printing a bare literal.
    let built_for = if arch == "default" {
        "<hipRTC default arch>"
    } else {
        arch
    };
    let mut msg = format!(
        "rlx-rocm: hipModuleLoadData failed for kernel `{entry}` \
         (code object built for {built_for}): {err}"
    );
    if err.to_string().contains("209") {
        let visible = std::env::var("HIP_VISIBLE_DEVICES")
            .or_else(|_| std::env::var("ROCR_VISIBLE_DEVICES"))
            .unwrap_or_else(|_| "<unset → device 0>".to_string());
        msg.push_str(&format!(
            "\n  hipError 209 is hipErrorNoBinaryForGpu: this device does not accept a \
             {built_for} code object.\
             \n  Targeted device: HIP_VISIBLE_DEVICES={visible}\
             \n  Arch resolution order: RLX_ROCM_ARCH > HSA_OVERRIDE_GFX_VERSION > \
             rocm_agent_enumerator (see rlx_rocm::hip::rocm_target_arch).\
             \n  On a mixed-arch box, `rocm_agent_enumerator` lists GPUs in physical order; \
             the arch must match the device HIP_VISIBLE_DEVICES selects.\
             \n  Recompiled and retried for: {tried:?} — the device accepted none of them."
        ));
    }
    msg
}

/// Dump one hipRTC translation unit for offline kernel inspection. Writes
/// `<dir>/cu/<entry>.cu` (the shared source hipRTC compiled) and
/// `<dir>/codeobj/<entry>.hsaco` (the arch-specific AMD code object), and
/// appends one JSON line to `<dir>/manifest.jsonl`. `arch` (gfx908/gfx1100/…)
/// is recorded because an `.hsaco` is only meaningful for its target GPU.
/// Best-effort; dump failures never disturb the compile.
fn dump_kernel(dir: &str, entry: &str, arch: &str, src: &str, hsaco: &[u8]) {
    use std::io::Write;
    let base = std::path::Path::new(dir);
    let cu_dir = base.join("cu");
    let co_dir = base.join("codeobj");
    let _ = std::fs::create_dir_all(&cu_dir);
    let _ = std::fs::create_dir_all(&co_dir);
    let _ = std::fs::write(cu_dir.join(format!("{entry}.cu")), src);
    let _ = std::fs::write(co_dir.join(format!("{entry}.hsaco")), hsaco);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(base.join("manifest.jsonl"))
    {
        let _ = writeln!(
            f,
            "{{\"entry\":\"{entry}\",\"arch\":\"{arch}\",\"src_hash\":\"{:016x}\",\"src_bytes\":{},\"code_bytes\":{}}}",
            fnv1a64(src),
            src.len(),
            hsaco.len()
        );
    }
}

macro_rules! kernel_cache {
    ($static_name:ident, $fn_name:ident, $src:expr, $entry:expr) => {
        static $static_name: OnceLock<HipKernel> = OnceLock::new();
        pub fn $fn_name(ctx: &Arc<RocmContext>) -> &'static HipKernel {
            $static_name.get_or_init(|| compile(ctx, $src, $entry))
        }
    };
}

kernel_cache!(BINARY, binary_kernel, BINARY_CU, "binary");
kernel_cache!(GRU, gru_kernel, GRU_CU, "gru");
kernel_cache!(RNN, rnn_kernel, RNN_CU, "rnn");
kernel_cache!(MAMBA2, mamba2_kernel, MAMBA2_CU, "mamba2");
// On-device complex simulation (f32-lane): standalone complex cast + C64
// binary. Shared CUDA-C sources compiled via hipRTC (identical to rlx-cuda).
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
    MXFP4X2_DEQUANT,
    mxfp4x2_dequant_kernel,
    SCALED_LOWP_GENERAL_CU,
    "mxfp4x2_dequant"
);
kernel_cache!(
    UNARY,
    unary_kernel,
    rlx_gpu_kernels::unary_cuda_src(),
    "unary"
);
kernel_cache!(COPY, copy_kernel, COPY_CU, "copy");
kernel_cache!(
    MATMUL,
    matmul_kernel,
    rlx_gpu_kernels::matmul_cuda_src(),
    "matmul"
);
kernel_cache!(
    MATMUL_EPILOGUE,
    matmul_epilogue_kernel,
    rlx_gpu_kernels::matmul_epilogue_cuda_src(),
    "matmul_epilogue"
);
kernel_cache!(
    MATMUL_MFMA,
    matmul_mfma_kernel,
    MATMUL_MFMA_CU,
    "matmul_mfma"
);
kernel_cache!(
    GEMV_SPLITK,
    gemv_splitk_kernel,
    GEMV_SPLITK_CU,
    "gemv_splitk"
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
// Fused Q4_K GEMV (m == 1) from the SHARED gpu-kernels source rlx-cuda already
// uses. Without it every ROCm decode step dequantized each weight into f32
// scratch and ran a gemm — for Muse-Glimmer-30B that is ~111 GB of scratch
// traffic per token (measured ~10 s/token on an MI100).
kernel_cache!(
    DEQUANT_MATMUL_GGUF_Q4K_GEMV,
    dequant_matmul_gguf_q4k_gemv_kernel,
    DEQUANT_MATMUL_GGUF_CU,
    "dequant_matmul_gguf_q4k_gemv"
);
kernel_cache!(SAMPLE, sample_kernel, SAMPLE_CU, "sample");
// On-device Philox4×32-10 RNG (shared `rng_philox.cu`, bit-matched to
// `rlx_ir::Philox4x32`). Three entry points share one source; each gets its
// own hipRTC cache. Replaces the D2H→CPU→H2D `rng_host` bubble for the
// Philox / Zero policies (Ort / Bnns still host-fill).
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
kernel_cache!(
    SELECTIVE_SCAN,
    selective_scan_kernel,
    SELECTIVE_SCAN_CU,
    "selective_scan"
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
kernel_cache!(
    CONV_BIAS_ACT_EPILOGUE,
    conv_bias_act_epilogue_kernel,
    rlx_gpu_kernels::conv_bias_act_epilogue_cuda_src(),
    "conv_bias_act_epilogue"
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
kernel_cache!(
    GAUSSIAN_SPLAT_RASTERIZE,
    gaussian_splat_rasterize_kernel,
    GAUSSIAN_SPLAT_RASTERIZE_CU,
    "gaussian_splat_rasterize"
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

pub fn dispatch_grid_1d(n: u32, block_x: u32) -> (u32, u32) {
    (n.div_ceil(block_x), block_x)
}

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

/// AOT pre-warm: force-compile every kernel up-front. Mirrors
/// `rlx-cuda::backend::prewarm_all`.
pub fn prewarm_all(ctx: &Arc<RocmContext>) {
    let _ = binary_kernel(ctx);
    let _ = complex_cast_kernel(ctx);
    let _ = binary_c64_kernel(ctx);
    let _ = complex_norm_sq_kernel(ctx);
    let _ = complex_norm_sq_backward_kernel(ctx);
    let _ = conjugate_c64_kernel(ctx);
    let _ = fused_binary_unary_kernel(ctx);
    let _ = unary_kernel(ctx);
    let _ = copy_kernel(ctx);
    let _ = matmul_kernel(ctx);
    let _ = matmul_epilogue_kernel(ctx);
    let _ = compare_kernel(ctx);
    let _ = where_kernel(ctx);
    let _ = reduce_kernel(ctx);
    let _ = softmax_kernel(ctx);
    let _ = relu_backward_kernel(ctx);
    let _ = activation_backward_kernel(ctx);
    let _ = softmax_cross_entropy_kernel(ctx);
    let _ = softmax_cross_entropy_with_logits_kernel(ctx);
    let _ = softmax_cross_entropy_backward_kernel(ctx);
    let _ = layernorm_kernel(ctx);
    let _ = rms_norm_backward_kernel(ctx);
    let _ = rms_norm_bwd_zero_kernel(ctx);
    let _ = cumsum_backward_kernel(ctx);
    let _ = rope_backward_kernel(ctx);
    let _ = gather_backward_kernel(ctx);
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
    let _ = attention_warp_kernel(ctx);
    let _ = kv_append_kernel(ctx);
    let _ = attention_bwd_kernel(ctx);
    let _ = argmax_kernel(ctx);
    let _ = rope_kernel(ctx);
    let _ = cumsum_kernel(ctx);
    let _ = cum_scan_kernel(ctx);
    let _ = topk_kernel(ctx);
    let _ = grouped_matmul_kernel(ctx);
    let _ = scatter_add_zero_kernel(ctx);
    let _ = scatter_add_acc_kernel(ctx);
    let _ = dequant_matmul_kernel(ctx);
    let _ = dequant_matmul_mlx_kernel(ctx);
    let _ = dequant_gguf_kernel(ctx);
    let _ = sample_kernel(ctx);
    let _ = selective_scan_kernel(ctx);
    let _ = gru_kernel(ctx);
    let _ = rnn_kernel(ctx);
    let _ = mamba2_kernel(ctx);
    let _ = pool1d_kernel(ctx);
    let _ = pool2d_kernel(ctx);
    let _ = maxpool2d_backward_kernel(ctx);
    let _ = pool3d_kernel(ctx);
    let _ = conv1d_kernel(ctx);
    let _ = conv2d_kernel(ctx);
    let _ = conv2d_backward_input_kernel(ctx);
    let _ = conv2d_backward_weight_kernel(ctx);
    let _ = conv_bias_act_epilogue_kernel(ctx);
    let _ = im2col_kernel(ctx);
    let _ = conv3d_kernel(ctx);
    let _ = layer_norm2d_kernel(ctx);
    let _ = conv_transpose2d_kernel(ctx);
    let _ = conv_transpose3d_kernel(ctx);
    let _ = fused_swiglu_kernel(ctx);
    let _ = axial_rope2d_kernel(ctx);
    let _ = group_norm_kernel(ctx);
    let _ = group_norm_bwd_input_kernel(ctx);
    let _ = group_norm_bwd_gamma_kernel(ctx);
    let _ = group_norm_bwd_beta_kernel(ctx);
    let _ = batch_norm_inference_kernel(ctx);
    let _ = batch_norm_inference_bwd_input_kernel(ctx);
    let _ = batch_norm_inference_bwd_gamma_kernel(ctx);
    let _ = batch_norm_inference_bwd_beta_kernel(ctx);
    let _ = layer_norm_bwd_input_kernel(ctx);
    let _ = layer_norm_bwd_gamma_kernel(ctx);
    let _ = fake_quantize_fixed_kernel(ctx);
    let _ = fake_quantize_perbatch_kernel(ctx);
    let _ = fake_quantize_ema_kernel(ctx);
    let _ = fake_quantize_lsq_bwd_x_kernel(ctx);
    let _ = fake_quantize_lsq_bwd_scale_kernel(ctx);
    let _ = fake_quantize_backward_kernel(ctx);
    let _ = quantize_i8_kernel(ctx);
    let _ = dequantize_i8_kernel(ctx);
    let _ = q_matmul_kernel(ctx);
    let _ = q_conv2d_kernel(ctx);
    let _ = resize_nearest_2x_kernel(ctx);
    let _ = elementwise_region_kernel(ctx);
    let _ = batch_elementwise_region_kernel(ctx);
    let _ = fma_kernel(ctx);
    let _ = fft_radix2_full_kernel(ctx);
    let _ = fft_bit_reverse_kernel(ctx);
    let _ = fft_inner_kernel(ctx);
    let _ = fft_outer_r4_kernel(ctx);
    let _ = fft_outer_r2_kernel(ctx);
    let _ = gaussian_splat_rasterize_kernel(ctx);
    let _ = welch_peaks_gpu_kernel(ctx);
    let _ = fft_butterfly_stage_kernel(ctx);
}
