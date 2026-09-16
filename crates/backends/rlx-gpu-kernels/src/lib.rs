// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared GPU kernel sources for RLX CUDA and ROCm backends.
//!
//! Each constant is the full `.cu` source text, embedded at compile time.
//! Backends JIT-compile via NVRTC / hipRTC on first use.
//!
//! Because the sources are shared, so are the decisions *about* them — but the
//! decisions live one crate down, in `rlx-gpu-dispatch`, so Metal and wgpu can
//! use the same routing policy without linking any of this CUDA/HIP text.
//! [`tiles`] owns the tile-schedule parameter space and its legality rules;
//! [`dispatch`] owns the shape-keyed table. Both are re-exported here because
//! the CUDA-side source templating below is their main consumer.

// The table and the tile parameter space moved to `rlx-gpu-dispatch` so Metal
// and wgpu can share them without depending on these CUDA/HIP sources. Re-export
// under the old paths — this crate is still where the CUDA-side templating
// (`matmul_cuda_src_tiled`) lives, so `rlx_gpu_kernels::tiles` remains the
// natural spelling at those call sites.
pub use rlx_gpu_dispatch::{cost, dispatch, tiles};

pub mod kernel_schedule_port;

/// Generate the `matmul` entry point *from* a typed schedule rather than
/// templating `kernels/matmul.cu`.
///
/// Feature-gated because it is a second implementation of a shipping kernel:
/// until it is measured at least as fast on a given target, the hand-written
/// text stays the default and this is opt-in.
#[cfg(feature = "schedule-codegen")]
pub mod kernel_schedule_emit;

pub const BINARY_CU: &str = include_str!(concat!(env!("OUT_DIR"), "/binary.cu"));
/// Standalone complex `Op::Cast` on the f32-uniform arena (real<->C64,
/// real<->C128, C64<->C128 — six pure lane-move modes). f32-uniform GPU
/// backends simulate complex as interleaved f32 lanes; this re-pairs them.
pub const COMPLEX_CAST_CU: &str = include_str!("../kernels/complex_cast.cu");
/// Element-wise C64 binary op (add/sub/mul/div) reading both `[re, im]`
/// lanes per element, with modulo broadcast. Mirrors rlx-cpu
/// `exec_binary_full_c64`; C128 arithmetic + C64 max/min/pow are rejected.
pub const BINARY_C64_CU: &str = include_str!("../kernels/binary_c64.cu");
/// C64 Wirtinger surface: `ComplexNormSq` / `ComplexNormSqBackward` /
/// `Conjugate` on interleaved `[re, im]` f32 pairs. Mirrors rlx-cpu
/// `exec_complex_norm_sq{,_backward}_f32` / `exec_conjugate_c64`.
pub const COMPLEX_WIRINGER_CU: &str = include_str!("../kernels/complex_wirtinger.cu");
pub const FUSED_BINARY_UNARY_CU: &str = include_str!("../kernels/fused_binary_unary.cu");
pub const CAST_F32_TO_HALF_CU: &str = include_str!("../kernels/cast_f32_to_half.cu");
/// Native FP8 quantize producers (per-tensor scale + E4M3/E5M2 encode) for
/// `Op::ScaledMatMul`. Shared by the CUDA (cublasLt) and ROCm (hipBLASLt) paths.
pub const SCALED_LOWP_CU: &str = include_str!("../kernels/scaled_lowp.cu");
/// General (all-format, all-scale-layout) low-precision quantize + decode-GEMM
/// for `Op::ScaledMatMul` — the on-device decode-and-accumulate fallback for
/// block-scaled / FP4 / FP6 configs the FP8 tensor-core path can't do.
pub const SCALED_LOWP_GENERAL_CU: &str = include_str!("../kernels/scaled_lowp_general.cu");
/// Unary/activation + cast kernel. The activation dispatch (op 0..28) is
/// @generated from the shared `rlxsl` manifest and prepended to the hand-written
/// plumbing + cast selectors (`kernels/unary_main.cu`) at build time. Self-
/// contained (the generated `rlx_activation_apply` inlines gelu), so — unlike
/// the other kernels — it needs no `gelu.cuh` prepend.
pub const UNARY_CU: &str = include_str!(concat!(env!("OUT_DIR"), "/unary.cu"));
pub const LSTM_CU: &str = include_str!("../kernels/lstm.cu");
/// Philox4×32-10 RNG fill (`rng_normal_philox` / `rng_uniform_philox` / `rng_fill_zero`).
pub const RNG_PHILOX_CU: &str = include_str!("../kernels/rng_philox.cu");
/// Single-layer unidirectional GRU (`gru`). Gate order r/z/n; separate b_ih/b_hh.
pub const GRU_CU: &str = include_str!("../kernels/gru.cu");
/// Single-layer unidirectional Elman RNN (`rnn`). `relu_flag` selects relu vs tanh.
pub const RNN_CU: &str = include_str!("../kernels/rnn.cu");
/// Mamba-2 / SSD scalar-decay scan (`mamba2`). `state_size ≤ 256`.
pub const MAMBA2_CU: &str = include_str!("../kernels/mamba2.cu");
pub const BINARY_BROADCAST_CU: &str = include_str!("../kernels/binary_broadcast.cu");
pub const COPY_CU: &str = include_str!("../kernels/copy.cu");
pub const PAD_CU: &str = include_str!("../kernels/pad.cu");
pub const SLICE_CU: &str = include_str!("../kernels/slice.cu");
pub const MATMUL_CU: &str = include_str!("../kernels/matmul.cu");
pub const MATMUL_BT_CU: &str = include_str!("../kernels/matmul_bt.cu");
pub const MATMUL_EPILOGUE_CU: &str = include_str!("../kernels/matmul_epilogue.cu");
pub const MATMUL_WMMA_CU: &str = include_str!("../kernels/matmul_wmma.cu");
/// Hopper (sm_90) TMA-staged fp32 GEMM. Bulk-copies A/B tiles global->shared
/// via the Tensor Memory Accelerator + mbarrier, then register-blocked FMA.
/// Compiled only under `compute_90a`; opt-in via `RLX_CUDA_TMA` (CUDA-only).
pub const MATMUL_TMA_CU: &str = include_str!("../kernels/matmul_tma.cu");
/// TMA-staged NT GEMM `C = A·Wᵀ` (transposed-B twin of `MATMUL_TMA_CU`) for the
/// GGUF prefill path's post-dequant matmul. Same Hopper-only constraints.
pub const MATMUL_BT_TMA_CU: &str = include_str!("../kernels/matmul_bt_tma.cu");
pub const COMPARE_CU: &str = include_str!(concat!(env!("OUT_DIR"), "/compare.cu"));
pub const WHERE_CU: &str = include_str!("../kernels/where_select.cu");
pub const FMA_CU: &str = include_str!("../kernels/fma.cu");
pub const REDUCE_CU: &str = include_str!("../kernels/reduce.cu");
pub const SOFTMAX_CU: &str = include_str!("../kernels/softmax.cu");
/// Activation backward (`dx = act'(x)·dy`). The derivative switch (op 0..17) is
/// @generated from the rlxsl manifest — auto-differentiated from the forward, so
/// it is exactly the gradient of the forward we ship — and prepended to the
/// plumbing in `kernels/activation_backward_main.cu` at build time.
pub const ACTIVATION_BACKWARD_CU: &str =
    include_str!(concat!(env!("OUT_DIR"), "/activation_backward.cu"));
pub const SOFTMAX_CROSS_ENTROPY_CU: &str = include_str!("../kernels/softmax_cross_entropy.cu");
pub const LAYERNORM_CU: &str = include_str!("../kernels/layernorm.cu");
pub const LAYER_NORM_BWD_CU: &str = include_str!("../kernels/layer_norm_backward.cu");
pub const RMS_NORM_BWD_CU: &str = include_str!("../kernels/rms_norm_backward.cu");
pub const FAKE_QUANTIZE_CU: &str = include_str!("../kernels/fake_quantize.cu");
/// INT8 asymmetric Quantize / Dequantize (`quantize_i8` / `dequantize_i8`).
pub const QUANTIZE_CU: &str = include_str!("../kernels/quantize.cu");
/// Real INT8 `Op::QMatMul` (`q_matmul`) — packed i8 x/w/out + f32-lane bias.
pub const Q_MATMUL_CU: &str = include_str!("../kernels/q_matmul.cu");
/// Real INT8 `Op::QConv2d` (`q_conv2d`) — NCHW packed i8 + f32-lane bias.
pub const Q_CONV2D_CU: &str = include_str!("../kernels/q_conv2d.cu");
pub const CUMSUM_BWD_CU: &str = include_str!("../kernels/cumsum_backward.cu");
pub const ROPE_BWD_CU: &str = include_str!("../kernels/rope_backward.cu");
pub const GATHER_BWD_CU: &str = include_str!("../kernels/gather_backward.cu");
pub const FUSED_RESIDUAL_LN_CU: &str = include_str!("../kernels/fused_residual_ln.cu");
pub const FUSED_RESIDUAL_RMS_NORM_CU: &str = include_str!("../kernels/fused_residual_rms_norm.cu");
pub const ADA_LAYER_NORM_CU: &str = include_str!("../kernels/ada_layer_norm.cu");
pub const GATED_RESIDUAL_CU: &str = include_str!("../kernels/gated_residual.cu");
pub const ADA_LAYER_NORM_BACKWARD_CU: &str = include_str!("../kernels/ada_layer_norm_backward.cu");
pub const GATED_RESIDUAL_BACKWARD_CU: &str = include_str!("../kernels/gated_residual_backward.cu");
pub const GATHER_CU: &str = include_str!("../kernels/gather.cu");
pub const GATHER_AXIS_CU: &str = include_str!("../kernels/gather_axis.cu");
pub const NARROW_CU: &str = include_str!("../kernels/narrow.cu");
pub const CONCAT_CU: &str = include_str!("../kernels/concat.cu");
pub const TRANSPOSE_CU: &str = include_str!("../kernels/transpose.cu");
pub const EXPAND_CU: &str = include_str!("../kernels/expand.cu");
pub const ATTENTION_CU: &str = include_str!("../kernels/attention.cu");
/// Tensor-Core (fp16 WMMA) FlashAttention — a CUDA-only drop-in for `attention`
/// (same signature/entry `attention_wmma`). Uses `nvcuda::wmma`, so it is only
/// ever NVRTC-compiled by rlx-cuda; rlx-rocm never references it (this const is
/// just embedded text, harmless in the shared crate).
pub const ATTENTION_WMMA_CU: &str = include_str!("../kernels/attention_wmma.cu");
pub const FUSED_ATTN_CU: &str = include_str!("../kernels/fused_attn.cu");
pub const ATTENTION_ROW_CU: &str = include_str!("../kernels/attention_row.cu");
/// In-place KV append on the f32 arena. Output aliases input 0 via the shared
/// memory planner, so this grows the resident cache on-device instead of
/// re-uploading the padded cache per token.
pub const KV_APPEND_CU: &str = include_str!("../kernels/kv_append.cu");
/// Warp-per-row SDPA — drop-in for [`ATTENTION_ROW_CU`] (identical parameter
/// list). One 32-lane group per (batch, head, q_row) instead of one thread, so
/// the head-dim accumulators stay in registers instead of spilling to local
/// memory, and decode (`seq_q == 1`) gets 32x the lanes per row.
pub const ATTENTION_WARP_CU: &str = include_str!("../kernels/attention_warp.cu");
pub const ATTENTION_BWD_CU: &str = include_str!("../kernels/attention_bwd.cu");
pub const ARGMAX_CU: &str = include_str!("../kernels/argmax.cu");
pub const ROPE_CU: &str = include_str!("../kernels/rope.cu");
pub const CUMSUM_CU: &str = include_str!("../kernels/cumsum.cu");
pub const CUM_SCAN_CU: &str = include_str!("../kernels/cum_scan.cu");
pub const TOPK_CU: &str = include_str!("../kernels/topk.cu");
pub const GROUPED_MATMUL_CU: &str = include_str!("../kernels/grouped_matmul.cu");
pub const SCATTER_ADD_CU: &str = include_str!("../kernels/scatter_add.cu");
/// ONNX ScatterND (reduction=none) on the f32-uniform arena.
pub const SCATTER_ND_CU: &str = include_str!("../kernels/scatter_nd.cu");
/// ONNX GatherND / GatherElements / ScatterElements, plus a rank-general
/// reduction-capable ScatterND, on the f32-uniform arena. These four used to
/// take `Step::CpuIndexing` (a full D2H/H2D round trip) on every arena backend;
/// shapes and strides ride in a `meta` u32 buffer planned by
/// [`rlx_gpu_dispatch::indexing`], so there is no rank cap and no per-launch
/// allocation.
pub const INDEXING_ND_CU: &str = include_str!("../kernels/indexing_nd.cu");
pub const DEQUANT_MATMUL_CU: &str = include_str!("../kernels/dequant_matmul.cu");
pub const DEQUANT_GGUF_CU: &str = include_str!("../kernels/dequant_gguf.cu");
pub const DEQUANT_MATMUL_GGUF_CU: &str = include_str!("../kernels/dequant_matmul_gguf.cu");
/// MLX affine / mxfp4 / mxfp8 fused dequant-matmul (`[n,k]` pack along K).
pub const DEQUANT_MATMUL_MLX_CU: &str = include_str!("../kernels/dequant_matmul_mlx.cu");
pub const SAMPLE_CU: &str = include_str!("../kernels/sample.cu");
pub const SELECTIVE_SCAN_CU: &str = include_str!("../kernels/selective_scan.cu");
pub const GATED_DELTA_NET_CU: &str = include_str!("../kernels/gated_delta_net.cu");
/// FlashKDA-style chunked-parallel gated-delta-net (Kimi Delta Attention).
pub const KIMI_DELTA_CHUNK_CU: &str = include_str!("../kernels/kimi_delta_chunk.cu");
pub const POOL1D_CU: &str = include_str!("../kernels/pool1d.cu");
pub const POOL2D_CU: &str = include_str!("../kernels/pool2d.cu");
pub const POOL3D_CU: &str = include_str!("../kernels/pool3d.cu");
pub const MAXPOOL2D_BACKWARD_CU: &str = include_str!("../kernels/maxpool2d_backward.cu");
pub const MAXPOOL3D_BACKWARD_CU: &str = include_str!("../kernels/maxpool3d_backward.cu");
pub const CONV1D_CU: &str = include_str!("../kernels/conv1d.cu");
pub const CONV2D_CU: &str = include_str!("../kernels/conv2d.cu");
pub const CONV2D_BACKWARD_INPUT_CU: &str = include_str!("../kernels/conv2d_backward_input.cu");
pub const CONV2D_BACKWARD_WEIGHT_CU: &str = include_str!("../kernels/conv2d_backward_weight.cu");
pub const IM2COL_CU: &str = include_str!("../kernels/im2col.cu");
pub const CONV3D_CU: &str = include_str!("../kernels/conv3d.cu");
pub const CONV3D_BACKWARD_INPUT_CU: &str = include_str!("../kernels/conv3d_backward_input.cu");
pub const CONV3D_BACKWARD_WEIGHT_CU: &str = include_str!("../kernels/conv3d_backward_weight.cu");
pub const CONV_TRANSPOSE3D_CU: &str = include_str!("../kernels/conv_transpose3d.cu");
pub const LAYER_NORM2D_CU: &str = include_str!("../kernels/layer_norm2d.cu");
pub const CONV_TRANSPOSE2D_CU: &str = include_str!("../kernels/conv_transpose2d.cu");
pub const FUSED_SWIGLU_CU: &str = include_str!("../kernels/fused_swiglu.cu");
pub const AXIAL_ROPE2D_CU: &str = include_str!("../kernels/axial_rope2d.cu");
pub const GROUP_NORM_CU: &str = include_str!("../kernels/group_norm.cu");
pub const GROUP_NORM_BWD_CU: &str = include_str!("../kernels/group_norm_backward.cu");
pub const BATCH_NORM_INFERENCE_CU: &str = include_str!("../kernels/batch_norm_inference.cu");
pub const RESIZE_NEAREST_2X_CU: &str = include_str!("../kernels/resize_nearest_2x.cu");
pub const INTERPOLATE3D_CU: &str = include_str!("../kernels/interpolate3d.cu");
pub const ELEMENTWISE_REGION_CU: &str = include_str!("../kernels/elementwise_region.cu");
pub const BATCH_ELEMENTWISE_REGION_CU: &str =
    include_str!("../kernels/batch_elementwise_region.cu");
pub const GAUSSIAN_SPLAT_RASTERIZE_CU: &str =
    include_str!("../kernels/gaussian_splat_rasterize.cu");
pub const FFT_CU: &str = include_str!("../kernels/fft.cu");
pub const FFT_BUTTERFLY_STAGE_CU: &str = include_str!("../kernels/fft_butterfly_stage.cu");
pub const WELCH_PEAKS_CU: &str = include_str!("../kernels/welch_peaks.cu");

const GELU_CUH: &str = include_str!("../kernels/gelu.cuh");

use std::sync::OnceLock;

macro_rules! cuda_src_with_gelu {
    ($name:ident, $body:expr) => {
        pub fn $name() -> &'static str {
            static S: OnceLock<String> = OnceLock::new();
            S.get_or_init(|| format!("{GELU_CUH}\n{}", $body))
        }
    };
}

/// Unary/activation + cast kernel source. Already self-contained (activation
/// dispatch is @generated with gelu inlined), so no `gelu.cuh` prepend.
pub fn unary_cuda_src() -> &'static str {
    UNARY_CU
}
cuda_src_with_gelu!(
    fused_binary_unary_cuda_src,
    include_str!("../kernels/fused_binary_unary.cu")
);
cuda_src_with_gelu!(matmul_cuda_src, include_str!("../kernels/matmul.cu"));

/// `matmul.cu` compiled for an explicit tile schedule.
///
/// The tile is pinned by a `#define` prelude ahead of the kernel body, whose
/// own `#ifndef` defaults then fall away. For [`tiles::TileParams::DEFAULT_MATMUL`]
/// the prelude is empty and this returns exactly [`matmul_cuda_src`] — same
/// bytes, so the same NVRTC/hipRTC module and the same disk-cache slot as before
/// the tile became a parameter.
///
/// Returns `Err` for a tile that violates a kernel invariant, so an illegal
/// schedule is refused on the host rather than silently reading out of bounds on
/// the device. Callers cache the resulting `String` per tile — the backends' JIT
/// caches already key on the source hash, so distinct tiles never collide.
pub fn matmul_cuda_src_tiled(tile: tiles::TileParams) -> Result<String, tiles::TileError> {
    tile.validate()?;
    // The tile's typed schedule must verify before any source is emitted.
    //
    // This is where `rlx_ir::kernel_schedule` earns its place: `validate()` checks the
    // algebra the kernel body's indexing relies on, and the schedule checks
    // what that algebra cannot see — that the shared tiles fit a portable
    // budget, that the two `__syncthreads()` order the staging correctly, and
    // that no two accesses to a tile disagree about its layout. A tile that
    // fails here would have compiled and produced wrong numbers.
    kernel_schedule_port::verify_tile_schedule(tile).map_err(|errors| {
        tiles::TileError::UnsoundSchedule(
            errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; "),
        )
    })?;
    let defines = tile.defines();
    if defines.is_empty() {
        return Ok(matmul_cuda_src().to_string());
    }
    Ok(format!("{defines}{}", matmul_cuda_src()))
}

/// Which physical schedule the `matmul` entry should be built from.
///
/// This is the knob the A/B experiment turns. `Default` is the shipping
/// hand-written text; the others are *generated* from a `KernelSchedule`, so
/// the difference between them is a declaration, not a second kernel file.
#[cfg(feature = "schedule-codegen")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MatmulSchedule {
    /// `kernels/matmul.cu` templated by `#define`. The baseline.
    #[default]
    Default,
    /// Emitted from the schedule that *describes* `matmul.cu`: one stage, two
    /// block syncs per K iteration. Exists to separate "the emitter is correct
    /// and costs nothing" from "the pipeline is faster" — without it a win
    /// cannot be attributed to the schedule rather than to codegen luck.
    EmittedSerial,
    /// Emitted from [`kernel_schedule_emit::matmul_pipelined_schedule`]:
    /// `stages`-deep rotation, `cp.async` staging, one block sync per K
    /// iteration.
    EmittedPipelined { stages: usize },
}

#[cfg(feature = "schedule-codegen")]
impl MatmulSchedule {
    /// Parse `default | serial | pipelined[:N]`.
    ///
    /// Shared by every backend that exposes this knob, so CUDA and ROCm cannot
    /// drift into accepting different spellings of the same experiment. An
    /// unrecognized value is an `Err` carrying the offending text rather than a
    /// silent fall-through to `Default`: a typo that quietly runs the baseline
    /// makes an A/B measure one path twice and report it as parity, which is
    /// the `metal-variant-typo-silent` defect this tree already shipped once.
    pub fn parse(value: &str) -> Result<Self, String> {
        let v = value.trim();
        if v.eq_ignore_ascii_case("default") || v == "0" {
            return Ok(Self::Default);
        }
        if v.eq_ignore_ascii_case("serial") {
            return Ok(Self::EmittedSerial);
        }
        if v.to_ascii_lowercase().starts_with("pipelined") {
            let stages = v
                .split([':', '='])
                .nth(1)
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(2);
            return Ok(Self::EmittedPipelined {
                stages: stages.max(2),
            });
        }
        Err(format!(
            "{value:?} is not one of default|serial|pipelined[:N]"
        ))
    }

    /// Short stable label for reports and cache keys.
    pub fn label(&self) -> String {
        match self {
            Self::Default => "default".into(),
            Self::EmittedSerial => "serial".into(),
            Self::EmittedPipelined { stages } => format!("pipe:{stages}"),
        }
    }
}

/// Build the `matmul` CUDA source for `tile` under `schedule`.
///
/// The emitted variants carry the same entry name and parameter list as
/// `matmul.cu`, so every caller and launcher is unchanged and the experiment
/// varies one thing.
#[cfg(feature = "schedule-codegen")]
pub fn matmul_cuda_src_scheduled(
    tile: tiles::TileParams,
    schedule: MatmulSchedule,
    target: rlx_ir::kernel_schedule::Target,
) -> Result<(String, Option<kernel_schedule_emit::EmitFacts>), String> {
    matmul_src_scheduled(tile, schedule, target, kernel_schedule_emit::Lang::Cuda)
}

/// Build the `matmul` source for `tile` under `schedule`, in `lang`.
///
/// The `.cu` text is shared between NVRTC and hipRTC, so this is too — but only
/// the portable subset is genuinely shared, and `lang` is what makes that a
/// typed rejection instead of a hipRTC error on inline PTX. See
/// [`kernel_schedule_emit::Lang`].
///
/// ROCm reaches this through the same tile parameter space and the same
/// dispatch table as CUDA, which is the whole reason the kernel text lives in
/// this crate rather than in either backend.
#[cfg(feature = "schedule-codegen")]
pub fn matmul_src_scheduled(
    tile: tiles::TileParams,
    schedule: MatmulSchedule,
    target: rlx_ir::kernel_schedule::Target,
    lang: kernel_schedule_emit::Lang,
) -> Result<(String, Option<kernel_schedule_emit::EmitFacts>), String> {
    let sched = match schedule {
        MatmulSchedule::Default => {
            return matmul_cuda_src_tiled(tile)
                .map(|s| (s, None))
                .map_err(|e| e.to_string());
        }
        MatmulSchedule::EmittedSerial => kernel_schedule_port::matmul_schedule(tile),
        MatmulSchedule::EmittedPipelined { stages } => {
            let mut s = kernel_schedule_emit::matmul_pipelined_schedule(tile, stages);
            // HIP gets the rotation without the async-copy declaration. This is
            // NOT the emitter quietly downgrading a capability — the schedule
            // is a different one, it says so in its `requires`, and the
            // resulting `EmitFacts.async_copy` is `false` so a report cannot
            // claim an async pipeline that was never emitted.
            if lang == kernel_schedule_emit::Lang::Hip {
                s.requires.clear();
            }
            s
        }
    };
    let (body, facts) =
        kernel_schedule_emit::emit(&sched, tile, target, lang).map_err(|e| e.to_string())?;
    // `gelu_erf` / `gelu_approx` come from the shared header, exactly as they do
    // for the hand-written kernel.
    Ok((format!("{GELU_CUH}\n{body}"), Some(facts)))
}

cuda_src_with_gelu!(
    matmul_epilogue_cuda_src,
    include_str!("../kernels/matmul_epilogue.cu")
);
cuda_src_with_gelu!(
    conv_bias_act_epilogue_cuda_src,
    include_str!("../kernels/conv_bias_act_epilogue.cu")
);
cuda_src_with_gelu!(
    elementwise_region_cuda_src,
    include_str!("../kernels/elementwise_region.cu")
);
cuda_src_with_gelu!(
    batch_elementwise_region_cuda_src,
    include_str!("../kernels/batch_elementwise_region.cu")
);

/// AMD rocWMMA / MFMA matmul (`RLX_ROCM_MFMA=1`). Not used on CUDA.
#[cfg(feature = "rocm")]
pub mod rocm {
    pub const MATMUL_MFMA_CU: &str = include_str!("../kernels/rocm/matmul_mfma.cu");
    /// Skinny-m split-K GEMV — better fallback than the tiled GEMM when a vendor
    /// BLAS is unavailable and m is small (decode under-occupies the CU array).
    pub const GEMV_SPLITK_CU: &str = include_str!("../kernels/rocm/gemv_splitk.cu");
}

/// Replace `//…` and `/*…*/` with spaces, preserving byte offsets is not
/// required here — only that no comment text survives to be parsed.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
            out.push(' ');
        } else {
            // Multi-byte UTF-8 (comments carry λ, ∥, ²) must not be split.
            let ch_len = src[i..].chars().next().map_or(1, |c| c.len_utf8());
            out.push_str(&src[i..i + ch_len]);
            i += ch_len;
        }
    }
    out
}

/// Number of parameters `entry`'s `__global__` signature declares in `src`.
///
/// CUDA and HIP both launch through a bare `void**` whose length the driver
/// never sees: `cuLaunchKernel`/`hipModuleLaunchKernel` read exactly as many
/// pointers as the compiled kernel expects, so passing too few reads past the
/// end of the caller's array and passing too many silently ignores the tail.
/// Neither is a compile error, and neither reliably faults — the usual symptom
/// is a kernel that runs and writes nothing, or writes garbage, which is the
/// same class of defect that made the Metal ICB path silently produce zeros.
///
/// The kernels are plain C (no templates, no default arguments), so counting
/// top-level commas is exact. Returns `None` when the entry is not found —
/// callers treat that as "cannot check" rather than as a mismatch.
pub fn declared_param_count(src: &str, entry: &str) -> Option<usize> {
    // Comments first. Real signatures document each parameter inline, and those
    // comments contain both commas and parentheses — e.g.
    // `const float* eigvec, // [batch,n,n] col-major` and `[λ(n) ∥ U(n²)]` —
    // which a naive comma count and paren-depth scan both swallow, yielding 9
    // for a 5-parameter kernel. That is a false positive, and a checker that
    // cries wolf is worse than none.
    let src = &strip_comments(src);
    // Find `__global__ ... <entry> (`, not merely the name, so a call site or a
    // forward declaration elsewhere in the file cannot be mistaken for it.
    let mut from = 0usize;
    let open = loop {
        let g = src[from..].find("__global__")? + from;
        let after = &src[g..];
        let paren_rel = after.find('(')?;
        let head = &after[..paren_rel];
        // The token immediately before `(` must be the entry name.
        let name = head
            .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
            .next()
            .unwrap_or("");
        if name == entry {
            break g + paren_rel;
        }
        from = g + "__global__".len();
    };

    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut commas = 0usize;
    let mut any = false;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    // Empty parameter list is zero, not one.
                    let inner = src[open + 1..i].trim();
                    return Some(if inner.is_empty() || inner == "void" {
                        0
                    } else {
                        commas + 1
                    });
                }
            }
            b',' if depth == 1 => {
                commas += 1;
                any = true;
            }
            b if !b.is_ascii_whitespace() => any = any || depth == 1,
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tiled_src_tests {
    use super::*;
    use crate::tiles::{MATMUL_TILE_CANDIDATES, TileParams};

    /// The default tile must produce the *same bytes* as before the tile became
    /// a parameter — same JIT module, same disk-cache slot, same SASS.
    #[test]
    fn default_tile_source_is_unchanged() {
        let s = matmul_cuda_src_tiled(TileParams::DEFAULT_MATMUL).expect("default is legal");
        assert_eq!(s, matmul_cuda_src());
    }

    /// Every macro the kernel guards with `#ifndef` must be pinned by the
    /// prelude, and the prelude must come first so the guards see it.
    #[test]
    fn non_default_tile_pins_every_guarded_macro_before_the_body() {
        let t = MATMUL_TILE_CANDIDATES[0];
        assert_ne!(t, TileParams::DEFAULT_MATMUL);
        let s = matmul_cuda_src_tiled(t).expect("candidate is legal");
        for (m, v) in [
            ("BM", t.bm),
            ("BN", t.bn),
            ("BK", t.bk),
            ("TM", t.tm),
            ("TN", t.tn),
            ("BLOCK_DIM_X", t.bdx),
            ("BLOCK_DIM_Y", t.bdy),
        ] {
            let pin = format!("#define {m} {v}\n");
            let at = s
                .find(&pin)
                .unwrap_or_else(|| panic!("no `{pin:?}` in source"));
            let guard = s
                .find(&format!("#ifndef {m}"))
                .unwrap_or_else(|| panic!("kernel lost its `#ifndef {m}` guard"));
            assert!(at < guard, "`{m}` pinned after its own #ifndef guard");
        }
    }

    /// Distinct tiles must produce distinct source, or they would share a JIT
    /// cache slot and the second one would silently run the first one's module.
    #[test]
    fn distinct_tiles_produce_distinct_source() {
        let mut seen = std::collections::HashSet::new();
        for t in MATMUL_TILE_CANDIDATES {
            let s = matmul_cuda_src_tiled(*t).expect("candidate is legal");
            assert!(seen.insert(s), "source collision for {t:?}");
        }
    }

    /// An illegal tile is refused at source-generation time.
    #[test]
    fn illegal_tile_is_refused() {
        let bad = TileParams {
            bk: 15,
            ..TileParams::DEFAULT_MATMUL
        };
        assert!(matmul_cuda_src_tiled(bad).is_err());
    }

    /// The kernel body must not have re-acquired a hardcoded tile assumption:
    /// each guarded macro appears exactly once as an unconditional `#define`
    /// (inside its own `#ifndef`).
    #[test]
    fn kernel_body_defines_each_tile_macro_exactly_once() {
        let src = include_str!("../kernels/matmul.cu");
        // Directive lines only — prose in the header comment mentions these
        // macros too, and a substring scan would count it.
        let directives = |kind: &str, m: &str| {
            let want = format!("{kind} {m}");
            src.lines()
                .map(str::trim)
                .filter(|l| *l == want || l.starts_with(&format!("{want} ")))
                .count()
        };
        for m in ["BM", "BN", "BK", "TM", "TN", "BLOCK_DIM_X", "BLOCK_DIM_Y"] {
            let n = directives("#define", m);
            assert_eq!(n, 1, "`{m}` defined {n}× in matmul.cu");
            assert_eq!(directives("#ifndef", m), 1, "`{m}` is not #ifndef-guarded");
        }
    }
}

#[cfg(test)]
mod param_count_tests {
    use super::declared_param_count;

    #[test]
    fn counts_plain_signatures() {
        let src = "__global__ void foo(float* a, int n) {}\n\
                   __global__ void bar(float* a) {}\n\
                   __global__ void baz() {}\n";
        assert_eq!(declared_param_count(src, "foo"), Some(2));
        assert_eq!(declared_param_count(src, "bar"), Some(1));
        assert_eq!(declared_param_count(src, "baz"), Some(0));
        assert_eq!(declared_param_count(src, "missing"), None);
    }

    /// A name that is a suffix of another must not match it, and a call to the
    /// kernel elsewhere in the file must not be mistaken for its declaration.
    #[test]
    fn does_not_confuse_similar_names_or_call_sites() {
        let src = "__global__ void my_kernel(float* a, int n, int m) {}\n\
                   __global__ void kernel(float* a) {}\n\
                   void host() { my_kernel<<<1,1>>>(nullptr, 0, 0); }\n";
        assert_eq!(declared_param_count(src, "my_kernel"), Some(3));
        assert_eq!(declared_param_count(src, "kernel"), Some(1));
    }

    /// Inline per-parameter comments carry commas and parentheses, and a naive
    /// scan counts them. This is `eigh_assemble` verbatim: 5 parameters, but
    /// the comments hold 4 extra commas and unbalanced parens, which read as 9
    /// and produced a false "argument count mismatch" on the ROCm rig.
    #[test]
    fn ignores_commas_and_parens_inside_comments() {
        let src = "extern \"C\" __global__ void eigh_assemble(\n\
             const float* __restrict__ eigvec, // [batch,n,n] col-major: eigvec[b*n*n + i + j*n] = comp i of eigvec j\n\
             const float* __restrict__ eigval, // [batch,n] ascending\n\
             float* __restrict__ out,          // [batch, n*n+n] packed: [\u{3bb}(n) \u{2225} U(n\u{b2}) row-major]\n\
             int n, int batch)\n{\n}\n";
        assert_eq!(declared_param_count(src, "eigh_assemble"), Some(5));
    }

    /// Block comments too, including one that opens a paren it never closes.
    #[test]
    fn ignores_block_comments() {
        let src = "__global__ void k(float* a, /* shape (n, m) */ int n) {}\n";
        assert_eq!(declared_param_count(src, "k"), Some(2));
    }

    /// Real signatures wrap across lines and use qualifiers.
    #[test]
    fn handles_multiline_and_qualifiers() {
        let src = "extern \"C\" __global__ void wide(\n    const float* __restrict__ a,\n\
                   \n    float* out,\n    unsigned int n)\n{\n}\n";
        assert_eq!(declared_param_count(src, "wide"), Some(3));
    }
}
