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

pub(crate) fn compile(ctx: &Arc<CudaContext>, src: &str, entry: &str) -> CudaKernel {
    // Try the disk cache first. The cache key folds the kernel entry
    // name into the source hash so different entry-points sharing a
    // .cu file (scatter_add_zero / scatter_add_acc) get distinct
    // cache slots.
    let cache_path =
        ptx_cache_dir().map(|d| d.join(format!("{}-{:016x}.ptx", entry, fnv1a64(src))));

    let ptx = if let Some(ref p) = cache_path {
        if let Ok(cached) = std::fs::read_to_string(p) {
            cudarc::nvrtc::Ptx::from_src(cached)
        } else {
            let fresh = cudarc::nvrtc::compile_ptx(src)
                .unwrap_or_else(|e| panic!("rlx-cuda: NVRTC compile failed for {entry}: {e}"));
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
        cudarc::nvrtc::compile_ptx(src)
            .unwrap_or_else(|e| panic!("rlx-cuda: NVRTC compile failed for {entry}: {e}"))
    };

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

kernel_cache!(BINARY, binary_kernel, BINARY_CU, "binary");
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
    MATMUL_WMMA,
    matmul_wmma_kernel,
    MATMUL_WMMA_CU,
    "matmul_wmma"
);
kernel_cache!(COMPARE, compare_kernel, COMPARE_CU, "compare");
kernel_cache!(WHEREK, where_kernel, WHERE_CU, "where_select");
kernel_cache!(REDUCE, reduce_kernel, REDUCE_CU, "reduce");
kernel_cache!(SOFTMAX, softmax_kernel, SOFTMAX_CU, "softmax");
kernel_cache!(LAYERNORM, layernorm_kernel, LAYERNORM_CU, "rlx_norm");
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
kernel_cache!(GATHER, gather_kernel, GATHER_CU, "gather");
kernel_cache!(
    GATHER_AXIS,
    gather_axis_kernel,
    GATHER_AXIS_CU,
    "gather_axis"
);
kernel_cache!(NARROW, narrow_kernel, NARROW_CU, "narrow");
kernel_cache!(CONCAT, concat_kernel, CONCAT_CU, "concat");
kernel_cache!(TRANSPOSE, transpose_kernel, TRANSPOSE_CU, "transpose");
kernel_cache!(EXPAND, expand_kernel, EXPAND_CU, "expand");
kernel_cache!(ATTENTION, attention_kernel, ATTENTION_CU, "attention");
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
    ATTENTION_BWD,
    attention_bwd_kernel,
    ATTENTION_BWD_CU,
    "attention_bwd"
);
kernel_cache!(ARGMAX, argmax_kernel, ARGMAX_CU, "argmax");
kernel_cache!(ROPE, rope_kernel, ROPE_CU, "rope");
kernel_cache!(CUMSUM, cumsum_kernel, CUMSUM_CU, "cumsum");
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
    DEQUANT_GGUF,
    dequant_gguf_kernel,
    DEQUANT_GGUF_CU,
    "dequant_gguf"
);
kernel_cache!(SAMPLE, sample_kernel, SAMPLE_CU, "sample");
kernel_cache!(
    SELECTIVE_SCAN,
    selective_scan_kernel,
    SELECTIVE_SCAN_CU,
    "selective_scan"
);
kernel_cache!(POOL1D, pool1d_kernel, POOL1D_CU, "pool1d");
kernel_cache!(POOL2D, pool2d_kernel, POOL2D_CU, "pool2d");
const MAXPOOL2D_BWD_CU: &str = r#"
extern "C" __global__ void maxpool2d_backward(
    float* arena, unsigned n, unsigned c, unsigned h, unsigned w,
    unsigned h_out, unsigned w_out, unsigned kh, unsigned kw,
    unsigned sh, unsigned sw, unsigned ph, unsigned pw,
    unsigned x_off, unsigned dy_off, unsigned dx_off)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = n*c*h*w;
    if (idx >= total) return;
    unsigned iw = idx % w;
    unsigned ih = (idx / w) % h;
    unsigned cc = (idx / (w*h)) % c;
    unsigned nn = idx / (w*h*c);
    const float* x = arena + x_off;
    unsigned base_nc = (nn*c + cc)*h*w;
    int ph_i = (int)ph, pw_i = (int)pw, sh_i = (int)sh, sw_i = (int)sw;
    // output windows (ho,wo) whose receptive field covers input (ih,iw)
    int ho_lo = (int)ih + ph_i - (int)kh + 1;
    ho_lo = ho_lo <= 0 ? 0 : (ho_lo + sh_i - 1) / sh_i;
    int ho_hi = ((int)ih + ph_i) / sh_i;
    int wo_lo = (int)iw + pw_i - (int)kw + 1;
    wo_lo = wo_lo <= 0 ? 0 : (wo_lo + sw_i - 1) / sw_i;
    int wo_hi = ((int)iw + pw_i) / sw_i;
    float acc = 0.0f;
    for (int ho = ho_lo; ho <= ho_hi && ho < (int)h_out; ho++) {
        int hstart = ho*sh_i - ph_i;
        for (int wo = wo_lo; wo <= wo_hi && wo < (int)w_out; wo++) {
            int wstart = wo*sw_i - pw_i;
            float best = -3.402823466e+38f; int best_idx = -1;
            for (unsigned i=0;i<kh;i++){
                int ir = hstart + (int)i;
                if (ir < 0 || ir >= (int)h) continue;
                for (unsigned j=0;j<kw;j++){
                    int ic = wstart + (int)j;
                    if (ic < 0 || ic >= (int)w) continue;
                    unsigned id2 = base_nc + (unsigned)ir*w + (unsigned)ic;
                    float v = x[id2];
                    if (v > best){ best = v; best_idx = (int)id2; }
                }
            }
            if (best_idx == (int)idx) {
                acc += arena[dy_off + ((nn*c+cc)*h_out + (unsigned)ho)*w_out + (unsigned)wo];
            }
        }
    }
    arena[dx_off + idx] = acc;
}
"#;
kernel_cache!(
    MAXPOOL2D_BWD,
    maxpool2d_backward_kernel,
    MAXPOOL2D_BWD_CU,
    "maxpool2d_backward"
);
kernel_cache!(POOL3D, pool3d_kernel, POOL3D_CU, "pool3d");
kernel_cache!(CONV1D, conv1d_kernel, CONV1D_CU, "conv1d");
kernel_cache!(CONV2D, conv2d_kernel, CONV2D_CU, "conv2d");
kernel_cache!(IM2COL, im2col_kernel, IM2COL_CU, "im2col");
kernel_cache!(CONV3D, conv3d_kernel, CONV3D_CU, "conv3d");
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
kernel_cache!(GROUP_NORM, group_norm_kernel, GROUP_NORM_CU, "group_norm");
kernel_cache!(
    RESIZE_NEAREST_2X,
    resize_nearest_2x_kernel,
    RESIZE_NEAREST_2X_CU,
    "resize_nearest_2x"
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

/// Dispatch grid for a 1-D workload of `n` threads with workgroup
/// size `block_x`. CUDA's per-grid-dim limit is 2^31-1 on the X axis,
/// so the 2-D fallback wgpu requires isn't needed here.
pub fn dispatch_grid_1d(n: u32, block_x: u32) -> (u32, u32) {
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
