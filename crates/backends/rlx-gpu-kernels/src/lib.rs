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

//! Shared GPU kernel sources for RLX CUDA and ROCm backends.
//!
//! Each constant is the full `.cu` source text, embedded at compile time.
//! Backends JIT-compile via NVRTC / hipRTC on first use.

pub const BINARY_CU: &str = include_str!("../kernels/binary.cu");
/// Standalone complex `Op::Cast` on the f32-uniform arena (real<->C64,
/// real<->C128, C64<->C128 — six pure lane-move modes). f32-uniform GPU
/// backends simulate complex as interleaved f32 lanes; this re-pairs them.
pub const COMPLEX_CAST_CU: &str = include_str!("../kernels/complex_cast.cu");
/// Element-wise C64 binary op (add/sub/mul/div) reading both `[re, im]`
/// lanes per element, with modulo broadcast. Mirrors rlx-cpu
/// `exec_binary_full_c64`; C128 arithmetic + C64 max/min/pow are rejected.
pub const BINARY_C64_CU: &str = include_str!("../kernels/binary_c64.cu");
pub const FUSED_BINARY_UNARY_CU: &str = include_str!("../kernels/fused_binary_unary.cu");
pub const CAST_F32_TO_HALF_CU: &str = include_str!("../kernels/cast_f32_to_half.cu");
/// Native FP8 quantize producers (per-tensor scale + E4M3/E5M2 encode) for
/// `Op::ScaledMatMul`. Shared by the CUDA (cublasLt) and ROCm (hipBLASLt) paths.
pub const SCALED_LOWP_CU: &str = include_str!("../kernels/scaled_lowp.cu");
/// General (all-format, all-scale-layout) low-precision quantize + decode-GEMM
/// for `Op::ScaledMatMul` — the on-device decode-and-accumulate fallback for
/// block-scaled / FP4 / FP6 configs the FP8 tensor-core path can't do.
pub const SCALED_LOWP_GENERAL_CU: &str = include_str!("../kernels/scaled_lowp_general.cu");
pub const UNARY_CU: &str = include_str!("../kernels/unary.cu");
pub const LSTM_CU: &str = include_str!("../kernels/lstm.cu");
pub const BINARY_BROADCAST_CU: &str = include_str!("../kernels/binary_broadcast.cu");
pub const COPY_CU: &str = include_str!("../kernels/copy.cu");
pub const MATMUL_CU: &str = include_str!("../kernels/matmul.cu");
pub const MATMUL_BT_CU: &str = include_str!("../kernels/matmul_bt.cu");
pub const MATMUL_EPILOGUE_CU: &str = include_str!("../kernels/matmul_epilogue.cu");
pub const MATMUL_WMMA_CU: &str = include_str!("../kernels/matmul_wmma.cu");
pub const COMPARE_CU: &str = include_str!("../kernels/compare.cu");
pub const WHERE_CU: &str = include_str!("../kernels/where_select.cu");
pub const REDUCE_CU: &str = include_str!("../kernels/reduce.cu");
pub const SOFTMAX_CU: &str = include_str!("../kernels/softmax.cu");
pub const LAYERNORM_CU: &str = include_str!("../kernels/layernorm.cu");
pub const RMS_NORM_BWD_CU: &str = include_str!("../kernels/rms_norm_backward.cu");
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
pub const FUSED_ATTN_CU: &str = include_str!("../kernels/fused_attn.cu");
pub const ATTENTION_ROW_CU: &str = include_str!("../kernels/attention_row.cu");
pub const ATTENTION_BWD_CU: &str = include_str!("../kernels/attention_bwd.cu");
pub const ARGMAX_CU: &str = include_str!("../kernels/argmax.cu");
pub const ROPE_CU: &str = include_str!("../kernels/rope.cu");
pub const CUMSUM_CU: &str = include_str!("../kernels/cumsum.cu");
pub const TOPK_CU: &str = include_str!("../kernels/topk.cu");
pub const GROUPED_MATMUL_CU: &str = include_str!("../kernels/grouped_matmul.cu");
pub const SCATTER_ADD_CU: &str = include_str!("../kernels/scatter_add.cu");
pub const DEQUANT_MATMUL_CU: &str = include_str!("../kernels/dequant_matmul.cu");
pub const DEQUANT_GGUF_CU: &str = include_str!("../kernels/dequant_gguf.cu");
pub const DEQUANT_MATMUL_GGUF_CU: &str = include_str!("../kernels/dequant_matmul_gguf.cu");
pub const SAMPLE_CU: &str = include_str!("../kernels/sample.cu");
pub const SELECTIVE_SCAN_CU: &str = include_str!("../kernels/selective_scan.cu");
pub const GATED_DELTA_NET_CU: &str = include_str!("../kernels/gated_delta_net.cu");
pub const POOL1D_CU: &str = include_str!("../kernels/pool1d.cu");
pub const POOL2D_CU: &str = include_str!("../kernels/pool2d.cu");
pub const POOL3D_CU: &str = include_str!("../kernels/pool3d.cu");
pub const CONV1D_CU: &str = include_str!("../kernels/conv1d.cu");
pub const CONV2D_CU: &str = include_str!("../kernels/conv2d.cu");
pub const CONV2D_BACKWARD_INPUT_CU: &str = include_str!("../kernels/conv2d_backward_input.cu");
pub const CONV2D_BACKWARD_WEIGHT_CU: &str = include_str!("../kernels/conv2d_backward_weight.cu");
pub const IM2COL_CU: &str = include_str!("../kernels/im2col.cu");
pub const CONV3D_CU: &str = include_str!("../kernels/conv3d.cu");
pub const LAYER_NORM2D_CU: &str = include_str!("../kernels/layer_norm2d.cu");
pub const CONV_TRANSPOSE2D_CU: &str = include_str!("../kernels/conv_transpose2d.cu");
pub const GROUP_NORM_CU: &str = include_str!("../kernels/group_norm.cu");
pub const RESIZE_NEAREST_2X_CU: &str = include_str!("../kernels/resize_nearest_2x.cu");
pub const ELEMENTWISE_REGION_CU: &str = include_str!("../kernels/elementwise_region.cu");
pub const BATCH_ELEMENTWISE_REGION_CU: &str =
    include_str!("../kernels/batch_elementwise_region.cu");
pub const GAUSSIAN_SPLAT_RASTERIZE_CU: &str =
    include_str!("../kernels/gaussian_splat_rasterize.cu");
pub const FFT_CU: &str = include_str!("../kernels/fft.cu");
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

cuda_src_with_gelu!(unary_cuda_src, include_str!("../kernels/unary.cu"));
cuda_src_with_gelu!(
    fused_binary_unary_cuda_src,
    include_str!("../kernels/fused_binary_unary.cu")
);
cuda_src_with_gelu!(matmul_cuda_src, include_str!("../kernels/matmul.cu"));
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
}
