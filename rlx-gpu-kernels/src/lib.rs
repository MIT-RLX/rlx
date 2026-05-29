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
pub const FUSED_BINARY_UNARY_CU: &str = include_str!("../kernels/fused_binary_unary.cu");
pub const CAST_F32_TO_HALF_CU: &str = include_str!("../kernels/cast_f32_to_half.cu");
pub const UNARY_CU: &str = include_str!("../kernels/unary.cu");
pub const COPY_CU: &str = include_str!("../kernels/copy.cu");
pub const MATMUL_CU: &str = include_str!("../kernels/matmul.cu");
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
pub const GATHER_CU: &str = include_str!("../kernels/gather.cu");
pub const GATHER_AXIS_CU: &str = include_str!("../kernels/gather_axis.cu");
pub const NARROW_CU: &str = include_str!("../kernels/narrow.cu");
pub const CONCAT_CU: &str = include_str!("../kernels/concat.cu");
pub const TRANSPOSE_CU: &str = include_str!("../kernels/transpose.cu");
pub const EXPAND_CU: &str = include_str!("../kernels/expand.cu");
pub const ATTENTION_CU: &str = include_str!("../kernels/attention.cu");
pub const ATTENTION_BWD_CU: &str = include_str!("../kernels/attention_bwd.cu");
pub const ARGMAX_CU: &str = include_str!("../kernels/argmax.cu");
pub const ROPE_CU: &str = include_str!("../kernels/rope.cu");
pub const CUMSUM_CU: &str = include_str!("../kernels/cumsum.cu");
pub const TOPK_CU: &str = include_str!("../kernels/topk.cu");
pub const GROUPED_MATMUL_CU: &str = include_str!("../kernels/grouped_matmul.cu");
pub const SCATTER_ADD_CU: &str = include_str!("../kernels/scatter_add.cu");
pub const DEQUANT_MATMUL_CU: &str = include_str!("../kernels/dequant_matmul.cu");
pub const DEQUANT_GGUF_CU: &str = include_str!("../kernels/dequant_gguf.cu");
pub const SAMPLE_CU: &str = include_str!("../kernels/sample.cu");
pub const SELECTIVE_SCAN_CU: &str = include_str!("../kernels/selective_scan.cu");
pub const POOL1D_CU: &str = include_str!("../kernels/pool1d.cu");
pub const POOL2D_CU: &str = include_str!("../kernels/pool2d.cu");
pub const POOL3D_CU: &str = include_str!("../kernels/pool3d.cu");
pub const CONV1D_CU: &str = include_str!("../kernels/conv1d.cu");
pub const CONV2D_CU: &str = include_str!("../kernels/conv2d.cu");
pub const CONV3D_CU: &str = include_str!("../kernels/conv3d.cu");
pub const LAYER_NORM2D_CU: &str = include_str!("../kernels/layer_norm2d.cu");
pub const CONV_TRANSPOSE2D_CU: &str = include_str!("../kernels/conv_transpose2d.cu");
pub const GROUP_NORM_CU: &str = include_str!("../kernels/group_norm.cu");
pub const RESIZE_NEAREST_2X_CU: &str = include_str!("../kernels/resize_nearest_2x.cu");
pub const ELEMENTWISE_REGION_CU: &str = include_str!("../kernels/elementwise_region.cu");
pub const GAUSSIAN_SPLAT_RASTERIZE_CU: &str =
    include_str!("../kernels/gaussian_splat_rasterize.cu");
pub const FFT_CU: &str = include_str!("../kernels/fft.cu");

/// AMD rocWMMA / MFMA matmul (`RLX_ROCM_MFMA=1`). Not used on CUDA.
#[cfg(feature = "rocm")]
pub mod rocm {
    pub const MATMUL_MFMA_CU: &str = include_str!("../kernels/rocm/matmul_mfma.cu");
}
