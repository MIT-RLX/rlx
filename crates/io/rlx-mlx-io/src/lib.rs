// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load MLX weight layouts into RLX.
//!
//! # Crate boundary
//!
//! This is a leaf **I/O** crate: it may depend on [`rlx_ir`] and host crates
//! (`anyhow`, `safetensors`, …) but **must not** depend on runtime or backend
//! crates (`rlx-runtime`, `rlx-cpu`, `rlx-metal`, …). Backend crates may depend
//! on this one for shared host dequant (same posture as `rlx-gguf`).
//!
//! Supported inputs:
//! - HuggingFace **mlx-community** directories (`config.json` + `model*.safetensors`)
//! - Single `.safetensors` files
//! - MLX / NumPy `.npz` / `.npy` dumps (`mx.savez`, `nn.Module.save_weights`)
//!
//! Quantized mlx-lm checkpoints (`quantization.mode` = `affine` / `mxfp4` /
//! `nvfp4` / `mxfp8`) can be:
//! - **dequantized to f32** via [`MlxWeights::into_f32_map`] /
//!   [`MlxWeights::into_shaped_f32`], or
//! - kept as packed triples for [`rlx_ir::Op::DequantMatMul`] with
//!   [`rlx_ir::QuantScheme::MlxAffine`] / [`MlxMxfp4`] / [`MlxMxfp8`].
//!   mlx-lm `nvfp4` maps to [`MlxMxfp4`] (same pack layout; typical
//!   `group_size=16` with FP8 E4M3 scales) — not GGUF/NVIDIA [`Nvfp4Block`].

mod arch;
mod config;
mod dequant;
mod dtype;
mod graph;
mod hf;
mod load;
mod npz;
mod rope;

pub use arch::{
    build_llama_decoder_layer, build_llama_like_decode, build_llama_like_from_dir,
    build_llama_like_prefill,
};
pub use config::{MlxArchConfig, MlxConfig, MlxQuantConfig, MlxQuantMode};
pub use dequant::{
    QuantizedLayer, dequant_affine_f32, dequant_matmul_affine, dequant_matmul_mxfp4,
    dequant_matvec_affine,
    dequant_mxfp4_f32, dequant_mxfp8_f32, mxfp4_scale_e8m0_to_f32, pack_factor,
    validate_dequant_matmul_dims,
};
pub use graph::{
    PackedLinearBinding, build_mlp_chain_graph, build_parallel_dequant_graph,
    collect_packed_linears, param_bindings_for,
};
pub use hf::{
    DEFAULT_HF_MLX_REPO, fetch_default_mlx_community, fetch_mlx_community, fetch_ok, hf_cache_dir,
    write_fetch_ok,
};
pub use load::{
    LazyMlxWeights, MlxPackedLinear, MlxRead, MlxTensor, MlxWeights, load_path, load_path_lazy,
};
pub use rope::{build_default_tables, default_inv_freq};

/// Dense f32 payload with logical shape preserved.
pub type ShapedF32 = (Vec<usize>, Vec<f32>);

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

/// Convenience: load any supported path and return dense f32 tensors
/// (affine / mxfp layers are dequantized).
pub fn load_f32_map(path: impl AsRef<Path>) -> Result<HashMap<String, Vec<f32>>> {
    load_path(path)?.into_f32_map()
}
