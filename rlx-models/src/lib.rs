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

//! RLX model loading — parse configs, load weights, build IR graphs.
//!
//! Supports BERT, NomicBERT, and NomicVision architectures.
//!
//! # Quick start
//! ```rust,ignore
//! use rlx_models::embed::RlxEmbed;
//!
//! let mut model = RlxEmbed::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
//! ```

pub mod arch_registry;
pub mod bert;
pub mod config;
pub mod dataprocessing;
pub mod dinov2;
pub mod embed;
pub mod nomic;
pub mod qwen3;
pub mod qwen35;
pub mod run;
pub mod sam;
pub mod sam2;
pub mod sam3;
pub mod vision;
pub mod weight_loader;
pub mod weight_map;

pub use weight_loader::{WeightLoader, load_from_path};

pub use arch_registry::{ArchFamily, ArchSpec};

pub use bert::{build_bert_graph, build_bert_graph_sized};
pub use config::{BertConfig, NomicBertConfig, NomicVisionConfig};
pub use dinov2::{DinoV2Config, DinoV2PreprocessWeights, build_dinov2_graph_sized};
pub use embed::{Arch, Pooling, RlxEmbed};
pub use nomic::{build_nomic_diagnostic_graph, build_nomic_graph_sized};
pub use qwen3::{
    Qwen3Config, Qwen3Generator, Qwen3Speculator, SampleOpts, build_qwen3_graph_sized, sample_token,
};
pub use qwen35::{
    Qwen35Config, Qwen35FullAttnLayer, Qwen35LinearLayer, Qwen35MtpLayer, Qwen35PrefillOutput,
    Qwen35Runner, Qwen35RunnerBuilder, Qwen35TrunkLayer, Qwen35Weights,
    build_qwen35_graph_sized, build_qwen35_graph_sized_stub,
};
pub use sam::{
    NeckWeights as SamNeckWeights, SamConfig, SamEncoderConfig, SamPreprocessWeights,
    apply_neck_host as sam_apply_neck_host, assemble_patch_tokens as sam_assemble_patch_tokens,
    build_sam_encoder_graph, preprocess_image as sam_preprocess_image,
};
pub use sam2::{
    FpnLevel as Sam2FpnLevel, FpnNeckWeights as Sam2FpnNeckWeights, Sam2, Sam2Config,
    Sam2DecoderConfig, Sam2FpnConfig, Sam2HieraConfig, Sam2ImagePrediction, Sam2MaskDecoderOutput,
    Sam2MaskDecoderWeights, Sam2MemoryAttentionWeights, Sam2MemoryConfig, Sam2MemoryEncoderConfig,
    Sam2MemoryEncoderOutput, Sam2MemoryEncoderWeights, Sam2PreprocessWeights,
    Sam2PromptEncoderOutput, Sam2PromptEncoderWeights, Sam2TwoWayTransformerWeights,
    Sam2VideoState, apply_fpn_neck_host as sam2_apply_fpn_neck_host,
    assemble_patch_tokens as sam2_assemble_patch_tokens, build_sam2_image_encoder_graph,
    mask_decoder_forward as sam2_mask_decoder_forward,
    memory_attention_forward as sam2_memory_attention_forward,
    memory_encoder_forward as sam2_memory_encoder_forward,
    preprocess_image as sam2_preprocess_image,
    prompt_encoder_forward as sam2_prompt_encoder_forward,
    two_way_transformer_forward as sam2_two_way_transformer_forward,
};
pub use sam3::{
    Sam3, Sam3Config, Sam3DetectorConfig, Sam3EncodedImage, Sam3ImagePrediction,
    Sam3PreprocessWeights, Sam3TextConfig, Sam3TrackerConfig, Sam3VideoFramePrediction,
    Sam3VideoState, Sam3VitConfig, assemble_patch_tokens as sam3_assemble_patch_tokens,
    preprocess_image as sam3_preprocess_image,
};
pub use run::{
    ConfigSource, DinoV2Output, DinoV2Runner, DinoV2RunnerBuilder, DinoV2Variant, ModelRunner,
    Precision, Qwen3Runner, Qwen3RunnerBuilder, SamArch, SamPredictionAny, SamRunner,
    SamRunnerBuilder, WeightFormat, debug_resolve_name, dispatch, dispatch_help, list_mtp_keys,
    open_loader, register_runner, registered_runners, run_registered,
};
pub use vision::{VisionPreprocessWeights, build_vision_graph_sized};
pub use weight_map::WeightMap;
