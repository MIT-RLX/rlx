// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Block assembly-line API for RLX model builders.

pub mod blocks;
mod context;
mod dsl;
mod composite;
mod execution;
mod extension;
mod stage_interfaces;
pub mod escape;
mod flow;
mod layer;
mod profile;
mod plugin;
mod recipe;
mod side;
mod stage;
mod stage_contract;
pub mod stream;
mod value;
mod weight;

pub mod prelude;

pub use blocks::{
    BertEncoderLayerSpec, BertEncoderLayerStage, BertQkvStyle, ClsTokenPoolStage,
    NomicEncoderLayerSpec, NomicEncoderLayerStage, Qwen3DecodeLayerSpec, Qwen3DecoderSpec,
    Qwen3DecoderStage, VitSelfAttnSpec, dinov2_layer_fused, nomic_vision_layer_fused,
};
pub use blocks::RopeTablesStage;
pub use context::{DecodeBindings, FlowState, GdnInputSlots};
pub use escape::Emit;
pub use composite::LayerComposition;
pub use execution::{ExecutionPreset, ModelExecutionConfig};
pub use extension::FlowExtensionPlan;
pub use stage_interfaces::{AttentionStage, FfnStage, KvCacheContract, NormStage};
pub use flow::{BuiltModel, ModelFlow};
pub use layer::LayerStack;
pub use recipe::ModelRecipe;
pub use stage_contract::{BlockAsLayer, LayerStage, StageArtifacts};
pub use profile::{
    BackendOverrides, CompileProfile, CpuBackendProfile, FusionPolicyKind, FusionProfile,
    FusionTargetKind, MetalBackendProfile, MixedPrecisionKind, PassProfile, PrecisionKind,
    PrecisionProfile,
};
pub use plugin::{plugin, plugin_named, PluginStage};
pub use stream::{dual_stream_stage, id as stream_id, DualStreamStage, LoadStreamStage, StoreStreamStage};
pub use side::SideOutputs;
pub use stage::FlowStage;
pub use value::FlowValue;
pub use weight::{MapWeights, WeightSource};
