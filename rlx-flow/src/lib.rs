// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Block assembly-line API for RLX model builders.

pub mod blocks;
mod composite;
mod context;
mod dsl;
pub mod escape;
mod execution;
mod extension;
mod flow;
mod layer;
mod plugin;
mod profile;
mod recipe;
mod side;
mod stage;
mod stage_contract;
mod stage_interfaces;
pub mod stream;
mod value;
mod weight;

pub mod prelude;

pub use blocks::RopeTablesStage;
pub use blocks::{
    BertEncoderLayerSpec, BertEncoderLayerStage, BertQkvStyle, ClsTokenPoolStage,
    NomicEncoderLayerSpec, NomicEncoderLayerStage, Qwen3DecodeLayerSpec, Qwen3DecoderSpec,
    Qwen3DecoderStage, VitSelfAttnSpec, dinov2_layer_fused, nomic_vision_layer_fused,
};
pub use composite::LayerComposition;
pub use context::{DecodeBindings, FlowState, GdnInputSlots};
pub use escape::Emit;
pub use execution::{ExecutionPreset, ModelExecutionConfig};
pub use extension::FlowExtensionPlan;
pub use flow::{BuiltModel, ModelFlow};
pub use layer::LayerStack;
pub use plugin::{PluginStage, plugin, plugin_named};
pub use profile::{
    BackendOverrides, CompileProfile, CpuBackendProfile, FusionPolicyKind, FusionProfile,
    FusionTargetKind, MetalBackendProfile, MixedPrecisionKind, PassProfile, PrecisionKind,
    PrecisionProfile,
};
pub use recipe::ModelRecipe;
pub use side::SideOutputs;
pub use stage::FlowStage;
pub use stage_contract::{BlockAsLayer, LayerStage, StageArtifacts};
pub use stage_interfaces::{AttentionStage, FfnStage, KvCacheContract, NormStage};
pub use stream::{
    DualStreamStage, LoadStreamStage, StoreStreamStage, dual_stream_stage, id as stream_id,
};
pub use value::FlowValue;
pub use weight::{MapWeights, WeightSource};
