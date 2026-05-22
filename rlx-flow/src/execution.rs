// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Unified execution configuration — variant + compile preset + cache key.

use rlx_ir::{
    CompilationMode, DimBinding, KernelDispatchConfig, ModelComponent, ModelVariant, QuantScheme,
};

use crate::composite::LayerComposition;
use crate::profile::CompileProfile;

/// Named compile presets (fusion policy, precision, pass toggles).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionPreset {
    Llama32Prefill,
    Llama32Decode,
    Qwen35Prefill,
    Qwen35Decode,
    Encoder,
}

impl ExecutionPreset {
    pub fn profile(&self) -> CompileProfile {
        match self {
            Self::Llama32Prefill => CompileProfile::llama32_prefill(),
            Self::Llama32Decode => CompileProfile::llama32_decode(),
            Self::Qwen35Prefill => CompileProfile::qwen35_prefill(),
            Self::Qwen35Decode => CompileProfile::qwen35_decode(),
            Self::Encoder => CompileProfile::encoder(),
        }
    }

    pub fn profile_key(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        format!("{self:?}").hash(&mut h);
        h.finish()
    }
}

/// Shader-component-style bundle: one object for specialize + compile + cache.
#[derive(Debug, Clone)]
pub struct ModelExecutionConfig {
    pub component: ModelComponent,
    pub preset: ExecutionPreset,
}

impl ModelExecutionConfig {
    pub fn from_component(component: ModelComponent, preset: ExecutionPreset) -> Self {
        Self { component, preset }
    }

    pub fn prefill(batch: usize, seq: usize) -> Self {
        Self::from_component(
            ModelComponent::new(ModelVariant::prefill(batch, seq))
                .with_profile_key(ExecutionPreset::Llama32Prefill.profile_key()),
            ExecutionPreset::Llama32Prefill,
        )
    }

    pub fn decode(batch: usize, past_seq: usize, new_tokens: usize) -> Self {
        Self::from_component(
            ModelComponent::new(ModelVariant::decode(batch, past_seq, new_tokens))
                .with_profile_key(ExecutionPreset::Llama32Decode.profile_key()),
            ExecutionPreset::Llama32Decode,
        )
    }

    pub fn qwen35_prefill(batch: usize, seq: usize) -> Self {
        Self::from_component(
            ModelComponent::new(ModelVariant::prefill(batch, seq))
                .with_profile_key(ExecutionPreset::Qwen35Prefill.profile_key()),
            ExecutionPreset::Qwen35Prefill,
        )
    }

    pub fn qwen35_decode(batch: usize, past_seq: usize) -> Self {
        Self::from_component(
            ModelComponent::new(ModelVariant::decode(batch, past_seq, 1))
                .with_profile_key(ExecutionPreset::Qwen35Decode.profile_key()),
            ExecutionPreset::Qwen35Decode,
        )
    }

    pub fn with_preset(mut self, preset: ExecutionPreset) -> Self {
        self.preset = preset;
        self.component.profile_key = preset.profile_key();
        self
    }

    pub fn with_kernel_dispatch(mut self, config: KernelDispatchConfig) -> Self {
        self.component.kernel_dispatch = config;
        self
    }

    pub fn with_compilation_mode(mut self, mode: CompilationMode) -> Self {
        self.component.compilation_mode = mode;
        self
    }

    pub fn with_quant(mut self, scheme: QuantScheme) -> Self {
        self.component.quant = Some(scheme);
        self
    }

    pub fn with_layer_composition(mut self, composition: &LayerComposition) -> Self {
        self.component.layer_composition_key = composition.cache_key();
        self
    }

    pub fn cache_key(&self) -> u64 {
        self.component.cache_key()
    }

    pub fn dim_binding(&self) -> DimBinding {
        self.component.dim_binding()
    }

    pub fn compile_profile(&self) -> CompileProfile {
        self.preset.profile()
    }

    pub fn component(&self) -> &ModelComponent {
        &self.component
    }

    pub fn variant(&self) -> &ModelVariant {
        &self.component.variant
    }
}
