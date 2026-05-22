// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Unified model component — one object drives specialization, compile cache, and binding.
//!
//! Mirrors Slang “shader components”: the same granularity selects **what to specialize**
//! (dims, dispatch, compilation mode) and **how host code binds** (via [`BindingManifest`]
//! after specialize). Works across eager, lazy, and AOT pipelines while keeping HIR/MIR/LIR.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::logical_kernel::KernelDispatchConfig;
use crate::quant::QuantScheme;
use crate::variant::ModelVariant;

/// When the backend executable is produced relative to the host loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompilationMode {
    /// Compile before first `run` (default inference).
    #[default]
    Eager,
    /// Build template at load; specialize/compile on first use per variant.
    Lazy,
    /// Serialize LIR / executable to disk; load without re-running fusion.
    Aot,
}

/// Full specialization + binding bundle (Slang shader-component analogue).
#[derive(Debug, Clone)]
pub struct ModelComponent {
    pub variant: ModelVariant,
    pub kernel_dispatch: KernelDispatchConfig,
    pub compilation_mode: CompilationMode,
    /// Hash of tier-1 [`CompileProfile`] or arch preset (see `rlx-flow` presets).
    pub profile_key: u64,
    /// Optional quant scheme affecting lowers and weight layout.
    pub quant: Option<QuantScheme>,
    /// Composite layer-stack fingerprint (homogeneous depth, pair nesting).
    pub layer_composition_key: u64,
}

impl ModelComponent {
    pub fn new(variant: ModelVariant) -> Self {
        Self {
            variant,
            kernel_dispatch: KernelDispatchConfig::default(),
            compilation_mode: CompilationMode::Eager,
            profile_key: 0,
            quant: None,
            layer_composition_key: 0,
        }
    }

    pub fn with_kernel_dispatch(mut self, config: KernelDispatchConfig) -> Self {
        self.kernel_dispatch = config;
        self
    }

    pub fn with_compilation_mode(mut self, mode: CompilationMode) -> Self {
        self.compilation_mode = mode;
        self
    }

    pub fn with_profile_key(mut self, key: u64) -> Self {
        self.profile_key = key;
        self
    }

    pub fn with_quant(mut self, scheme: QuantScheme) -> Self {
        self.quant = Some(scheme);
        self
    }

    pub fn with_layer_composition_key(mut self, key: u64) -> Self {
        self.layer_composition_key = key;
        self
    }

    /// Stable key for compile caches (variant + dispatch + profile + composition).
    pub fn cache_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.variant.cache_key().hash(&mut h);
        (self.kernel_dispatch.policy as u8).hash(&mut h);
        for k in self.kernel_dispatch.force_common_kinds.iter() {
            k.hash(&mut h);
        }
        for k in self.kernel_dispatch.force_native_kinds.iter() {
            k.hash(&mut h);
        }
        self.compilation_mode.hash(&mut h);
        self.profile_key.hash(&mut h);
        if let Some(q) = &self.quant {
            format!("{q:?}").hash(&mut h);
        }
        self.layer_composition_key.hash(&mut h);
        h.finish()
    }

    pub fn dim_binding(&self) -> crate::DimBinding {
        self.variant.dim_binding()
    }

    /// Stable on-disk prefix for [`rlx_runtime::AotCache`] (`{base}__{binding_hash}` per variant).
    pub fn aot_disk_base(&self) -> String {
        format!("rlx_{:016x}", self.cache_key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_kernel::KernelDispatchPolicy;
    use crate::ModelVariant;

    #[test]
    fn cache_key_changes_with_mode_and_profile() {
        let v = ModelVariant::prefill(1, 8);
        let a = ModelComponent::new(v.clone()).cache_key();
        let b = ModelComponent::new(v.clone())
            .with_compilation_mode(CompilationMode::Lazy)
            .cache_key();
        let c = ModelComponent::new(v)
            .with_profile_key(42)
            .with_kernel_dispatch(KernelDispatchConfig::new(KernelDispatchPolicy::ForceCommon))
            .cache_key();
        assert_ne!(a, b);
        assert_ne!(a, c);
    }
}
