// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Composite layer environments (Slang `LightPair` / `LightArray`).
//!
//! Describe static layer stacks for specialization keys and flow assembly.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::stage::FlowStage;

/// Static description of a layer stack for cache keys and recipes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerComposition {
    /// Single named stage.
    Single { name: String },
    /// `count` identical layers (homogeneous array).
    Homogeneous { layer_name: String, count: usize },
    /// Heterogeneous head + tail (pair).
    Pair {
        head: Box<LayerComposition>,
        tail: Box<LayerComposition>,
    },
}

impl LayerComposition {
    pub fn single(name: impl Into<String>) -> Self {
        Self::Single { name: name.into() }
    }

    pub fn homogeneous(layer_name: impl Into<String>, count: usize) -> Self {
        Self::Homogeneous {
            layer_name: layer_name.into(),
            count,
        }
    }

    pub fn pair(head: LayerComposition, tail: LayerComposition) -> Self {
        Self::Pair {
            head: Box::new(head),
            tail: Box::new(tail),
        }
    }

    /// Fingerprint for [`rlx_ir::ModelComponent::layer_composition_key`].
    pub fn cache_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash_fragment(&mut h);
        h.finish()
    }

    fn hash_fragment(&self, h: &mut DefaultHasher) {
        match self {
            Self::Single { name } => {
                0u8.hash(h);
                name.hash(h);
            }
            Self::Homogeneous { layer_name, count } => {
                1u8.hash(h);
                layer_name.hash(h);
                count.hash(h);
            }
            Self::Pair { head, tail } => {
                2u8.hash(h);
                head.hash_fragment(h);
                tail.hash_fragment(h);
            }
        }
    }

    /// Expand into a [`FlowStage::Sequence`] by repeating `build_layer`.
    pub fn to_flow_stage(&self, build_layer: &dyn Fn(&str, usize) -> FlowStage) -> FlowStage {
        match self {
            Self::Single { name } => build_layer(name, 0),
            Self::Homogeneous { layer_name, count } => {
                let stages: Vec<_> = (0..*count)
                    .map(|i| FlowStage::Named {
                        name: format!("{layer_name}{i}"),
                        inner: Arc::new(build_layer(layer_name, i)),
                    })
                    .collect();
                FlowStage::Sequence(stages)
            }
            Self::Pair { head, tail } => FlowStage::Sequence(vec![
                head.to_flow_stage(build_layer),
                tail.to_flow_stage(build_layer),
            ]),
        }
    }

    pub fn depth_hint(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::Homogeneous { count, .. } => *count,
            Self::Pair { head, tail } => head.depth_hint() + tail.depth_hint(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::FlowStage;

    #[test]
    fn homogeneous_cache_key_scales_with_count() {
        let a = LayerComposition::homogeneous("layer", 8).cache_key();
        let b = LayerComposition::homogeneous("layer", 32).cache_key();
        assert_ne!(a, b);
    }

    #[test]
    fn pair_expands_two_stages() {
        let comp =
            LayerComposition::pair(LayerComposition::single("a"), LayerComposition::single("b"));
        let stage = comp.to_flow_stage(&|name, _| FlowStage::Named {
            name: name.into(),
            inner: Arc::new(FlowStage::Sequence(vec![])),
        });
        assert!(matches!(stage, FlowStage::Sequence(s) if s.len() == 2));
    }
}
