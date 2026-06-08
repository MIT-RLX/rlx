// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! FKL-style fusion fragment registry - extensible op roles for region passes.
//!
//! Third-party or closed-source ops can register [`FusionFragment`] implementations
//! so transform / prologue / batch passes discover them without editing core matchers.

use rlx_ir::{Op, OpKind};
use std::sync::{OnceLock, RwLock};

/// How an op participates in FKL-inspired region fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionRole {
    /// Sampling / geometry (FKL ReadBack): resize, crop, color convert.
    TransformStep,
    /// Register-only chain step inside [`Op::ElementwiseRegion`].
    RegionCompute,
    /// Pre-region memory transform fused into an element-wise region.
    RegionPrologue,
    /// Independent batch plane (horizontal fusion).
    BatchPlane,
}

/// One registrable fusion participant (built-in or plugin).
pub trait FusionFragment: Send + Sync {
    fn name(&self) -> &'static str;
    fn role(&self) -> FusionRole;
    fn op_kinds(&self) -> &'static [OpKind];
    fn matches_op(&self, op: &Op) -> bool {
        self.op_kinds().iter().any(|k| op.kind() == *k)
    }
}

struct BuiltinResizeNearest2x;
impl FusionFragment for BuiltinResizeNearest2x {
    fn name(&self) -> &'static str {
        "resize_nearest_2x"
    }
    fn role(&self) -> FusionRole {
        FusionRole::TransformStep
    }
    fn op_kinds(&self) -> &'static [OpKind] {
        &[OpKind::ResizeNearest2x]
    }
}

struct BuiltinElementwiseRegion;
impl FusionFragment for BuiltinElementwiseRegion {
    fn name(&self) -> &'static str {
        "elementwise_region"
    }
    fn role(&self) -> FusionRole {
        FusionRole::RegionCompute
    }
    fn op_kinds(&self) -> &'static [OpKind] {
        &[OpKind::ElementwiseRegion]
    }
}

static REGISTRY: OnceLock<RwLock<Vec<&'static dyn FusionFragment>>> = OnceLock::new();

fn registry() -> &'static RwLock<Vec<&'static dyn FusionFragment>> {
    REGISTRY.get_or_init(|| RwLock::new(vec![&BuiltinResizeNearest2x, &BuiltinElementwiseRegion]))
}

/// Register an additional fragment (e.g. from a downstream crate at startup).
pub fn register_fusion_fragment(fragment: &'static dyn FusionFragment) {
    registry()
        .write()
        .expect("fusion fragment registry poisoned")
        .push(fragment);
}

/// All registered fragments (built-in + plugins).
pub fn fusion_fragments() -> Vec<&'static dyn FusionFragment> {
    registry()
        .read()
        .expect("fusion fragment registry poisoned")
        .clone()
}

/// True if any registered transform fragment matches `op`.
pub fn is_registered_transform_op(op: &Op) -> bool {
    fusion_fragments()
        .iter()
        .any(|f| f.role() == FusionRole::TransformStep && f.matches_op(op))
}

/// True if `op` may start or extend a transform region chain.
pub fn transform_chain_eligible(op: &Op) -> bool {
    op.is_transform_eligible() || is_registered_transform_op(op)
}

/// Registered prologue kinds derived from transform fragments (today: resize 2x).
pub fn prologue_for_transform_op(op: &Op) -> Option<rlx_ir::RegionPrologue> {
    if matches!(op, Op::ResizeNearest2x) {
        Some(rlx_ir::RegionPrologue::ResizeNearest2x)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyCrop;
    impl FusionFragment for DummyCrop {
        fn name(&self) -> &'static str {
            "dummy_crop"
        }
        fn role(&self) -> FusionRole {
            FusionRole::TransformStep
        }
        fn op_kinds(&self) -> &'static [OpKind] {
            &[]
        }
        fn matches_op(&self, op: &Op) -> bool {
            matches!(op, Op::Narrow { axis: 0, .. })
        }
    }

    #[test]
    fn registry_lists_builtins() {
        let names: Vec<_> = fusion_fragments().iter().map(|f| f.name()).collect();
        assert!(names.contains(&"resize_nearest_2x"));
    }

    #[test]
    fn plugin_fragment_is_discovered() {
        register_fusion_fragment(&DummyCrop);
        assert!(is_registered_transform_op(&Op::Narrow {
            axis: 0,
            start: 0,
            len: 1,
        }));
    }
}
