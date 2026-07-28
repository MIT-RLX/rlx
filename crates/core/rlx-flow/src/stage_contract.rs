// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed stage contracts — associated artifacts per layer (Slang-style associated types).

use std::sync::Arc;

use anyhow::Result;
use rlx_ir::Shape;

use crate::blocks::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Outputs a layer stage may publish beyond the main hidden tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageArtifacts {
    pub hidden: Shape,
    pub side_outputs: Vec<(String, Shape)>,
}

impl StageArtifacts {
    pub fn hidden_only(shape: Shape) -> Self {
        Self {
            hidden: shape,
            side_outputs: Vec::new(),
        }
    }

    pub fn with_side(mut self, name: impl Into<String>, shape: Shape) -> Self {
        self.side_outputs.push((name.into(), shape));
        self
    }
}

/// Layer block with an explicit artifact contract (for new blocks and plugins).
///
/// This is the **downstream extension seam**: a crate outside `rlx-flow` (e.g. a
/// model crate in `rlx-models`) implements `LayerStage`, then drops it into any
/// flow with [`ModelFlow::layer`](crate::ModelFlow::layer) — no new
/// [`FlowStage`](crate::FlowStage) enum variant and no core edit required. The
/// block composes ordinary primitives through [`FlowCtx`], so it still fuses and
/// hits the fast path on every backend, unlike an opaque `Op::Custom`.
pub trait LayerStage: Send + Sync {
    fn name(&self) -> &str;

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)>;
}

/// Type-erased [`LayerStage`] handle embedded in [`FlowStage::Dynamic`].
///
/// Wraps `Arc<dyn LayerStage>` so [`FlowStage`](crate::FlowStage) can stay
/// `Debug + Clone` (the wrapper prints the stage's `name()` and clones the
/// `Arc`). Construct via [`FlowStage::dynamic`](crate::FlowStage::dynamic) or the
/// `.layer(..)` builder methods rather than reaching for this directly.
#[derive(Clone)]
pub struct DynStage(pub Arc<dyn LayerStage>);

impl DynStage {
    pub fn new(stage: impl LayerStage + 'static) -> Self {
        Self(Arc::new(stage))
    }

    pub fn name(&self) -> &str {
        self.0.name()
    }
}

impl std::fmt::Debug for DynStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DynStage").field(&self.0.name()).finish()
    }
}

/// Bridge existing [`BlockStage`] impls to [`LayerStage`] with hidden-only artifacts.
pub struct BlockAsLayer<S>(pub S);

impl<S: BlockStage + Send + Sync> LayerStage for BlockAsLayer<S> {
    fn name(&self) -> &str {
        "block"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let out = self.0.emit(ctx, input.clone())?;
        let value = match out {
            Some(v) => v,
            None => input,
        };
        Ok((
            value.clone(),
            StageArtifacts::hidden_only(value.shape.clone()),
        ))
    }
}
