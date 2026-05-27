// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Typed stage contracts — associated artifacts per layer (Slang-style associated types).

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
pub trait LayerStage: Send + Sync {
    fn name(&self) -> &str;

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)>;
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
