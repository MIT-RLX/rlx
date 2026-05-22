// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;

use crate::context::FlowCtx;
use crate::stage::FlowStage;
use crate::value::FlowValue;
use crate::weight::WeightSource;

pub struct RepeatStage {
    pub count: usize,
    pub stage_for_index: std::sync::Arc<dyn Fn(usize) -> FlowStage + Send + Sync>,
}

impl std::fmt::Debug for RepeatStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepeatStage")
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl RepeatStage {
    pub fn new(count: usize, stage_for_index: impl Fn(usize) -> FlowStage + Send + Sync + 'static) -> Self {
        Self {
            count,
            stage_for_index: std::sync::Arc::new(stage_for_index),
        }
    }
}

impl Clone for RepeatStage {
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            stage_for_index: std::sync::Arc::clone(&self.stage_for_index),
        }
    }
}

impl RepeatStage {
    pub fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        mut input: Option<FlowValue>,
    ) -> Result<Option<FlowValue>> {
        for i in 0..self.count {
            let stage = (self.stage_for_index)(i);
            input = stage.emit(ctx, input)?;
        }
        Ok(input)
    }
}
