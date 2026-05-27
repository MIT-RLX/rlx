// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use std::fmt;
use std::sync::Arc;

use anyhow::Result;

use crate::context::FlowCtx;
use crate::escape::Emit;
use crate::value::FlowValue;
type CustomFn =
    Arc<dyn Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>> + Send + Sync>;

/// User-defined stage — tier-2 escape hatch for novel subgraphs.
#[derive(Clone)]
pub struct CustomStage {
    pub name: Option<String>,
    f: CustomFn,
}

impl fmt::Debug for CustomStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomStage")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl CustomStage {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: None,
            f: Arc::new(f),
        }
    }

    pub fn named<F>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&mut Emit<'_>, Option<FlowValue>) -> Result<Option<FlowValue>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: Some(name.into()),
            f: Arc::new(f),
        }
    }

    pub fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: Option<FlowValue>,
    ) -> Result<Option<FlowValue>> {
        let mut emit = Emit::from_ctx(ctx);
        (self.f)(&mut emit, input)
    }
}
