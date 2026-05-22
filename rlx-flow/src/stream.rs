// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Named tensor streams — dual-/multi-stream models without IR in recipes.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use anyhow::Result;

use crate::context::FlowCtx;
use crate::escape::Emit;
use crate::stage::FlowStage;
use crate::value::FlowValue;

/// Well-known stream ids (conventions only — any string works).
pub mod id {
    pub const MAIN: &str = "main";
    pub const IMG: &str = "img";
    pub const TXT: &str = "txt";
}

type DualFn = Arc<
    dyn Fn(&mut Emit<'_>, FlowValue, FlowValue) -> Result<(FlowValue, FlowValue)> + Send + Sync,
>;

/// Transform two named streams in place (e.g. FLUX img/txt dual block).
#[derive(Clone)]
pub struct DualStreamStage {
    pub name: String,
    pub stream_a: String,
    pub stream_b: String,
    inner: DualFn,
}

impl fmt::Debug for DualStreamStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DualStreamStage")
            .field("name", &self.name)
            .field("stream_a", &self.stream_a)
            .field("stream_b", &self.stream_b)
            .finish_non_exhaustive()
    }
}

impl DualStreamStage {
    pub fn new<F>(
        name: impl Into<String>,
        stream_a: impl Into<String>,
        stream_b: impl Into<String>,
        f: F,
    ) -> Self
    where
        F: Fn(&mut Emit<'_>, FlowValue, FlowValue) -> Result<(FlowValue, FlowValue)>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: name.into(),
            stream_a: stream_a.into(),
            stream_b: stream_b.into(),
            inner: Arc::new(f),
        }
    }

    pub fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: Option<FlowValue>,
    ) -> Result<Option<FlowValue>> {
        let a = ctx
            .state
            .streams
            .get(&self.stream_a)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("dual stream missing `{}`", self.stream_a))?;
        let b = ctx
            .state
            .streams
            .get(&self.stream_b)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("dual stream missing `{}`", self.stream_b))?;
        let mut emit = Emit::from_ctx(ctx);
        let (na, nb) = (self.inner)(&mut emit, a, b)?;
        ctx.state.streams.insert(self.stream_a.clone(), na);
        ctx.state.streams.insert(self.stream_b.clone(), nb);
        Ok(input)
    }
}

/// Copy the active tensor flow into a named stream.
#[derive(Debug, Clone)]
pub struct StoreStreamStage {
    pub name: String,
}

impl StoreStreamStage {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: Option<FlowValue>,
    ) -> Result<Option<FlowValue>> {
        let v = input.ok_or_else(|| anyhow::anyhow!("StoreStream requires input"))?;
        ctx.state.streams.insert(self.name.clone(), v.clone());
        Ok(Some(v))
    }
}

/// Replace the active tensor flow from a named stream.
#[derive(Debug, Clone)]
pub struct LoadStreamStage {
    pub name: String,
}

impl LoadStreamStage {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: Option<FlowValue>,
    ) -> Result<Option<FlowValue>> {
        let _ = input;
        ctx.state
            .streams
            .get(&self.name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("LoadStream missing `{}`", self.name))
            .map(Some)
    }
}

pub(crate) fn stream_snapshot(state: &crate::context::FlowState) -> HashMap<String, FlowValue> {
    state.streams.clone()
}

pub fn dual_stream_stage(
    name: impl Into<String>,
    stream_a: impl Into<String>,
    stream_b: impl Into<String>,
    f: impl Fn(&mut Emit<'_>, FlowValue, FlowValue) -> Result<(FlowValue, FlowValue)>
        + Send
        + Sync
        + 'static,
) -> FlowStage {
    FlowStage::DualStream(DualStreamStage::new(name, stream_a, stream_b, f))
}
