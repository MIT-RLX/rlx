// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Type-erased arch blocks — keep model-specific emission out of the core enum.

use crate::blocks::CustomStage;
use crate::escape::Emit;
use crate::stage::FlowStage;
use crate::value::FlowValue;

/// Named plugin stage (alias over tier-2 custom emission).
pub struct PluginStage(CustomStage);

impl PluginStage {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&mut Emit<'_>, Option<FlowValue>) -> anyhow::Result<Option<FlowValue>>
            + Send
            + Sync
            + 'static,
    {
        Self(CustomStage::new(f))
    }

    pub fn named<F>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&mut Emit<'_>, Option<FlowValue>) -> anyhow::Result<Option<FlowValue>>
            + Send
            + Sync
            + 'static,
    {
        Self(CustomStage::named(name, f))
    }

    pub(crate) fn into_stage(self) -> FlowStage {
        FlowStage::Custom(self.0)
    }
}

pub fn plugin<F>(f: F) -> FlowStage
where
    F: Fn(&mut Emit<'_>, Option<FlowValue>) -> anyhow::Result<Option<FlowValue>>
        + Send
        + Sync
        + 'static,
{
    PluginStage::new(f).into_stage()
}

pub fn plugin_named<F>(name: impl Into<String>, f: F) -> FlowStage
where
    F: Fn(&mut Emit<'_>, Option<FlowValue>) -> anyhow::Result<Option<FlowValue>>
        + Send
        + Sync
        + 'static,
{
    PluginStage::named(name, f).into_stage()
}
