// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
