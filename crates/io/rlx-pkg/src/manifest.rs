// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Root `rlx.json` package manifest.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Package format identifier (`"rlxp"`).
pub const FORMAT_NAME: &str = "rlxp";

/// Current structural schema version.
pub const FORMAT_VERSION: u32 = 1;

/// Minimum loader `compat_version` this crate supports.
pub const COMPAT_VERSION: u32 = 1;

/// Graph member reference inside the pack.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphRef {
    pub path: String,
    /// `bincode_graph_v1` today.
    pub encoding: String,
}

/// Weights catalog pointer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WeightsRef {
    pub index: String,
}

/// Named sidecar blob.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SidecarRef {
    pub id: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default)]
    pub codec: crate::tier::Codec,
}

/// Optional distribution section.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DistRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<String>,
}

/// Root package manifest (`rlx.json`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub format: String,
    pub format_version: u32,
    pub compat_version: u32,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<String>,
    pub graph: GraphRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<WeightsRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sidecars: Vec<SidecarRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dist: Option<DistRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_unix: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

impl Manifest {
    /// Build a v1 manifest for a named model.
    pub fn new_v1(name: impl Into<String>) -> Self {
        Self {
            format: FORMAT_NAME.to_string(),
            format_version: FORMAT_VERSION,
            compat_version: COMPAT_VERSION,
            name: name.into(),
            producer: None,
            features: Vec::new(),
            graph: GraphRef {
                path: "graph/mir.bin".into(),
                encoding: "bincode_graph_v1".into(),
            },
            weights: Some(WeightsRef {
                index: "weights/index.json".into(),
            }),
            sidecars: Vec::new(),
            dist: None,
            created_unix: None,
            extensions: BTreeMap::new(),
        }
    }

    /// Reject unsupported format / compat versions.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.format != FORMAT_NAME {
            anyhow::bail!(
                "unknown package format {:?} (expected {:?})",
                self.format,
                FORMAT_NAME
            );
        }
        if self.format_version > FORMAT_VERSION {
            anyhow::bail!(
                "package format_version {} newer than loader {}",
                self.format_version,
                FORMAT_VERSION
            );
        }
        if self.compat_version > COMPAT_VERSION {
            anyhow::bail!(
                "package requires compat_version {} (loader supports {})",
                self.compat_version,
                COMPAT_VERSION
            );
        }
        if self.graph.encoding != "bincode_graph_v1" && self.graph.encoding != "none" {
            anyhow::bail!("unsupported graph encoding {:?}", self.graph.encoding);
        }
        Ok(())
    }
}
