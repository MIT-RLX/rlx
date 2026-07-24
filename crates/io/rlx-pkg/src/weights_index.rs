// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! `weights/index.json` tensor catalog.

use crate::tier::{Codec, StorageTier};
use serde::{Deserialize, Serialize};

/// One weight tensor in the package catalog.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WeightEntry {
    pub name: String,
    pub shape: Vec<usize>,
    /// `f32` | `gguf_q8_0` | `gguf_tq2_0` | …
    pub scheme: String,
    #[serde(default = "default_layout")]
    pub layout: String,
    /// Path relative to package root (`__flat__` for flat packs).
    pub shard: String,
    pub offset: u64,
    /// Stored (on-disk) byte length.
    pub length: u64,
    /// Uncompressed length when compressed; defaults to `length`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_length: Option<u64>,
    #[serde(default)]
    pub tier: StorageTier,
    #[serde(default)]
    pub codec: Codec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

fn default_layout() -> String {
    "row_major".into()
}

impl WeightEntry {
    pub fn is_mmapable(&self) -> bool {
        self.tier == StorageTier::Hot && self.codec == Codec::None
    }
}

/// Weight catalog file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct WeightsIndex {
    pub tensors: Vec<WeightEntry>,
}

impl WeightsIndex {
    pub fn get(&self, name: &str) -> Option<&WeightEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }
}
