// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! `dist/placement.json` — TP/PP/EP maps.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How one tensor is sharded across ranks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorShard {
    /// Axis along which the tensor is split.
    pub dim: usize,
    /// Ranks that own a slice (in order along `dim`).
    pub ranks: Vec<u32>,
}

/// Expert-parallel placement for one expert id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertPlacement {
    /// Primary owning rank.
    pub rank: u32,
    /// Optional replica ranks (hot-expert copies).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub replicas: Vec<u32>,
}

/// Distribution placement metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Placement {
    /// Parallelism kinds present: `dp`, `tp`, `pp`, `ep`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parallelism: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology: Option<String>,
    /// Tensor name → shard map.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tensors: BTreeMap<String, TensorShard>,
    /// Expert id (string) → placement.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub experts: BTreeMap<String, ExpertPlacement>,
}

impl Placement {
    /// Ranks that own `tensor`, if listed.
    pub fn ranks_for(&self, tensor: &str) -> Option<&[u32]> {
        self.tensors.get(tensor).map(|s| s.ranks.as_slice())
    }

    /// Whether this pack expects a physical `dist/rank-{r}/` tree.
    pub fn has_tensor_map(&self) -> bool {
        !self.tensors.is_empty() || !self.experts.is_empty()
    }
}
