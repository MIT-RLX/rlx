// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Auto-tier heuristics and optional name allow-lists.

use crate::tier::StorageTier;
use crate::write::PackedWeight;
use std::collections::HashSet;

/// Options for [`apply_auto_tier`].
#[derive(Debug, Clone)]
pub struct AutoTierOptions {
    /// Names forced to hot (exact match).
    pub hot_names: HashSet<String>,
    /// If true, names containing these substrings become warm.
    pub warm_substrings: Vec<String>,
    /// Tensors larger than this (bytes) default to warm unless hot-listed.
    pub warm_min_bytes: usize,
}

impl Default for AutoTierOptions {
    fn default() -> Self {
        Self {
            hot_names: HashSet::new(),
            warm_substrings: vec![
                "expert".into(),
                "experts".into(),
                "lora".into(),
                "ffn_gate_exps".into(),
                "ffn_up_exps".into(),
                "ffn_down_exps".into(),
            ],
            warm_min_bytes: 16 << 20, // 16 MiB
        }
    }
}

/// Mutate `weights` tiers in place using heuristics / allow-lists.
///
/// Does **not** replace GGUF-class quantization — warm is for cold/redundant
/// host blobs, not a substitute for Q4_K.
pub fn apply_auto_tier(weights: &mut [PackedWeight], opts: &AutoTierOptions) {
    for w in weights.iter_mut() {
        if opts.hot_names.contains(&w.name) {
            w.tier = StorageTier::Hot;
            continue;
        }
        let lower = w.name.to_ascii_lowercase();
        if opts
            .warm_substrings
            .iter()
            .any(|s| lower.contains(&s.to_ascii_lowercase()))
        {
            w.tier = StorageTier::Warm;
            continue;
        }
        if w.data.len() >= opts.warm_min_bytes && w.tier == StorageTier::Hot {
            w.tier = StorageTier::Warm;
        }
    }
}
