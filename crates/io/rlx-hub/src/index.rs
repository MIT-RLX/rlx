// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parse a `model.safetensors.index.json` and plan which shard **files** a
//! subset of tensors needs — the core of downloading (or loading) only *part*
//! of a checkpoint across a distributed pipeline: give each node its stage's
//! layer range and it fetches only the shards those layers touch.

use anyhow::{Result, anyhow};
use std::collections::{BTreeSet, HashMap};
use std::ops::Range;

/// A safetensors index: `tensor name → shard filename`.
#[derive(Debug, Clone, Default)]
pub struct SafetensorsIndex {
    pub weight_map: HashMap<String, String>,
}

impl SafetensorsIndex {
    /// Parse `model.safetensors.index.json` bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(bytes)?;
        let wm = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("index.json has no `weight_map`"))?;
        let weight_map = wm
            .iter()
            .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();
        Ok(Self { weight_map })
    }

    /// All shard files, sorted.
    pub fn shards(&self) -> Vec<String> {
        self.weight_map.values().cloned().collect::<BTreeSet<_>>().into_iter().collect()
    }

    /// Shard files holding any tensor for which `pred` is true, sorted.
    pub fn shards_for(&self, pred: impl Fn(&str) -> bool) -> Vec<String> {
        let mut s = BTreeSet::new();
        for (t, f) in &self.weight_map {
            if pred(t) {
                s.insert(f.clone());
            }
        }
        s.into_iter().collect()
    }

    /// The transformer-layer index of a tensor named `…layers.N.…`, or `None`
    /// for layer-less tensors (embeddings, final norm, LM head).
    pub fn tensor_layer(name: &str) -> Option<usize> {
        let after = name.split("layers.").nth(1)?;
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() { None } else { digits.parse().ok() }
    }
}

/// The shard files one pipeline stage must download / load.
#[derive(Debug, Clone)]
pub struct StageShards {
    pub stage: usize,
    /// Contiguous layer range this stage owns.
    pub layers: Range<usize>,
    /// Shard files (sorted). Boundary shards appear in adjacent stages too, since
    /// a layer's tensors can straddle a shard boundary — every stage gets a
    /// *complete* set of its layers' tensors.
    pub shards: Vec<String>,
}

/// Split a checkpoint into `stages.len()` pipeline stages by contiguous layer
/// ranges, returning the shard files each stage needs. `extra[i]` names the
/// layer-less tensor *prefixes* (e.g. `["model.embed_tokens"]`,
/// `["lm_head", "model.norm"]`) that belong to stage `i` — typically embeddings
/// on the first stage and the LM head/final norm on the last.
pub fn plan_layer_stages(
    index: &SafetensorsIndex,
    stages: &[Range<usize>],
    extra: &[Vec<&str>],
) -> Vec<StageShards> {
    stages
        .iter()
        .enumerate()
        .map(|(i, range)| {
            let extras = extra.get(i).cloned().unwrap_or_default();
            let shards = index.shards_for(|t| match SafetensorsIndex::tensor_layer(t) {
                Some(l) => range.contains(&l),
                None => extras.iter().any(|p| t.starts_with(p)),
            });
            StageShards { stage: i, layers: range.clone(), shards }
        })
        .collect()
}

/// Evenly split `n_layers` into `n_stages` contiguous ranges (front stages get
/// the extra layer when it doesn't divide). A convenience for `plan_layer_stages`;
/// for heterogeneous nodes pass custom ranges instead.
pub fn even_layer_ranges(n_layers: usize, n_stages: usize) -> Vec<Range<usize>> {
    assert!(n_stages >= 1);
    let base = n_layers / n_stages;
    let rem = n_layers % n_stages;
    let mut out = Vec::with_capacity(n_stages);
    let mut start = 0;
    for i in 0..n_stages {
        let len = base + if i < rem { 1 } else { 0 };
        out.push(start..start + len);
        start += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx() -> SafetensorsIndex {
        // 4 layers across 3 shards; layer 1 straddles shard A/B, layer 2 straddles B/C.
        let mut weight_map = HashMap::new();
        let put = |m: &mut HashMap<String, String>, t: &str, s: &str| {
            m.insert(t.to_string(), s.to_string());
        };
        put(&mut weight_map, "model.embed_tokens.weight", "A");
        put(&mut weight_map, "model.layers.0.attn.weight", "A");
        put(&mut weight_map, "model.layers.1.attn.weight", "A");
        put(&mut weight_map, "model.layers.1.mlp.weight", "B"); // straddles A/B
        put(&mut weight_map, "model.layers.2.attn.weight", "B");
        put(&mut weight_map, "model.layers.2.mlp.weight", "C"); // straddles B/C
        put(&mut weight_map, "model.layers.3.attn.weight", "C");
        put(&mut weight_map, "model.norm.weight", "C");
        put(&mut weight_map, "lm_head.weight", "C");
        SafetensorsIndex { weight_map }
    }

    #[test]
    fn tensor_layer_parses() {
        assert_eq!(SafetensorsIndex::tensor_layer("model.layers.7.mlp.w"), Some(7));
        assert_eq!(SafetensorsIndex::tensor_layer("model.layers.42.x"), Some(42));
        assert_eq!(SafetensorsIndex::tensor_layer("lm_head.weight"), None);
        assert_eq!(SafetensorsIndex::tensor_layer("model.embed_tokens.weight"), None);
    }

    #[test]
    fn stages_get_complete_layers_with_boundary_overlap() {
        let index = idx();
        // stage0: layers 0-1 + embed; stage1: layers 2-3 + head/norm.
        let stages = plan_layer_stages(
            &index,
            &[0..2, 2..4],
            &[vec!["model.embed_tokens"], vec!["lm_head", "model.norm"]],
        );
        // stage0 owns layer 1 which straddles A/B → must have BOTH A and B.
        assert_eq!(stages[0].shards, vec!["A", "B"]);
        // stage1 owns layer 2 (B/C) + head/norm (C) → B and C.
        assert_eq!(stages[1].shards, vec!["B", "C"]);
        // boundary shard B is downloaded by both stages (complete layers > dedup).
        assert!(stages[0].shards.contains(&"B".to_string()));
        assert!(stages[1].shards.contains(&"B".to_string()));
    }

    #[test]
    fn even_ranges_front_loaded() {
        assert_eq!(even_layer_ranges(43, 3), vec![0..15, 15..29, 29..43]);
        assert_eq!(even_layer_ranges(4, 2), vec![0..2, 2..4]);
    }
}
