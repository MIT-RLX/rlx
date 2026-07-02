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

//! Shared helpers for ONNX `Random*` lowering and codegen.

use rlx_ir::Op;

use crate::bundle::BundleNode;

/// Stable per-node key mixed into [`rlx_ir::RngOptions::seed`] for Philox streams.
///
/// Used by both [`crate::lower::ops`] and [`crate::emit_codegen`] so direct
/// import and emitted Rust sources agree on per-op RNG keys.
pub fn node_name_tag(name: &str) -> u64 {
    name.bytes()
        .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)))
}

/// ONNX `seed` attribute cast to f32, when present.
pub fn op_seed(node: &BundleNode) -> Option<f32> {
    node.attrs
        .get("seed")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
}

/// Parsed distribution parameters for ONNX `Random*` / `Random*Like`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RandomDistribution {
    Normal { mean: f32, scale: f32 },
    Uniform { low: f32, high: f32 },
}

pub fn distribution(node: &BundleNode) -> RandomDistribution {
    match node.op.as_str() {
        "RandomNormalLike" | "RandomNormal" => RandomDistribution::Normal {
            mean: attr_f32(node, "mean", 0.0),
            scale: attr_f32(node, "scale", 1.0),
        },
        _ => RandomDistribution::Uniform {
            low: attr_f32(node, "low", 0.0),
            high: attr_f32(node, "high", 1.0),
        },
    }
}

fn attr_f32(node: &BundleNode, name: &str, default: f64) -> f32 {
    node.attrs
        .get(name)
        .and_then(|v| v.as_f64())
        .unwrap_or(default) as f32
}

/// Custom-op name when [`crate::lower::ImportOptions::lower_random_as_custom`] is set.
pub fn custom_name(node: &BundleNode) -> &'static str {
    match node.op.as_str() {
        "RandomNormalLike" => "onnx.RandomNormalLike",
        "RandomUniformLike" => "onnx.RandomUniformLike",
        "RandomNormal" => "onnx.RandomNormal",
        _ => "onnx.RandomUniform",
    }
}

/// Packed attrs blob for custom random ops (mean/scale or low/high + node tag).
pub fn custom_attrs(dist: RandomDistribution, tag: u64) -> Vec<u8> {
    match dist {
        RandomDistribution::Normal { mean, scale } => pack_random_attrs(mean, scale, tag),
        RandomDistribution::Uniform { low, high } => pack_random_attrs(low, high, tag),
    }
}

fn pack_random_attrs(a: f32, b: f32, tag: u64) -> Vec<u8> {
    let mut v = vec![0u8; 16];
    v[0..4].copy_from_slice(&a.to_le_bytes());
    v[4..8].copy_from_slice(&b.to_le_bytes());
    v[8..16].copy_from_slice(&tag.to_le_bytes());
    v
}

/// Native IR op for a parsed ONNX random node.
pub fn rng_op(dist: RandomDistribution, tag: u64, op_seed: Option<f32>) -> Op {
    match dist {
        RandomDistribution::Normal { mean, scale } => Op::RngNormal {
            mean,
            scale,
            key: tag,
            op_seed,
        },
        RandomDistribution::Uniform { low, high } => Op::RngUniform {
            low,
            high,
            key: tag,
            op_seed,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn node(op: &str, attrs: &[(&str, f64)]) -> BundleNode {
        BundleNode {
            name: "rng".to_string(),
            op: op.to_string(),
            inputs: vec![],
            outputs: vec!["out".to_string()],
            attrs: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), serde_json::json!(v)))
                .collect::<HashMap<_, _>>(),
            output_meta: vec![],
        }
    }

    #[test]
    fn node_name_tag_is_deterministic() {
        assert_eq!(node_name_tag("rng"), node_name_tag("rng"));
        assert_ne!(node_name_tag("rng"), node_name_tag("rng2"));
    }

    #[test]
    fn node_name_tag_matches_fixture_node_names() {
        assert_eq!(node_name_tag("rng"), 0x1b9ab);
    }

    #[test]
    fn distribution_parses_normal_defaults() {
        let d = distribution(&node("RandomNormalLike", &[]));
        assert_eq!(
            d,
            RandomDistribution::Normal {
                mean: 0.0,
                scale: 1.0
            }
        );
    }

    #[test]
    fn distribution_parses_uniform_attrs() {
        let d = distribution(&node("RandomUniform", &[("low", 0.1), ("high", 0.9)]));
        assert_eq!(
            d,
            RandomDistribution::Uniform {
                low: 0.1,
                high: 0.9
            }
        );
    }
}
