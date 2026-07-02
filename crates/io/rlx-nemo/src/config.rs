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

//! Thin typed view over a NeMo `model_config.yaml`. Models read the
//! hyperparameters they need via dotted paths (`encoder.d_model`,
//! `preprocessor.features`, …) so nothing is hard-coded.

use anyhow::{Context, Result};
use serde_yaml::Value as Yaml;

/// Parsed `model_config.yaml`.
#[derive(Debug, Clone)]
pub struct NemoConfig {
    root: Yaml,
}

impl NemoConfig {
    pub fn from_yaml_bytes(bytes: &[u8]) -> Result<Self> {
        let root: Yaml = serde_yaml::from_slice(bytes).context("parse model_config.yaml")?;
        Ok(Self { root })
    }

    /// The raw YAML root, for callers that want to walk it directly.
    pub fn root(&self) -> &Yaml {
        &self.root
    }

    /// Descend a dotted path through nested mappings (e.g. `encoder.d_model`).
    pub fn get(&self, dotted: &str) -> Option<&Yaml> {
        let mut node = &self.root;
        for seg in dotted.split('.') {
            node = node.get(seg)?;
        }
        Some(node)
    }

    pub fn get_i64(&self, dotted: &str) -> Option<i64> {
        match self.get(dotted)? {
            Yaml::Number(n) => n.as_i64(),
            Yaml::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    pub fn get_usize(&self, dotted: &str) -> Option<usize> {
        self.get_i64(dotted).and_then(|v| usize::try_from(v).ok())
    }

    pub fn get_f64(&self, dotted: &str) -> Option<f64> {
        match self.get(dotted)? {
            Yaml::Number(n) => n.as_f64(),
            Yaml::String(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    pub fn get_bool(&self, dotted: &str) -> Option<bool> {
        match self.get(dotted)? {
            Yaml::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn get_str(&self, dotted: &str) -> Option<&str> {
        match self.get(dotted)? {
            Yaml::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// A flat list of integers, e.g. `att_context_size: [56, 13]`.
    /// Returns `None` if the node is missing or any element is non-integer
    /// (e.g. a list-of-lists — use [`Self::get_i64_matrix`] for that).
    pub fn get_i64_vec(&self, dotted: &str) -> Option<Vec<i64>> {
        let seq = self.get(dotted)?.as_sequence()?;
        seq.iter()
            .map(|v| match v {
                Yaml::Number(n) => n.as_i64(),
                _ => None,
            })
            .collect()
    }

    /// A list of integer lists, e.g. cache-aware `att_context_size:
    /// [[70, 13], [70, 6], …]`. Returns `None` unless every element is
    /// itself an all-integer sequence.
    pub fn get_i64_matrix(&self, dotted: &str) -> Option<Vec<Vec<i64>>> {
        let seq = self.get(dotted)?.as_sequence()?;
        seq.iter()
            .map(|row| {
                let r = row.as_sequence()?;
                r.iter()
                    .map(|v| match v {
                        Yaml::Number(n) => n.as_i64(),
                        _ => None,
                    })
                    .collect::<Option<Vec<i64>>>()
            })
            .collect()
    }

    /// Length of a sequence node, if present.
    pub fn seq_len(&self, dotted: &str) -> Option<usize> {
        self.get(dotted)?.as_sequence().map(|s| s.len())
    }

    /// Read a `string -> integer` mapping (e.g. `prompt_dictionary:
    /// {en-US: 0, …}`) as `(key, value)` pairs.
    pub fn get_str_i64_map(&self, dotted: &str) -> Option<Vec<(String, i64)>> {
        let m = self.get(dotted)?.as_mapping()?;
        let mut out = Vec::new();
        for (k, v) in m {
            if let (Some(ks), Some(vi)) = (k.as_str(), v.as_i64()) {
                out.push((ks.to_string(), vi));
            }
        }
        Some(out)
    }

    /// Convenience: read the first present path among several aliases.
    pub fn get_i64_any(&self, dotted: &[&str]) -> Option<i64> {
        dotted.iter().find_map(|p| self.get_i64(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotted_access() {
        let yaml = b"encoder:\n  d_model: 1024\n  n_layers: 24\n  att_context_size: [56, 13]\npreprocessor:\n  features: 80\n  normalize: per_feature\n";
        let c = NemoConfig::from_yaml_bytes(yaml).unwrap();
        assert_eq!(c.get_usize("encoder.d_model"), Some(1024));
        assert_eq!(c.get_usize("encoder.n_layers"), Some(24));
        assert_eq!(c.get_usize("preprocessor.features"), Some(80));
        assert_eq!(c.get_str("preprocessor.normalize"), Some("per_feature"));
        assert_eq!(
            c.get_i64_vec("encoder.att_context_size"),
            Some(vec![56, 13])
        );
        assert_eq!(c.get_usize("encoder.missing"), None);
    }
}
