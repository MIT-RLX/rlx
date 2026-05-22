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
//! JSONL trajectory logging for offline flow-map training.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// One design point along an optimization or placement run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    pub topology: String,
    pub state_id: String,
    pub action: Vec<f64>,
    pub reward: f64,
    /// Minimization loss (lower is better); mirrors harness score when present.
    pub loss: f64,
    #[serde(default)]
    pub noise: Option<Vec<f64>>,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl TrajectoryRecord {
    pub fn new(
        topology: impl Into<String>,
        state_id: impl Into<String>,
        action: Vec<f64>,
        loss: f64,
    ) -> Self {
        Self {
            topology: topology.into(),
            state_id: state_id.into(),
            action,
            reward: -loss,
            loss,
            noise: None,
            tags: Vec::new(),
        }
    }
}

pub fn append_jsonl(path: &Path, rec: &TrajectoryRecord) -> std::io::Result<()> {
    let line = serde_json::to_string(rec).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{line}")?;
    Ok(())
}

pub fn load_jsonl(path: &Path) -> std::io::Result<Vec<TrajectoryRecord>> {
    let f = std::fs::File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let rec: TrajectoryRecord = serde_json::from_str(t)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        out.push(rec);
    }
    Ok(out)
}

/// Extract `(action, target_velocity)` pairs for diagonal flow matching: v* ≈ a₁ − a₀.
pub fn diagonal_flow_pairs(records: &[TrajectoryRecord]) -> Vec<(Vec<f64>, Vec<f64>)> {
    records
        .iter()
        .filter_map(|r| {
            let a0 = r.noise.as_ref()?;
            if a0.len() != r.action.len() {
                return None;
            }
            let vel: Vec<f64> = r
                .action
                .iter()
                .zip(a0.iter())
                .map(|(a1, a0)| a1 - a0)
                .collect();
            Some((r.action.clone(), vel))
        })
        .collect()
}
