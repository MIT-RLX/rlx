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

//! ONNX control-flow helpers (`Loop`, `SplitToSequence`, `ConcatFromSequence`).

use std::collections::HashMap;

use crate::bundle::BundleNode;

/// Feedback tensor for duration fixed-point import (cycle break on `duration` inputs).
pub const DURATION_CARRY: &str = "__onnx_import__/duration_carry";

/// Inputs for duration `ConcatFromSequence` fusion (`SplitToSequence` + `Loop`).
#[derive(Debug, Clone)]
pub struct DurationAlignInputs {
    pub duration_mask: String,
    pub range_ids: String,
    pub split_lens: String,
    pub trip_count: String,
}

/// Discover duration-alignment tensors from the bundle graph.
pub fn resolve_duration_align_inputs(nodes: &[BundleNode]) -> Option<DurationAlignInputs> {
    let producers: HashMap<&str, &BundleNode> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
        .collect();

    let loop_node = nodes.iter().find(|n| n.op == "Loop")?;
    let trip_count = loop_node.inputs.first()?.clone();

    let splits: Vec<&BundleNode> = nodes.iter().filter(|n| n.op == "SplitToSequence").collect();
    if splits.len() < 2 {
        return None;
    }

    let mut duration_split: Option<&BundleNode> = None;
    let mut range_split: Option<&BundleNode> = None;
    for split in &splits {
        let inp0 = split.inputs.first()?;
        let from_reshape = producers
            .get(inp0.as_str())
            .is_some_and(|p| p.op == "Reshape");
        if from_reshape {
            range_split = Some(*split);
        } else {
            duration_split = Some(*split);
        }
    }
    let duration_split = duration_split.or_else(|| splits.first().copied())?;
    let range_split = range_split.or_else(|| splits.get(1).copied())?;
    if duration_split.name == range_split.name {
        return None;
    }

    let duration_mask = duration_split.inputs.first()?.clone();
    let split_lens = duration_split.inputs.get(1)?.clone();
    let range_ids = range_split.inputs.first()?.clone();

    if !producers.contains_key(trip_count.as_str()) {
        return None;
    }

    Some(DurationAlignInputs {
        duration_mask,
        range_ids,
        split_lens,
        trip_count,
    })
}

/// Upper bound on alignment frames for static compile shapes (`seq * max_frames_per_token`).
pub fn alignment_frame_upper_bound(sequence_length: usize, max_frames_per_token: usize) -> usize {
    sequence_length.saturating_mul(max_frames_per_token)
}

fn split_1d(data: &[i64], lens: &[i64]) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    for &len in lens {
        let n = len.max(0) as usize;
        let end = (pos + n).min(data.len());
        out.push(data[pos..end].to_vec());
        pos = end;
        if pos >= data.len() {
            break;
        }
    }
    if out.is_empty() && !data.is_empty() {
        out.push(data.to_vec());
    }
    out
}

fn loop_body_frame(duration: i64, range_id: i64) -> Vec<i64> {
    let d = duration.max(0) as usize;
    vec![range_id; d]
}

/// Total alignment frame count (sum of per-token durations).
pub fn alignment_frame_count(duration_mask: &[i64]) -> usize {
    duration_mask.iter().map(|&d| d.max(0) as usize).sum()
}

/// Concatenate per-trip alignment rows (i64), matching ONNX `Loop` + `ConcatFromSequence`.
pub fn concat_alignment_durations(
    duration_mask: &[i64],
    range_ids: &[i64],
    split_lens: &[i64],
    trip_count: usize,
    out: &mut [i64],
) {
    let split0 = split_1d(duration_mask, split_lens);
    let split1 = split_1d(range_ids, split_lens);
    let mut pos = 0usize;
    for i in 0..trip_count {
        let duration = split0.get(i).and_then(|v| v.first().copied()).unwrap_or(0);
        let rid = split1
            .get(i)
            .and_then(|v| v.first().copied())
            .unwrap_or(i as i64);
        let row = loop_body_frame(duration, rid);
        for v in row {
            if pos < out.len() {
                out[pos] = v;
                pos += 1;
            }
        }
    }
    for slot in out.iter_mut().skip(pos) {
        *slot = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concat_matches_reference_pattern() {
        let mask = vec![19i64, 2, 1, 2, 3, 2, 3, 2];
        let range = (0i64..8).collect::<Vec<_>>();
        let lens = vec![1i64; 8];
        let mut out = vec![0i64; 64];
        concat_alignment_durations(&mask, &range, &lens, 8, &mut out);
        let expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5,
            6, 6, 6, 7, 7,
        ];
        assert_eq!(&out[..expected.len()], expected);
        assert_eq!(alignment_frame_count(&mask), 34);
    }
}
