// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ONNX control-flow helpers (`Loop`, `SplitToSequence`, `ConcatFromSequence`).

use std::collections::HashMap;

use crate::bundle::BundleNode;

/// Feedback tensor for duration fixed-point import (cycle break on `duration` inputs).
pub const DURATION_CARRY: &str = "__onnx_import__/duration_carry";

/// Runtime total alignment frame count (sum of per-token durations). Set before infer.
pub const ALIGNMENT_FRAME_COUNT: &str = "__onnx_runtime__/alignment_frame_count";

/// ONNX `ConcatFromSequence` output feeding mel alignment (`/Expand_3` path).
pub const CONCAT_FROM_SEQUENCE_OUTPUT: &str = "/ConcatFromSequence_output_0";

/// Inputs for duration `ConcatFromSequence` fusion (`SplitToSequence` + `Loop`).
#[derive(Debug, Clone)]
pub struct DurationAlignInputs {
    pub duration_mask: String,
    pub range_ids: String,
    pub split_lens: String,
    pub trip_count: String,
}

/// Break `duration` feedback for external carry iteration (`Expand` reads carry).
pub fn patch_duration_carry_inputs(nodes: &mut [BundleNode]) {
    for node in nodes.iter_mut() {
        if node.name.ends_with("/Expand_1") || node.name == "/Expand_1" {
            for inp in node.inputs.iter_mut() {
                if inp == "duration" {
                    *inp = DURATION_CARRY.to_string();
                }
            }
        }
    }
}

/// ORT single-pass: `/Where_1` false branch reads live `duration`, not stale carry.
pub fn patch_duration_where_live_input(nodes: &mut [BundleNode]) {
    for node in nodes.iter_mut() {
        if node.name != "/Where_1" || node.inputs.len() < 3 {
            continue;
        }
        if node.inputs[2] == DURATION_CARRY {
            node.inputs[2] = "duration".to_string();
        }
    }
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

/// Vocoder hop (Kitten mini 0.8): samples per alignment frame.
pub const SAMPLES_PER_ALIGNMENT_FRAME: usize = 600;

/// Upper bound on alignment frames for static compile shapes (`seq * max_frames_per_token`).
pub fn alignment_frame_upper_bound(sequence_length: usize, max_frames_per_token: usize) -> usize {
    sequence_length.saturating_mul(max_frames_per_token)
}

/// Static compile buffer for `ConcatFromSequence` / mel alignment (`Expand_3`, `Range_2`).
///
/// Must cover both per-token duration expansion and vocoder waveform length
/// (`max_waveform_samples / hop`).
pub fn alignment_buffer_upper_bound(
    sequence_length: usize,
    max_waveform_samples: usize,
    max_frames_per_token: usize,
) -> usize {
    max_waveform_samples
        .div_ceil(SAMPLES_PER_ALIGNMENT_FRAME)
        .max(alignment_frame_upper_bound(
            sequence_length,
            max_frames_per_token,
        ))
        .max(1)
}

fn producers_map(nodes: &[BundleNode]) -> HashMap<&str, &BundleNode> {
    nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
        .collect()
}

/// Whether `name` is the fused `ConcatFromSequence` buffer or is derived from it.
pub fn tensor_traces_concat_output(nodes: &[BundleNode], name: &str) -> bool {
    if name == CONCAT_FROM_SEQUENCE_OUTPUT {
        return true;
    }
    let producers = producers_map(nodes);
    let mut cur = name;
    let mut steps = 0usize;
    while steps < 32 {
        steps += 1;
        let Some(node) = producers.get(cur) else {
            return false;
        };
        if node.op == "ConcatFromSequence" {
            return true;
        }
        match node.op.as_str() {
            "Add" | "Expand" | "Unsqueeze" | "Squeeze" | "Identity" | "Cast" | "Gather"
            | "Where" | "Shape" | "Concat" | "Reshape" | "Slice" => {
                cur = node.inputs.first().map(|s| s.as_str()).unwrap_or("");
                if cur.is_empty() {
                    return false;
                }
            }
            _ => return false,
        }
    }
    false
}

/// Whether `name` feeds a runtime alignment length (Shape/Gather of concat output).
pub fn tensor_traces_alignment_length(nodes: &[BundleNode], name: &str) -> bool {
    if name == ALIGNMENT_FRAME_COUNT {
        return true;
    }
    let producers = producers_map(nodes);
    let Some(node) = producers.get(name) else {
        return false;
    };
    if node.op == "Gather" {
        let data = node.inputs.first().map(|s| s.as_str()).unwrap_or("");
        return tensor_traces_concat_output(nodes, data)
            || producers.get(data).is_some_and(|p| {
                p.op == "Shape"
                    && p.inputs
                        .first()
                        .is_some_and(|inp| tensor_traces_concat_output(nodes, inp))
            });
    }
    if node.op == "Shape" {
        return node
            .inputs
            .first()
            .is_some_and(|inp| tensor_traces_concat_output(nodes, inp));
    }
    false
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

fn per_trip_scalars(data: &[i64], split_lens: &[i64], trip_count: usize) -> Vec<i64> {
    if split_lens.len() <= 1 {
        return (0..trip_count)
            .map(|i| data.get(i).copied().unwrap_or(0))
            .collect();
    }
    let splits = split_1d(data, split_lens);
    (0..trip_count)
        .map(|i| splits.get(i).and_then(|v| v.first().copied()).unwrap_or(0))
        .collect()
}

/// Concatenate per-trip alignment rows (i64), matching ONNX `Loop` + `ConcatFromSequence`.
pub fn concat_alignment_durations(
    duration_mask: &[i64],
    range_ids: &[i64],
    split_lens: &[i64],
    trip_count: usize,
    out: &mut [i64],
) {
    let trips = per_trip_scalars(duration_mask, split_lens, trip_count);
    let ranges = per_trip_scalars(range_ids, split_lens, trip_count);
    let mut pos = 0usize;
    for i in 0..trip_count {
        let duration = trips.get(i).copied().unwrap_or(0);
        let rid = ranges.get(i).copied().unwrap_or(i as i64);
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
    fn traces_concat_through_expand() {
        use std::collections::HashMap;
        let nodes = vec![
            BundleNode {
                name: "/ConcatFromSequence".into(),
                op: "ConcatFromSequence".into(),
                inputs: vec!["/Loop_output_0".into()],
                outputs: vec![CONCAT_FROM_SEQUENCE_OUTPUT.into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
            BundleNode {
                name: "/Expand_3".into(),
                op: "Expand".into(),
                inputs: vec![
                    CONCAT_FROM_SEQUENCE_OUTPUT.into(),
                    "/Where_4_output_0".into(),
                ],
                outputs: vec!["/Expand_3_output_0".into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
        ];
        assert!(tensor_traces_concat_output(
            &nodes,
            CONCAT_FROM_SEQUENCE_OUTPUT
        ));
        assert!(tensor_traces_concat_output(&nodes, "/Expand_3_output_0"));
        assert!(!tensor_traces_concat_output(&nodes, "input_ids"));
    }

    #[test]
    fn alignment_buffer_covers_waveform_and_token_frames() {
        assert_eq!(
            super::alignment_buffer_upper_bound(74, 200_000, 512),
            74 * 512
        );
        assert_eq!(super::alignment_buffer_upper_bound(8, 48_000, 24), 8 * 24);
    }

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
