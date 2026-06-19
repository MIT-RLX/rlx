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

//! Per-infer active token count for ONNX control-flow custom kernels.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static ACTIVE_TOKENS: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_TOKENS_SET: AtomicBool = AtomicBool::new(false);

/// Hint the active padded token count for this infer (e.g. Kitten `[1, seq]` width).
///
/// Stored process-wide (not thread-local) so parallel CPU kernels (QMatMul pool,
/// Metal host fallback) observe the same active width as the caller thread.
pub fn set_active_token_count(count: Option<usize>) {
    match count.filter(|&n| n > 0) {
        Some(n) => {
            ACTIVE_TOKENS.store(n, Ordering::Release);
            ACTIVE_TOKENS_SET.store(true, Ordering::Release);
        }
        None => ACTIVE_TOKENS_SET.store(false, Ordering::Release),
    }
}

/// Active token count when set by [`set_active_token_count`].
pub fn active_token_count() -> Option<usize> {
    if ACTIVE_TOKENS_SET.load(Ordering::Acquire) {
        Some(ACTIVE_TOKENS.load(Ordering::Acquire))
    } else {
        None
    }
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

fn per_trip_scalars(data: &[i64], split_lens: &[i64], trip_count: usize) -> Vec<i64> {
    if split_lens.len() <= 1 {
        // ORT broadcasts scalar split length → one element per trip (value may be 0/1 in the buffer).
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
    if std::env::var("KITTEN_RLX_DEBUG_TRIP").is_ok_and(|v| v == "1") {
        eprintln!(
            "[onnx_cf] concat trips={trip_count} split_len={} mask_len={} wrote={pos}",
            split_lens.len(),
            duration_mask.len()
        );
    }
}

/// `Range(0, frame_count, 1)` for alignment indices (mel vocoder path).
pub fn alignment_range_ids(limit: &[i64], out: &mut [i64]) {
    let frames = limit.first().copied().unwrap_or(0).max(0) as usize;
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = if i < frames { i as i64 } else { 0 };
    }
}

/// Expand 1D i64 alignment buffer to runtime frame count.
pub fn expand_i64_align(data: &[i64], shape: &[i64], out: &mut [i64]) {
    let n = shape.first().copied().unwrap_or(0).max(0) as usize;
    out.fill(0);
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        *slot = data.get(i).copied().unwrap_or(0);
    }
}

/// Vocoder hop (Kitten mini 0.8): samples per alignment frame.
pub const SAMPLES_PER_ALIGNMENT_FRAME: i64 = 600;

/// Trim vocoder waveform on the time axis: `[start .. align_frames * hop - end_trim]`.
pub fn vocoder_waveform_slice(
    wave: &[f32],
    wave_shape: &[usize],
    time_axis: usize,
    align_frames: i64,
    out: &mut [f32],
    out_shape: &[usize],
) {
    let start = 10usize;
    let end_trim = 10usize;
    let frames = align_frames.max(0) as usize;
    let end = frames
        .saturating_mul(SAMPLES_PER_ALIGNMENT_FRAME as usize)
        .saturating_sub(end_trim);
    let copy_len = end.saturating_sub(start);
    out.fill(0.0);
    if copy_len == 0 || wave_shape.is_empty() || out_shape.is_empty() {
        return;
    }
    let rank = wave_shape.len().min(out_shape.len());
    let axis = time_axis.min(rank.saturating_sub(1));
    let in_time = wave_shape[axis];
    let out_time = out_shape[axis];
    let copy_len = copy_len.min(in_time.saturating_sub(start)).min(out_time);
    if rank == 1 {
        if copy_len > 0 && wave.len() >= start + copy_len {
            let n = copy_len.min(out.len());
            out[..n].copy_from_slice(&wave[start..start + n]);
        }
        return;
    }
    let in_stride: usize = wave_shape[axis + 1..].iter().product::<usize>().max(1);
    let out_stride: usize = out_shape[axis + 1..].iter().product::<usize>().max(1);
    let outer: usize = wave_shape[..axis].iter().product::<usize>().max(1);
    for outer_idx in 0..outer {
        let in_base = outer_idx * wave_shape[axis] * in_stride;
        let out_base = outer_idx * out_shape[axis] * out_stride;
        for t in 0..copy_len {
            let src = in_base + (start + t) * in_stride;
            let dst = out_base + t * out_stride;
            if src + in_stride <= wave.len() && dst + out_stride <= out.len() {
                out[dst..dst + out_stride].copy_from_slice(&wave[src..src + out_stride]);
            }
        }
    }
}

/// Resolve trip count from ORT tensor + optional runtime active token hint.
///
/// Kitten ONNX often binds trip count to compile-slot width (`80`) while the
/// runtime infer uses fewer active tokens (`74`). Always clamp to
/// [`active_token_count`] when set; substitute when the scalar trip is `0`/`1`.
pub fn resolve_concat_trip_count(
    trip: &[i64],
    duration_mask_len: usize,
    split_lens_len: usize,
) -> usize {
    let trip_from_tensor = trip.first().copied().unwrap_or(0).max(0) as usize;
    let mut trip_count = trip_from_tensor;
    if let Some(n) = active_token_count() {
        if n > 0 {
            trip_count = if trip_count <= 1 {
                n
            } else {
                trip_count.min(n)
            };
        }
    }
    let mask_cap = duration_mask_len.max(1);
    let split_cap = if split_lens_len <= 1 {
        // ORT broadcasts a scalar split length across the full duration mask.
        mask_cap
    } else {
        split_lens_len
    };
    let out = trip_count.min(mask_cap).min(split_cap);
    if std::env::var("KITTEN_RLX_DEBUG_TRIP").is_ok_and(|v| v == "1") {
        eprintln!(
            "[onnx_cf] trip_tensor={trip_from_tensor} active={:?} mask_len={duration_mask_len} \
             split_len={split_lens_len} -> {out}",
            active_token_count()
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_alignment_pattern() {
        let mask = vec![19i64, 2, 1, 2, 3, 2, 3, 2];
        let range = (0i64..8).collect::<Vec<_>>();
        let lens = vec![1i64; 8];
        let mut out = vec![0i64; 64];
        concat_alignment_durations(&mask, &range, &lens, 8, &mut out);
        assert_eq!(out[..34].iter().filter(|&&v| v >= 0).count(), 34);
    }

    #[test]
    fn trip_count_clamps_compile_slot_to_active_tokens() {
        set_active_token_count(Some(74));
        assert_eq!(resolve_concat_trip_count(&[80], 80, 80), 74);
        assert_eq!(resolve_concat_trip_count(&[1], 80, 80), 74);
        set_active_token_count(None);
    }

    #[test]
    fn concat_scalar_zero_split_lens() {
        let mask = vec![19i64, 2, 1, 2, 3, 2, 3, 2];
        let range = (0i64..8).collect::<Vec<_>>();
        let mut out = vec![0i64; 64];
        concat_alignment_durations(&mask, &range, &[0i64], 8, &mut out);
        let expected_sum: usize = mask.iter().map(|&d| d.max(0) as usize).sum();
        assert_eq!(
            out[..expected_sum].iter().filter(|&&v| v >= 0).count(),
            expected_sum
        );
    }

    #[test]
    fn concat_broadcast_scalar_split_lens() {
        let mask = vec![19i64, 2, 1, 2, 3, 2, 3, 2];
        let range = (0i64..8).collect::<Vec<_>>();
        let mut out_full = vec![0i64; 64];
        concat_alignment_durations(&mask, &range, &[1i64; 8], 8, &mut out_full);
        let mut out_scalar = vec![0i64; 64];
        concat_alignment_durations(&mask, &range, &[1i64], 8, &mut out_scalar);
        assert_eq!(out_scalar, out_full);
    }

    #[test]
    fn trip_count_broadcast_split_len() {
        set_active_token_count(Some(74));
        assert_eq!(resolve_concat_trip_count(&[1], 80, 1), 74);
        set_active_token_count(None);
    }

    #[test]
    fn trip_count_without_active_uses_tensor() {
        set_active_token_count(None);
        assert_eq!(resolve_concat_trip_count(&[8], 8, 8), 8);
    }
}
