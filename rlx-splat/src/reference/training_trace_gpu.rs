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
//! GPU training trace layout (Metal) ↔ CPU [`PixelTrace`].

use super::training::{PixelTrace, SplatHit};

/// Floats per hit on GPU: `alpha`, `trans_before`, `color_rgb`.
pub const TRAINING_HIT_META_FLOATS: usize = 6;

pub fn trace_buffer_sizes(width: u32, height: u32, max_splat_steps: u32) -> (usize, usize, usize) {
    let pixels = (width * height) as usize;
    let cap = max_splat_steps.max(1) as usize;
    (pixels, pixels * cap, pixels * cap * TRAINING_HIT_META_FLOATS)
}

/// Rebuild per-pixel traces from GPU readback (for geometry backprop on CPU).
pub fn traces_from_gpu_buffers(
    hit_counts: &[u32],
    hit_splat_ids: &[u32],
    hit_meta: &[f32],
    width: u32,
    height: u32,
    max_splat_steps: u32,
) -> Vec<PixelTrace> {
    let pixels = (width * height) as usize;
    let cap = max_splat_steps.max(1) as usize;
    let mut traces = vec![PixelTrace::default(); pixels];
    for pix in 0..pixels {
        let n = hit_counts[pix].min(cap as u32) as usize;
        let mut trans_final = 1.0f32;
        for step in 0..n {
            let idx = pix * cap + step;
            let splat_id = hit_splat_ids[idx];
            let mb = idx * TRAINING_HIT_META_FLOATS;
            let alpha = hit_meta[mb];
            let trans_before = hit_meta[mb + 1];
            let color = [hit_meta[mb + 2], hit_meta[mb + 3], hit_meta[mb + 4]];
            trans_final *= 1.0 - alpha;
            traces[pix].hits.push(SplatHit {
                splat_id,
                alpha,
                trans_before,
                color,
            });
        }
        traces[pix].trans_final = trans_final;
    }
    traces
}
