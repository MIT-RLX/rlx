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
use super::project::ProjectedSplats;
use crate::core::{ELLIPSE_EPS, MIN_CONIC_DET};

fn try_prepare_conic_for_binning(conic: [f32; 3], radius: f32, half_tile: f32) -> (bool, [f32; 2]) {
    let det = conic[0] * conic[2] - conic[1] * conic[1];
    let mut bbox = [radius, radius];
    if !conic[0].is_finite()
        || !conic[1].is_finite()
        || !conic[2].is_finite()
        || conic[0] <= 1e-10
        || conic[2] <= 1e-10
        || det <= MIN_CONIC_DET
    {
        return (false, bbox);
    }
    let trace = conic[0] + conic[2];
    let disc = (0.25 * trace * trace - det).max(0.0).sqrt();
    let eig_min = 0.5 * trace - disc;
    let eig_max = 0.5 * trace + disc;
    let max_axis = 1.0 / eig_min.max(1e-20).sqrt();
    let min_axis = 1.0 / eig_max.max(1e-20).sqrt();
    let axis_limit = radius.max(1.0) * 2.0 + half_tile;
    if !max_axis.is_finite()
        || !min_axis.is_finite()
        || eig_min <= 0.0
        || eig_max <= 0.0
        || max_axis > axis_limit
        || min_axis < 0.125
    {
        return (false, bbox);
    }
    let extent = [
        (conic[2] / det).max(0.0).sqrt(),
        (conic[0] / det).max(0.0).sqrt(),
    ];
    if !extent[0].is_finite() || !extent[1].is_finite() {
        return (false, bbox);
    }
    bbox = [
        extent[0].clamp(1e-4, radius),
        extent[1].clamp(1e-4, radius),
    ];
    (true, bbox)
}

fn eval_conic(conic: [f32; 3], x: f32, y: f32) -> f32 {
    conic[0] * x * x + 2.0 * conic[1] * x * y + conic[2] * y * y
}

fn min_conic_over_tile_box(conic: [f32; 3], x0: f32, x1: f32, y0: f32, y1: f32) -> f32 {
    let mut values = vec![
        eval_conic(conic, x0, y0),
        eval_conic(conic, x1, y0),
        eval_conic(conic, x0, y1),
        eval_conic(conic, x1, y1),
    ];
    if x0 <= 0.0 && 0.0 <= x1 && y0 <= 0.0 && 0.0 <= y1 {
        return 0.0;
    }
    let (a, b, c) = (conic[0], conic[1], conic[2]);
    if c > 1e-12 {
        for x in [x0, x1] {
            let y = (-b * x / c).clamp(y0, y1);
            values.push(eval_conic(conic, x, y));
        }
    }
    if a > 1e-12 {
        for y in [y0, y1] {
            let x = (-b * y / a).clamp(x0, x1);
            values.push(eval_conic(conic, x, y));
        }
    }
    values.into_iter().reduce(f32::min).unwrap_or(0.0)
}

fn tile_box(
    center: (f32, f32),
    scan_along_x: bool,
    tile_size: u32,
    line_tile: i32,
    minor_tile: i32,
) -> (f32, f32, f32, f32) {
    let (major_x, major_y) = if scan_along_x {
        (minor_tile as f32, line_tile as f32)
    } else {
        (line_tile as f32, minor_tile as f32)
    };
    let ts = tile_size as f32;
    let lo_x = major_x * ts - center.0;
    let hi_x = (major_x + 1.0) * ts - center.0;
    let lo_y = major_y * ts - center.1;
    let hi_y = (major_y + 1.0) * ts - center.1;
    (lo_x, hi_x, lo_y, hi_y)
}

fn tile_intersects_ellipse(
    center: (f32, f32),
    conic: [f32; 3],
    scan_along_x: bool,
    tile_size: u32,
    line_tile: i32,
    minor_tile: i32,
) -> bool {
    let (x0, x1, y0, y1) = tile_box(center, scan_along_x, tile_size, line_tile, minor_tile);
    min_conic_over_tile_box(conic, x0, x1, y0, y1) <= 1.0 + ELLIPSE_EPS
}

fn compute_scanline_tile_span(
    center: (f32, f32),
    conic: [f32; 3],
    scan_along_x: bool,
    tile_size: u32,
    line_coord_tile: i32,
    min_minor_tile: i32,
    max_minor_tile: i32,
) -> (bool, i32, i32) {
    let mut hits = Vec::new();
    for minor in min_minor_tile.max(0)..=max_minor_tile {
        if tile_intersects_ellipse(center, conic, scan_along_x, tile_size, line_coord_tile, minor) {
            hits.push(minor);
        }
    }
    if hits.is_empty() {
        (false, 0, 0)
    } else {
        (true, hits[0], hits[hits.len() - 1] - hits[0] + 1)
    }
}

fn write_span(
    keys: &mut [u32],
    values: &mut [u32],
    mut write_index: usize,
    count: i32,
    tile_width: u32,
    splat_id: u32,
    scan_along_x: bool,
    primary: i32,
    minor_start: i32,
) -> usize {
    for offset in 0..count {
        let (tile_x, tile_y) = if scan_along_x {
            (minor_start + offset, primary)
        } else {
            (primary, minor_start + offset)
        };
        let tile_id = (tile_y as u32) * tile_width + tile_x as u32;
        keys[write_index] = tile_id;
        values[write_index] = splat_id;
        write_index += 1;
    }
    write_index
}

pub fn build_tile_key_value_pairs(
    projected: &ProjectedSplats,
    tile_width: u32,
    tile_height: u32,
    tile_size: u32,
    max_list_entries: u32,
) -> (Vec<u32>, Vec<u32>, u32) {
    let mut keys = vec![0u32; max_list_entries as usize];
    let mut values = vec![0u32; max_list_entries as usize];
    let mut counter = 0u32;
    let count = projected.valid.len();
    let mut visible: Vec<(usize, f32)> = Vec::new();
    for i in 0..count {
        if projected.valid[i] != 0 {
            visible.push((i, projected.center_radius_depth[i * 4 + 3]));
        }
    }
    visible.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (splat_id, _) in visible {
        let cx = projected.center_radius_depth[splat_id * 4];
        let cy = projected.center_radius_depth[splat_id * 4 + 1];
        let radius = projected.center_radius_depth[splat_id * 4 + 2];
        if projected.valid[splat_id] == 0 {
            continue;
        }
        let conic = [
            projected.ellipse_conic[splat_id * 3],
            projected.ellipse_conic[splat_id * 3 + 1],
            projected.ellipse_conic[splat_id * 3 + 2],
        ];
        let (use_conic, bbox_extent) =
            try_prepare_conic_for_binning(conic, radius, 0.5 * tile_size as f32);
        if !use_conic {
            continue;
        }
        let extent = [
            bbox_extent[0] + 0.5 * tile_size as f32,
            bbox_extent[1] + 0.5 * tile_size as f32,
        ];
        let min_x = ((cx - extent[0]) / tile_size as f32).floor().max(0.0) as i32;
        let max_x = ((cx + extent[0]) / tile_size as f32)
            .ceil()
            .min((tile_width - 1) as f32) as i32;
        let min_y = ((cy - extent[1]) / tile_size as f32).floor().max(0.0) as i32;
        let max_y = ((cy + extent[1]) / tile_size as f32)
            .ceil()
            .min((tile_height - 1) as f32) as i32;
        if min_x > max_x || min_y > max_y {
            continue;
        }
        let scan_along_x = bbox_extent[0] > bbox_extent[1]
            || ((bbox_extent[0] - bbox_extent[1]).abs() < 1e-6 && (max_x - min_x) >= (max_y - min_y));
        let (primary_lo, primary_hi, minor_lo, minor_hi) = if scan_along_x {
            (min_y, max_y, min_x, max_x)
        } else {
            (min_x, max_x, min_y, max_y)
        };
        let mut spans = Vec::new();
        let mut total_count = 0i32;
        for primary in primary_lo..=primary_hi {
            let (has_span, minor_start, span_count) = compute_scanline_tile_span(
                (cx, cy),
                conic,
                scan_along_x,
                tile_size,
                primary,
                minor_lo,
                minor_hi,
            );
            if has_span {
                spans.push((primary, minor_start, span_count));
                total_count += span_count;
            }
        }
        if total_count <= 0 {
            continue;
        }
        let base_index = counter;
        counter += total_count as u32;
        if base_index >= max_list_entries {
            continue;
        }
        let write_limit = total_count.min((max_list_entries - base_index) as i32);
        let mut write_index = base_index as usize;
        let mut written = 0i32;
        for (primary, minor_start, span_count) in spans {
            if written >= write_limit {
                break;
            }
            let count = span_count.min(write_limit - written);
            write_index = write_span(
                &mut keys,
                &mut values,
                write_index,
                count,
                tile_width,
                splat_id as u32,
                scan_along_x,
                primary,
                minor_start,
            );
            written += count;
        }
    }
    (keys, values, counter)
}

pub fn sort_key_values(keys: &[u32], values: &[u32], count: u32) -> (Vec<u32>, Vec<u32>) {
    let count = count as usize;
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|&i| keys[i]);
    let sorted_keys: Vec<u32> = order.iter().map(|&i| keys[i]).collect();
    let sorted_values: Vec<u32> = order.iter().map(|&i| values[i]).collect();
    (sorted_keys, sorted_values)
}

pub fn build_tile_ranges(sorted_keys: &[u32], sorted_count: u32, tile_count: u32) -> Vec<u32> {
    let mut ranges = vec![0xFFFF_FFFFu32; (tile_count * 2) as usize];
    for i in (0..tile_count as usize).map(|i| i * 2 + 1) {
        ranges[i] = 0;
    }
    let sorted_count = sorted_count as usize;
    if sorted_count == 0 {
        return ranges;
    }
    let mut start = 0usize;
    while start < sorted_count {
        let tile = sorted_keys[start];
        let mut end = start + 1;
        while end < sorted_count && sorted_keys[end] == tile {
            end += 1;
        }
        let base = tile as usize * 2;
        if base + 1 < ranges.len() {
            ranges[base] = start as u32;
            ranges[base + 1] = end as u32;
        }
        start = end;
    }
    ranges
}
