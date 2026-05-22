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
//! Round-trip `pack_prepared` → `unpack_prepared` at full `prep_packed_len`.

use rlx_splat::prep_layout::{pack_prepared, prep_packed_len, unpack_prepared};
use rlx_splat::reference::native_prep::prepare_raster_from_slices;

fn tiny_scene() -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    let positions = vec![0.0, 0.0, 2.0, 0.5, 0.0, 2.5];
    let scales = vec![0.1, 0.1, 0.1, 0.12, 0.12, 0.12];
    let rotations = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let opacities = vec![0.9, 0.8];
    let colors = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let sh_coeffs = vec![0.0; 6];
    let meta = vec![
        0.0, 0.0, 0.0, // eye
        0.0, 0.0, 1.0, // target
        0.0, 1.0, 0.0, // up
        1.0, 1.0, 60.0, 45.0, // fx fy fov
        0.1, 0.1, 0.15, // bg
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    (
        positions, scales, rotations, opacities, colors, sh_coeffs, meta,
    )
}

#[test]
fn prep_pack_unpack_roundtrip() {
    let (positions, scales, rotations, opacities, colors, sh_coeffs, meta) = tiny_scene();
    let width = 32u32;
    let height = 24u32;
    let tile_size = 16u32;
    let max_list_entries = 64u32;
    let count = positions.len() / 3;

    let prep = prepare_raster_from_slices(
        &positions,
        &scales,
        &rotations,
        &opacities,
        &colors,
        &sh_coeffs,
        &meta,
        width,
        height,
        tile_size,
        1.0,
        0.01,
        8,
        0.05,
        max_list_entries,
    );

    let len = prep_packed_len(count, max_list_entries, width, height, tile_size);
    let mut packed = vec![0.0f32; len];
    pack_prepared(&mut packed, &prep, max_list_entries);

    let back = unpack_prepared(&packed, count, max_list_entries, width, height, tile_size);

    assert_eq!(back.color_alpha, prep.color_alpha);
    assert_eq!(back.valid, prep.valid);
    assert_eq!(back.pos_local, prep.pos_local);
    assert_eq!(back.inv_scale, prep.inv_scale);
    assert_eq!(back.quat, prep.quat);
    let mut expected_sorted = vec![0u32; max_list_entries as usize];
    for (i, v) in prep.sorted_values.iter().enumerate() {
        expected_sorted[i] = *v;
    }
    assert_eq!(back.sorted_values, expected_sorted);
    assert_eq!(back.tile_ranges, prep.tile_ranges);
    assert_eq!(back.rays, prep.rays);
    assert_eq!(back.params.width, prep.params.width);
    assert_eq!(back.params.height, prep.params.height);
    assert_eq!(back.params.tile_size, prep.params.tile_size);
    assert_eq!(back.params.tile_width, prep.params.tile_width);
    assert_eq!(back.params.alpha_cutoff, prep.params.alpha_cutoff);
    assert_eq!(
        back.params.transmittance_threshold,
        prep.params.transmittance_threshold
    );
    assert_eq!(back.params.bg_r, prep.params.bg_r);
    assert_eq!(back.params.bg_g, prep.params.bg_g);
    assert_eq!(back.params.bg_b, prep.params.bg_b);
}
