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
//! Packed arena layout for [`PreparedRaster`](crate::reference::native_prep::PreparedRaster).

use crate::reference::native_prep::{PreparedRaster, SplatRasterParams};

pub use rlx_ir::ops::splat::{
    gaussian_splat_prep_packed_len as prep_packed_len, gaussian_splat_tile_count as tile_count,
    GAUSSIAN_SPLAT_PREP_RASTER_PARAMS_FLOATS as SPLAT_RASTER_PARAMS_FLOATS,
};

fn copy_f32(out: &mut [f32], o: &mut usize, data: &[f32]) {
    let len = data.len();
    out[*o..*o + len].copy_from_slice(data);
    *o += len;
}

pub fn pack_prepared(out: &mut [f32], prep: &PreparedRaster, max_list_entries: u32) {
    let n = prep.valid.len().max(1);
    assert!(out.len() >= prep_packed_len(
        n,
        max_list_entries,
        prep.params.width,
        prep.params.height,
        prep.params.tile_size,
    ));

    let mut o = 0usize;
    copy_f32(out, &mut o, &prep.color_alpha);
    for v in &prep.valid {
        out[o] = *v as f32;
        o += 1;
    }
    copy_f32(out, &mut o, &prep.pos_local);
    copy_f32(out, &mut o, &prep.inv_scale);
    copy_f32(out, &mut o, &prep.quat);
    let list_cap = max_list_entries as usize;
    for i in 0..list_cap {
        out[o] = if i < prep.sorted_values.len() {
            prep.sorted_values[i] as f32
        } else {
            0.0
        };
        o += 1;
    }
    for v in &prep.tile_ranges {
        out[o] = *v as f32;
        o += 1;
    }
    copy_f32(out, &mut o, &prep.rays);
    let p = &prep.params;
    out[o..o + SPLAT_RASTER_PARAMS_FLOATS].copy_from_slice(&[
        p.width as f32,
        p.height as f32,
        p.tile_size as f32,
        p.tile_width as f32,
        p.alpha_cutoff,
        p.transmittance_threshold,
        p.bg_r,
        p.bg_g,
        p.bg_b,
        p.max_splat_steps as f32,
        0.0,
    ]);
}

pub fn unpack_prepared(packed: &[f32], count: usize, max_list_entries: u32, width: u32, height: u32, tile_size: u32) -> PreparedRaster {
    let n = count.max(1);
    let max_list = max_list_entries as usize;
    let tiles = tile_count(width, height, tile_size) as usize;
    let pixels = (width as usize).saturating_mul(height as usize).max(1);
    let need = prep_packed_len(n, max_list_entries, width, height, tile_size);
    assert!(packed.len() >= need);

    let mut o = 0usize;
    let take_f32 = |buf: &[f32], o: &mut usize, len: usize| -> Vec<f32> {
        let s = buf[*o..*o + len].to_vec();
        *o += len;
        s
    };
    let take_u32 = |buf: &[f32], o: &mut usize, len: usize| -> Vec<u32> {
        let v: Vec<u32> = buf[*o..*o + len].iter().map(|&x| x as u32).collect();
        *o += len;
        v
    };

    let color_alpha = take_f32(packed, &mut o, n * 4);
    let valid = take_u32(packed, &mut o, n);
    let pos_local = take_f32(packed, &mut o, n * 3);
    let inv_scale = take_f32(packed, &mut o, n * 3);
    let quat = take_f32(packed, &mut o, n * 4);
    let sorted_values = take_u32(packed, &mut o, max_list);
    let tile_ranges = take_u32(packed, &mut o, tiles * 2);
    let rays = take_f32(packed, &mut o, pixels * 3);
    let p = &packed[o..o + SPLAT_RASTER_PARAMS_FLOATS];
    let params = SplatRasterParams {
        width: p[0] as u32,
        height: p[1] as u32,
        tile_size: p[2] as u32,
        tile_width: p[3] as u32,
        alpha_cutoff: p[4],
        transmittance_threshold: p[5],
        bg_r: p[6],
        bg_g: p[7],
        bg_b: p[8],
        max_splat_steps: p[9] as u32,
    };

    PreparedRaster {
        color_alpha,
        valid,
        pos_local,
        inv_scale,
        quat,
        sorted_values,
        tile_ranges,
        rays,
        params,
    }
}
