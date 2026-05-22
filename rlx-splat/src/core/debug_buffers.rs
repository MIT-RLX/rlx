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
//! Projection-stage debug buffers shared by CPU reference and GPU paths.

#[derive(Clone, Debug, Default)]
pub struct ProjectionDebugBuffers {
    pub screen_center_radius_depth: Vec<f32>,
    pub screen_color_alpha: Vec<f32>,
    pub screen_ellipse_conic: Vec<f32>,
    pub splat_visible: Vec<u32>,
    pub splat_visible_area_px: Vec<f32>,
    pub raster_cache: Vec<f32>,
    /// Total tile-list entries generated during binning.
    pub generated_entries: u32,
    pub sorted_count: u32,
    pub keys: Vec<u32>,
    pub values: Vec<u32>,
    pub tile_ranges: Vec<u32>,
}

pub const RASTER_CACHE_PARAM_COUNT: usize = 13;
