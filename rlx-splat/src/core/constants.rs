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
//! Shared rendering constants (tile size, alpha cutoff, SH bands, …).

pub const SMALL_VALUE: f32 = 1e-6;
pub const TINY_VALUE: f32 = 1e-8;
pub const RAY_DENOMINATOR_FLOOR: f32 = 1e-10;
pub const POSITIVE_CONIC_FLOOR: f32 = 1e-10;
pub const CONIC_DETERMINANT_FLOOR: f32 = 1e-12;
pub const CONIC_AXIS_FLOOR: f32 = 1e-20;
pub const DEPTH_VISIBILITY_FLOOR: f32 = 1e-4;

pub const SPLAT_PIXEL_CLAMP_PX: f32 = 0.75;
pub const MIN_SCREEN_RADIUS_PX: f32 = 1.0;
pub const ELLIPSE_RADIUS_PAD_PX: f32 = 1.0;
pub const ELLIPSE_EPS: f32 = 1e-6;
pub const OUTPUT_GAMMA: f32 = 2.2;

pub const ALPHA_CUTOFF_DEFAULT: f32 = 1.0 / 255.0;
pub const GAUSSIAN_SUPPORT_SIGMA_RADIUS: f32 = 3.0;
pub const MIN_CONIC_DET: f32 = 1e-12;

pub const VEC_EPS: f32 = 1e-8;
pub const DISTORTION_EPS: f64 = 1e-12;
pub const DISTORTION_NEWTON_ITERS: usize = 8;

pub const DEFAULT_TILE_SIZE: u32 = 8;
pub const DEFAULT_RASTER_BATCH: u32 = 256;
pub const DEFAULT_MAX_SPLAT_STEPS: u32 = 32768;
pub const DEFAULT_TRANSMITTANCE_THRESHOLD: f32 = 1e-4;
pub const DEFAULT_RADIUS_SCALE: f32 = 1.0;
pub const DEFAULT_LIST_CAPACITY_MULTIPLIER: u32 = 32;
