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
//! Canonical render settings for cross-language parity with `slang-splat/reference_impls/reference_cpu.py`.
//!
//! Used by `tools/parity_baseline.py`, Rust integration tests, and `slang-splat-rs` parity harness.

use crate::core::Camera;
use crate::reference::RenderParams;

/// Background RGB for the tiny parity frame (sRGB, same as Python baseline).
pub const PARITY_BACKGROUND: [f32; 3] = [0.1, 0.15, 0.2];

/// Pinhole camera for the tiny parity scene.
pub fn parity_camera() -> Camera {
    Camera::look_at(
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        0.1,
        20.0,
    )
}

/// Render params for `tiny_render_seed5` / `make_parity_scene` (matches `parity_baseline.py`).
pub fn parity_tiny_render_params() -> RenderParams {
    let width = 64u32;
    let height = 64u32;
    let tile_size = crate::core::DEFAULT_TILE_SIZE;
    RenderParams {
        width,
        height,
        tile_size,
        radius_scale: 1.6,
        alpha_cutoff: crate::core::ALPHA_CUTOFF_DEFAULT,
        max_splat_steps: crate::core::DEFAULT_MAX_SPLAT_STEPS,
        transmittance_threshold: crate::core::DEFAULT_TRANSMITTANCE_THRESHOLD,
        max_list_entries: width * height * crate::core::DEFAULT_LIST_CAPACITY_MULTIPLIER,
    }
}
