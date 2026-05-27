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
//! Metal linear training forward ≈ CPU reference.

#![cfg(all(feature = "native-splat", target_os = "macos"))]

use rlx_splat::core::{Camera, GaussianScene};
use rlx_splat::reference::render_training_forward;

#[test]
fn metal_training_forward_matches_cpu_reference() {
    rlx_splat::register();
    let scene = rlx_splat::make_parity_scene();
    let camera = Camera::look_at(
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        0.1,
        20.0,
    );
    let bg = [0.1, 0.15, 0.2];
    let w = 64u32;
    let h = 64u32;
    let tile = 16u32;
    let tw = 4u32;
    let radius_scale = 1.6f32;
    let alpha_cutoff = 1.0 / 255.0;
    let max_steps = 8u32;
    let trans = 0.01f32;
    let max_list = 64 * 64 * 32;

    let cpu = render_training_forward(
        &scene,
        &camera,
        bg,
        w,
        h,
        tile,
        tw,
        radius_scale,
        alpha_cutoff,
        max_steps,
        trans,
        max_list,
    );
    let metal = rlx_metal::splat_training::render_training_forward_metal(
        &scene,
        &camera,
        bg,
        w,
        h,
        tile,
        tw,
        radius_scale,
        alpha_cutoff,
        max_steps,
        trans,
        max_list,
    );
    assert_eq!(cpu.rgba_linear.len(), metal.rgba_linear.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (a, b) in cpu.rgba_linear.iter().zip(&metal.rgba_linear) {
        assert!(a.is_finite() && b.is_finite());
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    assert!(cos > 0.999, "Metal training forward cosine vs CPU = {cos}");
}
