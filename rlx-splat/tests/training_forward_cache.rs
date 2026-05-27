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
//! Cached training forward avoids a second forward inside backward.

use rlx_splat::core::{Camera, GaussianScene};
use rlx_splat::reference::{
    backward_packed_from_training_forward, backward_packed_host_slices,
    clear_training_forward_cache, render_training_forward, set_training_forward_cache,
};

#[test]
fn cached_backward_matches_rerun_forward() {
    rlx_splat::register();
    let scene = GaussianScene::new(
        vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0],
        vec![0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.8, 0.7],
        vec![0.5, 0.6, 0.4, 0.3, 0.2, 0.1],
        vec![0.0; 6],
        1,
    );
    let camera = Camera::look_at(
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        0.1,
        20.0,
    );
    let background = [0.1, 0.15, 0.2];
    let width = 32u32;
    let height = 32u32;
    let tile_size = 16u32;
    let tile_width = 2u32;
    let forward = render_training_forward(
        &scene,
        &camera,
        background,
        width,
        height,
        tile_size,
        tile_width,
        1.6,
        1.0 / 255.0,
        8,
        0.01,
        32 * 32 * 16,
    );
    let pixel_count = (width * height) as usize;
    let mut pixel_rgb_grad = vec![0.001f32; pixel_count * 3];
    pixel_rgb_grad[0] = 0.01;

    let cached = backward_packed_from_training_forward(
        &scene,
        &camera,
        &forward,
        &pixel_rgb_grad,
        background,
        width,
        height,
        1.6,
        1.0 / 255.0,
        10.0,
        0,
        1.0,
    );

    let wh4 = pixel_count * 4;
    let mut d_loss = vec![0.0f32; wh4];
    for pix in 0..pixel_count {
        let b3 = pix * 3;
        let b4 = pix * 4;
        d_loss[b4] = pixel_rgb_grad[b3];
        d_loss[b4 + 1] = pixel_rgb_grad[b3 + 1];
        d_loss[b4 + 2] = pixel_rgb_grad[b3 + 2];
    }

    set_training_forward_cache(&forward);
    let via_cache = backward_packed_host_slices(
        &scene.positions,
        &scene.scales,
        &scene.rotations,
        &scene.opacities,
        &scene.colors,
        &scene.sh_coeffs,
        &[
            camera.position[0],
            camera.position[1],
            camera.position[2],
            camera.target[0],
            camera.target[1],
            camera.target[2],
            camera.up[0],
            camera.up[1],
            camera.up[2],
            camera.fov_y_degrees,
            camera.near,
            camera.far,
            background[0],
            background[1],
            background[2],
            width as f32,
            height as f32,
            tile_size as f32,
            1.6,
            1.0 / 255.0,
            8.0,
            0.01,
            (32 * 32 * 16) as f32,
        ],
        &d_loss,
        width,
        height,
        tile_size,
        1.6,
        1.0 / 255.0,
        8,
        0.01,
        32 * 32 * 16,
        1.0,
        0,
        10.0,
    );
    clear_training_forward_cache();

    let rerun = backward_packed_host_slices(
        &scene.positions,
        &scene.scales,
        &scene.rotations,
        &scene.opacities,
        &scene.colors,
        &scene.sh_coeffs,
        &[
            camera.position[0],
            camera.position[1],
            camera.position[2],
            camera.target[0],
            camera.target[1],
            camera.target[2],
            camera.up[0],
            camera.up[1],
            camera.up[2],
            camera.fov_y_degrees,
            camera.near,
            camera.far,
            background[0],
            background[1],
            background[2],
            width as f32,
            height as f32,
            tile_size as f32,
            1.6,
            1.0 / 255.0,
            8.0,
            0.01,
            (32 * 32 * 16) as f32,
        ],
        &d_loss,
        width,
        height,
        tile_size,
        1.6,
        1.0 / 255.0,
        8,
        0.01,
        32 * 32 * 16,
        1.0,
        0,
        10.0,
    );

    assert_eq!(cached.len(), via_cache.len());
    assert_eq!(cached.len(), rerun.len());
    for (a, b) in cached.iter().zip(&via_cache) {
        assert_eq!(
            a, b,
            "cached host backward must match direct cached backward"
        );
    }
    for (a, b) in cached.iter().zip(&rerun) {
        assert!(
            (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0),
            "cached vs rerun forward: {a} vs {b}"
        );
    }
}
