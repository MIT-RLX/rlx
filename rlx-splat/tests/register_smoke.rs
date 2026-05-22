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
#[test]
fn register_and_render_reference_smoke() {
    rlx_splat::register();

    use rlx_splat::core::{make_parity_scene, Camera};
    use rlx_splat::reference::{render_reference, RenderParams};

    let scene = make_parity_scene();
    let camera = Camera::look_at([0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 60.0, 0.1, 20.0);
    let params = RenderParams {
        width: 64,
        height: 64,
        tile_size: 16,
        radius_scale: 1.6,
        alpha_cutoff: 1.0 / 255.0,
        max_splat_steps: 32,
        transmittance_threshold: 0.01,
        max_list_entries: 18 * 32,
    };
    let rgba = render_reference(&scene, &camera, [0.1, 0.15, 0.2], &params);
    assert_eq!(rgba.len(), 64 * 64 * 4);
    assert!(rgba.iter().all(|v| v.is_finite()));
}
