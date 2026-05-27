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
//! COLMAP I/O tests (rlx-splat `io` feature).

use std::io::Write;

use rlx_splat::io_format::{
    initialize_scene_from_colmap_points, load_colmap_reconstruction, resolve_colmap_init_hparams,
    suggest_colmap_init_hparams,
};
use tempfile::tempdir;

fn write_cameras_bin(path: &std::path::Path) {
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap();
    file.write_all(&7i32.to_le_bytes()).unwrap();
    file.write_all(&1i32.to_le_bytes()).unwrap();
    file.write_all(&400u64.to_le_bytes()).unwrap();
    file.write_all(&200u64.to_le_bytes()).unwrap();
    for v in [400.0f64, 420.0, 200.0, 100.0] {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn write_images_bin(path: &std::path::Path, image_name: &str) {
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap();
    file.write_all(&3i32.to_le_bytes()).unwrap();
    for v in [1.0f64, 0.0, 0.0, 0.0] {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
    for v in [0.0f64, 0.0, -2.0] {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
    file.write_all(&7i32.to_le_bytes()).unwrap();
    file.write_all(image_name.as_bytes()).unwrap();
    file.write_all(&[0u8]).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
}

fn write_points3d_bin(path: &std::path::Path) {
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    for (point_id, xyz, rgb) in [
        (11u64, (1.0_f64, 2.0, 3.0), (255u8, 128, 64)),
        (12, (-1.0_f64, 0.0, 2.0), (12, 34, 56)),
    ] {
        file.write_all(&point_id.to_le_bytes()).unwrap();
        for v in [xyz.0, xyz.1, xyz.2] {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file.write_all(&[rgb.0, rgb.1, rgb.2]).unwrap();
        file.write_all(&0.5f64.to_le_bytes()).unwrap();
        file.write_all(&3u64.to_le_bytes()).unwrap();
        for _ in 0..3 {
            file.write_all(&1i32.to_le_bytes()).unwrap();
            file.write_all(&0i32.to_le_bytes()).unwrap();
        }
    }
}

fn build_tiny_colmap_tree(root: &std::path::Path) {
    let sparse = root.join("sparse/0");
    let images = root.join("images_4");
    std::fs::create_dir_all(&sparse).unwrap();
    std::fs::create_dir_all(&images).unwrap();
    write_cameras_bin(&sparse.join("cameras.bin"));
    write_images_bin(&sparse.join("images.bin"), "view.png");
    write_points3d_bin(&sparse.join("points3D.bin"));
    let img = image::RgbaImage::from_pixel(64, 48, image::Rgba([128, 64, 32, 255]));
    img.save_with_format(images.join("view.png"), image::ImageFormat::Png)
        .unwrap();
}

#[test]
fn load_colmap_reconstruction_reads_binary_sparse() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("scene");
    build_tiny_colmap_tree(&root);
    let recon = load_colmap_reconstruction(&root, "sparse/0").unwrap();
    assert_eq!(recon.cameras.len(), 1);
    assert_eq!(recon.images.len(), 1);
    assert_eq!(recon.points3d.len(), 2);
}

#[test]
fn suggest_and_initialize_colmap_scene() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("scene");
    build_tiny_colmap_tree(&root);
    let recon = load_colmap_reconstruction(&root, "sparse/0").unwrap();
    let h = suggest_colmap_init_hparams(&recon, 0, 3).unwrap();
    assert!(h.base_scale.is_some());
    let resolved = resolve_colmap_init_hparams(&recon, 0, None, 3).unwrap();
    let scene = initialize_scene_from_colmap_points(&recon, 0, 42, Some(&resolved), 3).unwrap();
    assert_eq!(scene.count(), 2);
}
