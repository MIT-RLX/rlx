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
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::ImageReader;

use super::downscale::{FrameDownscaleConfig, resolve_training_frame_image_size};
use super::init_hparams::{min_track_length_error, resolve_colmap_init_hparams};
use super::types::{
    ColmapCamera, ColmapFrame, ColmapImage, ColmapReconstruction, GaussianInitHyperParams,
    point_tables,
};
use crate::core::GaussianScene;
use crate::io_format::init::build_scene_from_positions_colors;

pub fn build_training_frames_from_root(
    recon: &ColmapReconstruction,
    images_root: impl AsRef<Path>,
    selected_camera_ids: &[i32],
) -> Result<Vec<ColmapFrame>> {
    build_training_frames_from_root_with_options(
        recon,
        images_root,
        selected_camera_ids,
        FrameDownscaleConfig::default(),
    )
}

pub fn build_training_frames_from_root_with_options(
    recon: &ColmapReconstruction,
    images_root: impl AsRef<Path>,
    selected_camera_ids: &[i32],
    downscale: FrameDownscaleConfig,
) -> Result<Vec<ColmapFrame>> {
    let images_root = images_root.as_ref().canonicalize().context("images root")?;
    if !images_root.is_dir() {
        bail!(
            "COLMAP image directory does not exist: {}",
            images_root.display()
        );
    }
    let selected: std::collections::HashSet<i32> = selected_camera_ids.iter().copied().collect();
    let mut frames = Vec::new();
    for (image_id, image) in &recon.images {
        if !selected.is_empty() && !selected.contains(&image.camera_id) {
            continue;
        }
        let Some(camera) = recon.cameras.get(&image.camera_id) else {
            continue;
        };
        let image_path = images_root.join(&image.name);
        if !image_path.is_file() {
            continue;
        }
        frames.push(build_training_frame(
            *image_id,
            image,
            camera,
            &image_path,
            downscale,
        )?);
    }
    if frames.is_empty() {
        bail!("no training frames were found in {}", images_root.display());
    }
    Ok(frames)
}

pub fn build_training_frames(
    recon: &ColmapReconstruction,
    images_subdir: &str,
) -> Result<Vec<ColmapFrame>> {
    build_training_frames_with_options(recon, images_subdir, FrameDownscaleConfig::default(), &[])
}

pub fn build_training_frames_with_options(
    recon: &ColmapReconstruction,
    images_subdir: &str,
    downscale: FrameDownscaleConfig,
    selected_camera_ids: &[i32],
) -> Result<Vec<ColmapFrame>> {
    let roots: Vec<PathBuf> = std::iter::once(recon.root.clone())
        .chain(std::iter::once(
            recon
                .sparse_dir
                .parent()
                .unwrap_or(&recon.root)
                .to_path_buf(),
        ))
        .collect();
    let subdirs: Vec<&str> = if images_subdir == "images_4" {
        vec!["images_4", "images", "."]
    } else {
        vec![images_subdir]
    };
    let mut first_error: Option<anyhow::Error> = None;
    for root in roots {
        for subdir in &subdirs {
            let images_root = root.join(subdir);
            match build_training_frames_from_root_with_options(
                recon,
                &images_root,
                selected_camera_ids,
                downscale,
            ) {
                Ok(frames) => return Ok(frames),
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
    }
    Err(first_error.unwrap_or_else(|| anyhow::anyhow!("no training frames found")))
}

fn build_training_frame(
    image_id: i32,
    image: &ColmapImage,
    camera: &ColmapCamera,
    image_path: &Path,
    downscale: FrameDownscaleConfig,
) -> Result<ColmapFrame> {
    let reader = ImageReader::open(image_path)?;
    let (src_width, src_height) = reader.into_dimensions()?;
    let (width, height) = resolve_training_frame_image_size(src_width, src_height, downscale)?;
    let sx = width as f64 / camera.width as f64;
    let sy = height as f64 / camera.height as f64;
    Ok(ColmapFrame {
        image_id,
        image_path: image_path.to_path_buf(),
        q_wxyz: image.q_wxyz,
        t_xyz: image.t_xyz,
        fx: (camera.fx * sx) as f32,
        fy: (camera.fy * sy) as f32,
        cx: (camera.cx * sx) as f32,
        cy: (camera.cy * sy) as f32,
        width,
        height,
        k1: camera.k1 as f32,
        k2: camera.k2 as f32,
        p1: camera.p1 as f32,
        p2: camera.p2 as f32,
        k3: camera.k3 as f32,
        k4: camera.k4 as f32,
        k5: camera.k5 as f32,
        k6: camera.k6 as f32,
        camera_id: image.camera_id,
    })
}

pub fn initialize_scene_from_colmap_points(
    recon: &ColmapReconstruction,
    max_gaussians: i32,
    seed: u64,
    init_hparams: Option<&GaussianInitHyperParams>,
    min_track_length: i32,
) -> Result<GaussianScene> {
    let (xyz, rgb) = point_tables(recon, min_track_length);
    if xyz.is_empty() {
        bail!("{}", min_track_length_error(min_track_length));
    }
    let resolved =
        resolve_colmap_init_hparams(recon, max_gaussians, init_hparams, min_track_length)?;
    let count = xyz.len() / 3;
    let chosen = if max_gaussians <= 0 {
        count
    } else {
        count.min(max_gaussians.max(1) as usize)
    };
    build_scene_from_positions_colors(
        xyz[..chosen * 3].to_vec(),
        rgb[..chosen * 3].to_vec(),
        seed,
        Some(&resolved),
    )
}

pub fn initialize_scene_from_points_colors(
    positions: Vec<f32>,
    colors: Vec<f32>,
    seed: u64,
    init_hparams: Option<&GaussianInitHyperParams>,
) -> Result<GaussianScene> {
    build_scene_from_positions_colors(positions, colors, seed, init_hparams)
}
