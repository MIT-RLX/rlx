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
pub mod binary;
pub mod dataset;
pub mod downscale;
pub mod init_hparams;
pub mod ops;
pub mod pca;
pub mod text;
pub mod types;

pub use dataset::{load_colmap_training_bundle, ColmapTrainConfig, ColmapTrainingBundle};
pub use downscale::{
    parse_downscale_mode, resolve_training_frame_image_size, FrameDownscaleConfig,
    FrameDownscaleMode,
};
pub use init_hparams::{
    resolve_colmap_init_hparams, suggest_colmap_init_hparams, suggest_points_init_hparams,
};
pub use ops::*;
pub use pca::{transform_colmap_reconstruction_pca, transform_poses_pca};
pub use types::*;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub const COLMAP_DEFAULT_SPARSE_SUBDIR: &str = "sparse/0";

pub fn load_colmap_reconstruction(
    root: impl AsRef<Path>,
    sparse_subdir: &str,
) -> Result<ColmapReconstruction> {
    let root_path = root.as_ref().canonicalize().context("resolve COLMAP root")?;
    let sparse_subdir_text = sparse_subdir.trim();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if !sparse_subdir_text.is_empty() {
        candidates.push(root_path.join(sparse_subdir_text));
    } else {
        candidates.push(root_path.clone());
    }
    if sparse_subdir_text.replace('\\', "/").trim_matches('/') == COLMAP_DEFAULT_SPARSE_SUBDIR {
        candidates.push(root_path.join("sparse"));
        candidates.push(root_path.clone());
        if root_path.is_dir() {
            for entry in std::fs::read_dir(&root_path)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    candidates.push(entry.path().join("sparse"));
                }
            }
        }
    }
    let (cameras_path, images_path, points_path, format_kind) =
        resolve_colmap_sparse_paths(&candidates)?;
    let sparse_dir = cameras_path.parent().unwrap().to_path_buf();
    Ok(ColmapReconstruction {
        root: root_path,
        sparse_dir,
        cameras: if format_kind == "bin" {
            binary::load_cameras_bin(&cameras_path)?
        } else {
            text::load_cameras_txt(&cameras_path)?
        },
        images: if format_kind == "bin" {
            binary::load_images_bin(&images_path)?
        } else {
            text::load_images_txt(&images_path)?
        },
        points3d: if format_kind == "bin" {
            binary::load_points3d_bin(&points_path)?
        } else {
            text::load_points3d_txt(&points_path)?
        },
    })
}

fn resolve_colmap_sparse_paths(
    sparse_dirs: &[PathBuf],
) -> Result<(PathBuf, PathBuf, PathBuf, &'static str)> {
    for sparse_dir in sparse_dirs {
        for format_kind in ["bin", "txt"] {
            let cameras = sparse_dir.join(format!("cameras.{format_kind}"));
            let images = sparse_dir.join(format!("images.{format_kind}"));
            let points = sparse_dir.join(format!("points3D.{format_kind}"));
            if cameras.is_file() && images.is_file() && points.is_file() {
                return Ok((cameras, images, points, format_kind));
            }
        }
    }
    bail!(
        "missing required COLMAP sparse files under: {}",
        sparse_dirs
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}
