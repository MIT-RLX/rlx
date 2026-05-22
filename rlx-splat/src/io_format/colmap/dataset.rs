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
//! High-level COLMAP → scene + training frames loader for RLX training.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::core::GaussianScene;

use super::downscale::FrameDownscaleConfig;
use super::types::{ColmapFrame, ColmapReconstruction, GaussianInitHyperParams};
use super::{
    build_training_frames_with_options, initialize_scene_from_colmap_points,
    load_colmap_reconstruction, transform_colmap_reconstruction_pca,
    DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH,
};

/// Configuration for loading a COLMAP dataset into RLX training types.
#[derive(Clone, Debug)]
pub struct ColmapTrainConfig {
    pub root: PathBuf,
    pub sparse_subdir: String,
    pub images_subdir: String,
    pub max_gaussians: i32,
    pub seed: u64,
    pub min_track_length: i32,
    pub init_hparams: Option<GaussianInitHyperParams>,
    /// Apply PCA pose/point alignment (same as Python viewer default path when enabled).
    pub pca_align: bool,
    pub pca_rescale: bool,
    pub max_frames: usize,
    pub downscale: FrameDownscaleConfig,
    pub selected_camera_ids: Vec<i32>,
}

impl ColmapTrainConfig {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            sparse_subdir: super::COLMAP_DEFAULT_SPARSE_SUBDIR.to_string(),
            images_subdir: "images".to_string(),
            max_gaussians: 0,
            seed: 42,
            min_track_length: DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH,
            init_hparams: None,
            pca_align: false,
            pca_rescale: false,
            max_frames: 0,
            downscale: FrameDownscaleConfig::default(),
            selected_camera_ids: Vec::new(),
        }
    }
}

/// Loaded COLMAP reconstruction, initialized scene, and training frames.
#[derive(Clone, Debug)]
pub struct ColmapTrainingBundle {
    pub reconstruction: ColmapReconstruction,
    pub scene: GaussianScene,
    pub frames: Vec<ColmapFrame>,
}

/// Load sparse reconstruction, optional PCA, scene init, and training frames.
pub fn load_colmap_training_bundle(config: &ColmapTrainConfig) -> Result<ColmapTrainingBundle> {
    let mut recon =
        load_colmap_reconstruction(&config.root, &config.sparse_subdir).context("load COLMAP")?;
    if config.pca_align {
        recon = transform_colmap_reconstruction_pca(&recon, config.pca_rescale).0;
    }
    let scene = initialize_scene_from_colmap_points(
        &recon,
        config.max_gaussians,
        config.seed,
        config.init_hparams.as_ref(),
        config.min_track_length,
    )?;
    let mut frames = super::build_training_frames_with_options(
        &recon,
        &config.images_subdir,
        config.downscale,
        &config.selected_camera_ids,
    )?;
    if config.max_frames > 0 && frames.len() > config.max_frames {
        frames.truncate(config.max_frames);
    }
    Ok(ColmapTrainingBundle {
        reconstruction: recon,
        scene,
        frames,
    })
}
