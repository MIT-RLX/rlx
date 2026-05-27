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
//! Scene I/O — PLY, COLMAP, image loaders.

pub mod colmap;
pub mod image;
pub mod init;
pub mod ply;

pub use colmap::{
    ColmapCamera, ColmapFrame, ColmapImage, ColmapPoint3D, ColmapReconstruction, ColmapTrainConfig,
    ColmapTrainingBundle, DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH, FrameDownscaleConfig,
    FrameDownscaleMode, GaussianInitHyperParams, build_training_frames,
    build_training_frames_from_root, build_training_frames_with_options, colmap_camera_centers,
    initialize_scene_from_colmap_points, initialize_scene_from_points_colors,
    load_colmap_reconstruction, load_colmap_training_bundle, parse_downscale_mode, point_tables,
    resolve_colmap_init_hparams, resolve_training_frame_image_size, suggest_colmap_init_hparams,
    suggest_points_init_hparams, transform_colmap_reconstruction_pca, transform_poses_pca,
};
pub use image::{load_rgba8_image, load_training_frame_rgba8, rgba8_to_f32};
pub use init::build_scene_from_positions_colors;
pub use ply::{SavePlyOptions, load_gaussian_ply, save_gaussian_ply};
