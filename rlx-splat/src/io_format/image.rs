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
use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, RgbaImage};

use super::colmap::ColmapFrame;

pub fn load_rgba8_image(path: impl AsRef<Path>, target_size: Option<(u32, u32)>) -> Result<Vec<u8>> {
    let path = path.as_ref();
    let img = image::open(path).with_context(|| format!("opening image {}", path.display()))?;
    let rgba = resize_rgba(img, target_size);
    Ok(rgba.into_raw())
}

pub fn load_training_frame_rgba8(frame: &ColmapFrame) -> Result<Vec<u8>> {
    load_rgba8_image(
        &frame.image_path,
        Some((frame.width.max(1), frame.height.max(1))),
    )
}

fn resize_rgba(img: DynamicImage, target_size: Option<(u32, u32)>) -> RgbaImage {
    let rgba = img.to_rgba8();
    let Some((tw, th)) = target_size else {
        return rgba;
    };
    if rgba.width() == tw && rgba.height() == th {
        return rgba;
    }
    image::imageops::resize(&rgba, tw, th, image::imageops::FilterType::Lanczos3)
}

/// Convert raw RGBA8 bytes to normalized f32 RGBA suitable for training targets.
pub fn rgba8_to_f32(rgba8: &[u8]) -> Vec<f32> {
    rgba8
        .chunks_exact(4)
        .flat_map(|px| {
            [
                px[0] as f32 / 255.0,
                px[1] as f32 / 255.0,
                px[2] as f32 / 255.0,
                px[3] as f32 / 255.0,
            ]
        })
        .collect()
}
