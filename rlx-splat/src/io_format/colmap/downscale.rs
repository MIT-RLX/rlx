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
//! Training-frame image downscale (parity with slang-splat `resolve_training_frame_image_size`).

use anyhow::{Result, bail};

/// How to resize COLMAP training images before building [`super::ColmapFrame`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrameDownscaleMode {
    #[default]
    Original,
    MaxSize,
    Scale,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameDownscaleConfig {
    pub mode: FrameDownscaleMode,
    pub max_size: Option<u32>,
    pub scale: f32,
}

impl Default for FrameDownscaleConfig {
    fn default() -> Self {
        Self {
            mode: FrameDownscaleMode::Original,
            max_size: None,
            scale: 1.0,
        }
    }
}

impl FrameDownscaleConfig {
    pub fn max_size(max: u32) -> Self {
        Self {
            mode: FrameDownscaleMode::MaxSize,
            max_size: Some(max.max(1)),
            scale: 1.0,
        }
    }

    pub fn scale(factor: f32) -> Self {
        Self {
            mode: FrameDownscaleMode::Scale,
            max_size: None,
            scale: factor.clamp(1e-6, 1.0),
        }
    }
}

pub fn resolve_training_frame_image_size(
    src_width: u32,
    src_height: u32,
    config: FrameDownscaleConfig,
) -> Result<(u32, u32)> {
    let src_width = src_width.max(1);
    let src_height = src_height.max(1);
    match config.mode {
        FrameDownscaleMode::Original => Ok((src_width, src_height)),
        FrameDownscaleMode::MaxSize => {
            let target_max = config
                .max_size
                .ok_or_else(|| anyhow::anyhow!("max_size required for MaxSize downscale"))?
                .max(1);
            let src_max = src_width.max(src_height);
            if target_max >= src_max {
                return Ok((src_width, src_height));
            }
            let scale = target_max as f32 / src_max as f32;
            let w = (src_width as f32 * scale).round() as u32;
            let h = (src_height as f32 * scale).round() as u32;
            Ok((w.max(1).min(src_width), h.max(1).min(src_height)))
        }
        FrameDownscaleMode::Scale => {
            let factor = config.scale.clamp(1e-6, 1.0);
            if factor >= 1.0 {
                return Ok((src_width, src_height));
            }
            let w = (src_width as f32 * factor).round() as u32;
            let h = (src_height as f32 * factor).round() as u32;
            Ok((w.max(1).min(src_width), h.max(1).min(src_height)))
        }
    }
}

pub fn parse_downscale_mode(mode: &str) -> Result<FrameDownscaleMode> {
    match mode.trim().to_lowercase().as_str() {
        "original" => Ok(FrameDownscaleMode::Original),
        "max_size" | "maxsize" => Ok(FrameDownscaleMode::MaxSize),
        "scale" => Ok(FrameDownscaleMode::Scale),
        other => bail!("unsupported image downscale mode: {other}"),
    }
}
