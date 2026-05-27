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
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use super::binary::{
    COLMAP_FULL_OPENCV_MODEL_ID, COLMAP_OPENCV_MODEL_ID, COLMAP_PINHOLE_MODEL_ID,
    COLMAP_RADIAL_MODEL_ID, COLMAP_SIMPLE_PINHOLE_MODEL_ID, COLMAP_SIMPLE_RADIAL_MODEL_ID,
    camera_intrinsics, camera_params_count,
};
use super::types::{ColmapCamera, ColmapImage, ColmapPoint3D};

fn model_id_from_name(name: &str) -> Result<i32> {
    Ok(match name {
        "SIMPLE_PINHOLE" => COLMAP_SIMPLE_PINHOLE_MODEL_ID,
        "PINHOLE" => COLMAP_PINHOLE_MODEL_ID,
        "SIMPLE_RADIAL" => COLMAP_SIMPLE_RADIAL_MODEL_ID,
        "RADIAL" => COLMAP_RADIAL_MODEL_ID,
        "OPENCV" => COLMAP_OPENCV_MODEL_ID,
        "FULL_OPENCV" => COLMAP_FULL_OPENCV_MODEL_ID,
        other => bail!("unsupported COLMAP camera model {other}"),
    })
}

fn iter_lines(path: &Path) -> Result<impl Iterator<Item = std::io::Result<String>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(BufReader::new(file).lines())
}

pub fn load_cameras_txt(path: &Path) -> Result<BTreeMap<i32, ColmapCamera>> {
    let mut cameras = BTreeMap::new();
    for line in iter_lines(path)? {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        ensure!(tokens.len() >= 5, "malformed cameras.txt line: {trimmed}");
        let camera_id: i32 = tokens[0].parse()?;
        let model_id = model_id_from_name(tokens[1])?;
        let width: u64 = tokens[2].parse()?;
        let height: u64 = tokens[3].parse()?;
        let params: Vec<f64> = tokens[4..]
            .iter()
            .map(|t| t.parse())
            .collect::<Result<_, _>>()?;
        ensure!(
            params.len() == camera_params_count(model_id)?,
            "wrong param count for camera {camera_id}"
        );
        let (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6) = camera_intrinsics(model_id, &params);
        cameras.insert(
            camera_id,
            ColmapCamera {
                camera_id,
                model_id,
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
                k3,
                k4,
                k5,
                k6,
            },
        );
    }
    Ok(cameras)
}

pub fn load_images_txt(path: &Path) -> Result<BTreeMap<i32, ColmapImage>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut images = BTreeMap::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        let header = header.trim().to_string();
        if header.is_empty() || header.starts_with('#') {
            continue;
        }
        let mut points_line = String::new();
        ensure!(
            reader.read_line(&mut points_line)? > 0,
            "unexpected EOF in images.txt"
        );
        let tokens: Vec<&str> = header.split_whitespace().collect();
        ensure!(tokens.len() >= 10, "malformed images.txt header");
        let image_id: i32 = tokens[0].parse()?;
        let q_wxyz = [
            tokens[1].parse()?,
            tokens[2].parse()?,
            tokens[3].parse()?,
            tokens[4].parse()?,
        ];
        let t_xyz = [tokens[5].parse()?, tokens[6].parse()?, tokens[7].parse()?];
        let camera_id: i32 = tokens[8].parse()?;
        let name = tokens[9..].join(" ");
        let point_tokens: Vec<f32> = points_line
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        let (points2d_xy, points2d_point3d_ids) = if point_tokens.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            ensure!(
                point_tokens.len().is_multiple_of(3),
                "malformed observation line"
            );
            let mut xy = Vec::new();
            let mut ids = Vec::new();
            for chunk in point_tokens.chunks(3) {
                xy.push([chunk[0], chunk[1]]);
                ids.push(chunk[2] as i64);
            }
            (xy, ids)
        };
        images.insert(
            image_id,
            ColmapImage {
                image_id,
                q_wxyz,
                t_xyz,
                camera_id,
                name,
                points2d_xy,
                points2d_point3d_ids,
            },
        );
    }
    Ok(images)
}

pub fn load_points3d_txt(path: &Path) -> Result<BTreeMap<u64, ColmapPoint3D>> {
    let mut points = BTreeMap::new();
    for line in iter_lines(path)? {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        ensure!(tokens.len() >= 8, "malformed points3D.txt line");
        let point_id: u64 = tokens[0].parse()?;
        let xyz = [tokens[1].parse()?, tokens[2].parse()?, tokens[3].parse()?];
        let rgb = [
            tokens[4].parse::<u8>()? as f32 / 255.0,
            tokens[5].parse::<u8>()? as f32 / 255.0,
            tokens[6].parse::<u8>()? as f32 / 255.0,
        ];
        let error: f64 = tokens[7].parse()?;
        let track_tokens = &tokens[8..];
        ensure!(track_tokens.len().is_multiple_of(2), "malformed track");
        points.insert(
            point_id,
            ColmapPoint3D {
                point_id,
                xyz,
                rgb,
                error,
                track_length: (track_tokens.len() / 2) as u64,
            },
        );
    }
    Ok(points)
}
