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
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};

use super::types::{ColmapCamera, ColmapImage, ColmapPoint3D};

pub const COLMAP_SIMPLE_PINHOLE_MODEL_ID: i32 = 0;
pub const COLMAP_PINHOLE_MODEL_ID: i32 = 1;
pub const COLMAP_SIMPLE_RADIAL_MODEL_ID: i32 = 2;
pub const COLMAP_RADIAL_MODEL_ID: i32 = 3;
pub const COLMAP_OPENCV_MODEL_ID: i32 = 4;
pub const COLMAP_FULL_OPENCV_MODEL_ID: i32 = 6;

pub fn camera_params_count(model_id: i32) -> Result<usize> {
    Ok(match model_id {
        COLMAP_SIMPLE_PINHOLE_MODEL_ID => 3,
        COLMAP_PINHOLE_MODEL_ID => 4,
        COLMAP_SIMPLE_RADIAL_MODEL_ID => 4,
        COLMAP_RADIAL_MODEL_ID => 5,
        COLMAP_OPENCV_MODEL_ID => 8,
        COLMAP_FULL_OPENCV_MODEL_ID => 12,
        other => bail!("unsupported COLMAP camera model id {other}"),
    })
}

pub fn camera_intrinsics(
    model_id: i32,
    params: &[f64],
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    match model_id {
        COLMAP_SIMPLE_PINHOLE_MODEL_ID => (
            params[0], params[0], params[1], params[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ),
        COLMAP_PINHOLE_MODEL_ID => (
            params[0], params[1], params[2], params[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ),
        COLMAP_SIMPLE_RADIAL_MODEL_ID => (
            params[0], params[0], params[1], params[2], params[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ),
        COLMAP_RADIAL_MODEL_ID => (
            params[0], params[0], params[1], params[2], params[3], params[4], 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ),
        COLMAP_OPENCV_MODEL_ID => (
            params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
            0.0, 0.0, 0.0, 0.0,
        ),
        COLMAP_FULL_OPENCV_MODEL_ID => (
            params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
            params[8], params[9], params[10], params[11],
        ),
        _ => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f64_array(r: &mut impl Read, count: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        out.push(f64::from_le_bytes(buf));
    }
    Ok(out)
}

fn read_string(r: &mut impl Read) -> Result<String> {
    let mut data = Vec::new();
    loop {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        if b[0] == 0 {
            break;
        }
        data.push(b[0]);
    }
    Ok(String::from_utf8(data)?)
}

pub fn load_cameras_bin(path: &Path) -> Result<BTreeMap<i32, ColmapCamera>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let count = read_u64(&mut reader)?;
    let mut cameras = BTreeMap::new();
    for _ in 0..count {
        let camera_id = read_i32(&mut reader)?;
        let model_id = read_i32(&mut reader)?;
        let width = read_u64(&mut reader)?;
        let height = read_u64(&mut reader)?;
        let params = read_f64_array(&mut reader, camera_params_count(model_id)?)?;
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

pub fn load_images_bin(path: &Path) -> Result<BTreeMap<i32, ColmapImage>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let count = read_u64(&mut reader)?;
    let mut images = BTreeMap::new();
    for _ in 0..count {
        let image_id = read_i32(&mut reader)?;
        let q = read_f64_array(&mut reader, 4)?;
        let t = read_f64_array(&mut reader, 3)?;
        let camera_id = read_i32(&mut reader)?;
        let name = read_string(&mut reader)?;
        let point_count = read_u64(&mut reader)? as usize;
        let mut points2d_xy = Vec::with_capacity(point_count);
        let mut points2d_point3d_ids = Vec::with_capacity(point_count);
        for _ in 0..point_count {
            let xy = read_f64_array(&mut reader, 2)?;
            let point_id = read_i64(&mut reader)?;
            points2d_xy.push([xy[0] as f32, xy[1] as f32]);
            points2d_point3d_ids.push(point_id);
        }
        images.insert(
            image_id,
            ColmapImage {
                image_id,
                q_wxyz: [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32],
                t_xyz: [t[0] as f32, t[1] as f32, t[2] as f32],
                camera_id,
                name,
                points2d_xy,
                points2d_point3d_ids,
            },
        );
    }
    Ok(images)
}

fn read_i64(r: &mut impl Read) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

pub fn load_points3d_bin(path: &Path) -> Result<BTreeMap<u64, ColmapPoint3D>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let count = read_u64(&mut reader)?;
    let mut points = BTreeMap::new();
    for _ in 0..count {
        let point_id = read_u64(&mut reader)?;
        let xyz = read_f64_array(&mut reader, 3)?;
        let mut rgb_buf = [0u8; 3];
        reader.read_exact(&mut rgb_buf)?;
        let error = read_f64_array(&mut reader, 1)?[0];
        let track_length = read_u64(&mut reader)?;
        let skip = track_length as usize * 8;
        let mut skip_buf = vec![0u8; skip];
        if skip > 0 {
            reader.read_exact(&mut skip_buf)?;
        }
        points.insert(
            point_id,
            ColmapPoint3D {
                point_id,
                xyz: [xyz[0] as f32, xyz[1] as f32, xyz[2] as f32],
                rgb: [
                    rgb_buf[0] as f32 / 255.0,
                    rgb_buf[1] as f32 / 255.0,
                    rgb_buf[2] as f32 / 255.0,
                ],
                error,
                track_length,
            },
        );
    }
    Ok(points)
}
