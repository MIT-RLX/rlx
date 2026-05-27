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
//! 3D Gaussian PLY load/save (3DGS / Inria format).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use crate::core::GaussianScene;
use crate::core::sh::{SUPPORTED_SH_COEFF_COUNT, pad_sh_coeffs, sh_coeffs_to_display_colors};
use anyhow::{Context, Result, bail, ensure};

const LOGIT_EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug)]
pub struct SavePlyOptions {
    pub include_sh: bool,
}

impl Default for SavePlyOptions {
    fn default() -> Self {
        Self { include_sh: true }
    }
}

pub fn load_gaussian_ply(path: impl AsRef<Path>) -> Result<GaussianScene> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening PLY {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let header = parse_ply_header(&mut reader)?;
    ensure!(
        header.elements.iter().any(|e| e.name == "vertex"),
        "PLY file does not contain a vertex element"
    );
    let vertex = header.elements.iter().find(|e| e.name == "vertex").unwrap();
    let n = vertex.count;
    let props: Vec<String> = vertex.properties.iter().map(|p| p.name.clone()).collect();
    let mut columns: HashMap<String, Vec<f32>> = HashMap::new();
    for prop in &props {
        columns.insert(prop.clone(), vec![0.0; n]);
    }

    if header.format == PlyFormat::Ascii {
        for row in 0..n {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let tokens: Vec<f32> = line
                .split_whitespace()
                .map(|t| t.parse())
                .collect::<std::result::Result<_, _>>()?;
            ensure!(tokens.len() >= props.len(), "short PLY row {row}");
            for (prop, value) in props.iter().zip(tokens.iter()) {
                columns.get_mut(prop).unwrap()[row] = *value;
            }
        }
    } else {
        let mut buf = vec![0u8; vertex.row_bytes];
        for row in 0..n {
            reader.read_exact(&mut buf)?;
            let mut offset = 0usize;
            for prop in &vertex.properties {
                let value = read_binary_property(&buf[offset..], prop)?;
                columns.get_mut(&prop.name).unwrap()[row] = value;
                offset += prop.byte_size();
            }
        }
    }

    let positions = stack_columns(&columns, &["x", "y", "z"], n)?;
    let opacities = sigmoid(column_or_zeros(&columns, "opacity", n));
    let scales = stack_sorted_prefix(&columns, "scale_", n, 3);
    let rotations = normalize_rotations(stack_sorted_prefix(&columns, "rot", n, 4));
    let f_dc = stack_columns(&columns, &["f_dc_0", "f_dc_1", "f_dc_2"], n)?;
    let rest_names = sorted_props(&props, "f_rest_");
    let coeff_count = if rest_names.is_empty() {
        1
    } else {
        ensure!(
            rest_names.len().is_multiple_of(3),
            "expected multiple-of-3 f_rest_* properties"
        );
        1 + rest_names.len() / 3
    };
    let mut sh_flat = vec![0.0f32; n * coeff_count * 3];
    for splat in 0..n {
        sh_flat[splat * coeff_count * 3] = f_dc[splat * 3];
        sh_flat[splat * coeff_count * 3 + 1] = f_dc[splat * 3 + 1];
        sh_flat[splat * coeff_count * 3 + 2] = f_dc[splat * 3 + 2];
    }
    if !rest_names.is_empty() {
        let rest_count = coeff_count - 1;
        for splat in 0..n {
            for idx in 0..rest_names.len() {
                let coeff = idx / 3 + 1;
                let ch = idx % 3;
                sh_flat[splat * coeff_count * 3 + coeff * 3 + ch] =
                    columns[&rest_names[idx]][splat];
            }
            let _ = rest_count;
        }
    }
    let sh_coeffs = pad_sh_coeffs(&sh_flat, n, SUPPORTED_SH_COEFF_COUNT);
    let colors = sh_coeffs_to_display_colors(&sh_coeffs, n, SUPPORTED_SH_COEFF_COUNT);
    Ok(GaussianScene::new(
        positions,
        scales,
        rotations,
        opacities,
        colors,
        sh_coeffs,
        SUPPORTED_SH_COEFF_COUNT,
    ))
}

pub fn save_gaussian_ply(
    path: impl AsRef<Path>,
    scene: &GaussianScene,
    options: SavePlyOptions,
) -> Result<PathBuf> {
    let output_path = path.as_ref().to_path_buf();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let count = scene.count();
    ensure!(
        scene.sh_coeff_count > 0,
        "GaussianScene.sh_coeffs must have coeff_count >= 1"
    );
    let export_coeff_count = if options.include_sh {
        SUPPORTED_SH_COEFF_COUNT
    } else {
        1
    };
    let sh_coeffs = pad_sh_coeffs(&scene.sh_coeffs, count, export_coeff_count);
    let rest_count = export_coeff_count.saturating_sub(1);
    let mut prop_names = vec![
        "x".to_string(),
        "y".to_string(),
        "z".to_string(),
        "opacity".to_string(),
        "f_dc_0".to_string(),
        "f_dc_1".to_string(),
        "f_dc_2".to_string(),
    ];
    for i in 0..rest_count * 3 {
        prop_names.push(format!("f_rest_{i}"));
    }
    prop_names.extend([
        "scale_0".to_string(),
        "scale_1".to_string(),
        "scale_2".to_string(),
        "rot_0".to_string(),
        "rot_1".to_string(),
        "rot_2".to_string(),
        "rot_3".to_string(),
    ]);

    let mut header = String::new();
    header.push_str("ply\nformat binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {count}\n"));
    for name in &prop_names {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("end_header\n");

    let mut file = File::create(&output_path)?;
    file.write_all(header.as_bytes())?;
    for splat in 0..count {
        let pos = scene.position(splat);
        file.write_all(&pos[0].to_le_bytes())?;
        file.write_all(&pos[1].to_le_bytes())?;
        file.write_all(&pos[2].to_le_bytes())?;
        file.write_all(&logit(scene.opacities[splat]).to_le_bytes())?;
        let sh_base = splat * export_coeff_count * 3;
        file.write_all(&sh_coeffs[sh_base].to_le_bytes())?;
        file.write_all(&sh_coeffs[sh_base + 1].to_le_bytes())?;
        file.write_all(&sh_coeffs[sh_base + 2].to_le_bytes())?;
        for coeff in 1..export_coeff_count {
            for ch in 0..3 {
                let v = sh_coeffs[sh_base + coeff * 3 + ch];
                file.write_all(&v.to_le_bytes())?;
            }
        }
        let scale = scene.scale(splat);
        file.write_all(&scale[0].to_le_bytes())?;
        file.write_all(&scale[1].to_le_bytes())?;
        file.write_all(&scale[2].to_le_bytes())?;
        let mut rot = scene.rotation(splat);
        let norm = (rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2] + rot[3] * rot[3])
            .sqrt()
            .max(1e-8);
        for q in &mut rot {
            *q /= norm;
        }
        for q in rot {
            file.write_all(&q.to_le_bytes())?;
        }
    }
    Ok(output_path)
}

fn sigmoid(values: Vec<f32>) -> Vec<f32> {
    values
        .into_iter()
        .map(|v| 1.0 / (1.0 + (-v).exp()))
        .collect()
}

fn logit(alpha: f32) -> f32 {
    let a = alpha.clamp(LOGIT_EPS, 1.0 - LOGIT_EPS);
    (a / (1.0 - a)).ln()
}

fn sorted_props(names: &[String], prefix: &str) -> Vec<String> {
    let mut filtered: Vec<String> = names
        .iter()
        .filter(|n| n.starts_with(prefix))
        .cloned()
        .collect();
    filtered.sort_by_key(|name| {
        name.split('_')
            .next_back()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    });
    filtered
}

fn stack_columns(
    columns: &HashMap<String, Vec<f32>>,
    names: &[&str],
    n: usize,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; n * names.len()];
    for (axis, name) in names.iter().enumerate() {
        let col = columns
            .get(*name)
            .with_context(|| format!("missing PLY property {name}"))?;
        for row in 0..n {
            out[row * names.len() + axis] = col[row];
        }
    }
    Ok(out)
}

fn column_or_zeros(columns: &HashMap<String, Vec<f32>>, name: &str, n: usize) -> Vec<f32> {
    columns.get(name).cloned().unwrap_or_else(|| vec![0.0; n])
}

fn stack_sorted_prefix(
    columns: &HashMap<String, Vec<f32>>,
    prefix: &str,
    n: usize,
    width: usize,
) -> Vec<f32> {
    let names = sorted_props(&columns.keys().cloned().collect::<Vec<_>>(), prefix);
    if names.is_empty() {
        let mut out = vec![0.0f32; n * width];
        if width >= 4 {
            for row in 0..n {
                out[row * width] = 1.0;
            }
        }
        return out;
    }
    let mut out = vec![0.0f32; n * width];
    for (axis, name) in names.iter().take(width).enumerate() {
        if let Some(col) = columns.get(name) {
            for row in 0..n {
                out[row * width + axis] = col[row];
            }
        }
    }
    out
}

fn normalize_rotations(rotations: Vec<f32>) -> Vec<f32> {
    let n = rotations.len() / 4;
    let mut out = rotations;
    for splat in 0..n {
        let base = splat * 4;
        let norm = (out[base] * out[base]
            + out[base + 1] * out[base + 1]
            + out[base + 2] * out[base + 2]
            + out[base + 3] * out[base + 3])
            .sqrt()
            .max(1e-8);
        for i in 0..4 {
            out[base + i] /= norm;
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittle,
    BinaryBig,
}

#[derive(Clone, Debug)]
struct PlyProperty {
    name: String,
    kind: PlyKind,
}

impl PlyProperty {
    fn byte_size(&self) -> usize {
        match self.kind {
            PlyKind::Float32 => 4,
            PlyKind::Float64 => 8,
            PlyKind::Int32 => 4,
            PlyKind::UInt8 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PlyKind {
    Float32,
    Float64,
    Int32,
    UInt8,
}

#[derive(Clone, Debug)]
struct PlyElement {
    name: String,
    count: usize,
    properties: Vec<PlyProperty>,
    row_bytes: usize,
}

#[derive(Clone, Debug)]
struct PlyHeader {
    format: PlyFormat,
    elements: Vec<PlyElement>,
}

fn parse_ply_header(reader: &mut impl BufRead) -> Result<PlyHeader> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    ensure!(line.trim() == "ply", "not a PLY file");
    let mut format = PlyFormat::Ascii;
    let mut elements = Vec::new();
    let mut current: Option<PlyElement> = None;
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed == "end_header" {
            break;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        match tokens[0] {
            "format" => {
                format = match tokens.get(1).copied() {
                    Some("ascii") => PlyFormat::Ascii,
                    Some("binary_little_endian") => PlyFormat::BinaryLittle,
                    Some("binary_big_endian") => PlyFormat::BinaryBig,
                    other => bail!("unsupported PLY format: {other:?}"),
                };
            }
            "element" => {
                if let Some(el) = current.take() {
                    elements.push(el);
                }
                current = Some(PlyElement {
                    name: tokens[1].to_string(),
                    count: tokens[2].parse()?,
                    properties: Vec::new(),
                    row_bytes: 0,
                });
            }
            "property" => {
                let el = current.as_mut().context("property outside element")?;
                let (kind, name) = if tokens[1] == "list" {
                    bail!("list properties are not supported");
                } else {
                    (parse_kind(tokens[1])?, tokens[2].to_string())
                };
                let prop = PlyProperty { name, kind };
                el.row_bytes += prop.byte_size();
                el.properties.push(prop);
            }
            _ => {}
        }
    }
    if let Some(el) = current {
        elements.push(el);
    }
    Ok(PlyHeader { format, elements })
}

fn parse_kind(token: &str) -> Result<PlyKind> {
    match token {
        "float" | "float32" => Ok(PlyKind::Float32),
        "double" | "float64" => Ok(PlyKind::Float64),
        "int" | "int32" => Ok(PlyKind::Int32),
        "uchar" | "uint8" => Ok(PlyKind::UInt8),
        other => bail!("unsupported PLY property type: {other}"),
    }
}

fn read_binary_property(bytes: &[u8], prop: &PlyProperty) -> Result<f32> {
    Ok(match prop.kind {
        PlyKind::Float32 => f32::from_le_bytes(bytes[..4].try_into()?),
        PlyKind::Float64 => bytes[..8].try_into().map(f64::from_le_bytes)? as f32,
        PlyKind::Int32 => i32::from_le_bytes(bytes[..4].try_into()?) as f32,
        PlyKind::UInt8 => bytes[0] as f32,
    })
}
