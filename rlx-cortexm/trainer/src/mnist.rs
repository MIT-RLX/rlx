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

//! MNIST IDX-format loader.
//!
//! IDX layout (network byte order, big-endian):
//! ```text
//! images: magic(0x00000803) | n | rows | cols | u8[n][rows*cols]
//! labels: magic(0x00000801) | n | u8[n]
//! ```
//!
//! Pixels are converted to `f32` in `[-1, 1]` by `(x / 255.0) * 2.0 - 1.0`,
//! matching the `transforms.Normalize((0.5,), (0.5,))` step in the
//! Python script the trainer replaces. Images stay flat
//! (`28 * 28 = 784` floats per row), batch reshaping happens at use time.

use std::fs;
use std::path::Path;

pub const ROWS: usize = 28;
pub const COLS: usize = 28;
pub const PIXELS: usize = ROWS * COLS;

pub struct Dataset {
    pub train: Split,
    pub test: Split,
}

pub struct Split {
    /// Flattened images: `images.len() == n * PIXELS`. Each `[i*PIXELS .. (i+1)*PIXELS]`
    /// chunk is one image in row-major order, normalized to `[-1, 1]`.
    pub images: Vec<f32>,
    /// Class label per image (0..=9), encoded as f32 to match the IR's
    /// labels-as-f32 convention used by `Op::SoftmaxCrossEntropyWithLogits`.
    pub labels: Vec<f32>,
}

impl Split {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Image `i` as a flat slice of 784 f32s in `[-1, 1]`.
    pub fn image(&self, i: usize) -> &[f32] {
        &self.images[i * PIXELS..(i + 1) * PIXELS]
    }
}

pub fn load(dir: &Path) -> Result<Dataset, String> {
    let train = load_split(
        &dir.join("train-images-idx3-ubyte"),
        &dir.join("train-labels-idx1-ubyte"),
    )?;
    let test = load_split(
        &dir.join("t10k-images-idx3-ubyte"),
        &dir.join("t10k-labels-idx1-ubyte"),
    )?;
    Ok(Dataset { train, test })
}

fn load_split(images_path: &Path, labels_path: &Path) -> Result<Split, String> {
    let images = read_images(images_path)?;
    let labels = read_labels(labels_path)?;
    let n_imgs = images.len() / PIXELS;
    if n_imgs != labels.len() {
        return Err(format!(
            "{}: {n_imgs} images vs {} labels",
            images_path.display(),
            labels.len()
        ));
    }
    Ok(Split { images, labels })
}

fn read_images(path: &Path) -> Result<Vec<f32>, String> {
    let raw = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if raw.len() < 16 {
        return Err(format!("{}: header too short", path.display()));
    }
    let magic = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    if magic != 0x00000803 {
        return Err(format!(
            "{}: bad magic 0x{magic:08x} (expected 0x00000803)",
            path.display()
        ));
    }
    let n = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    let rows = u32::from_be_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    let cols = u32::from_be_bytes([raw[12], raw[13], raw[14], raw[15]]) as usize;
    if rows != ROWS || cols != COLS {
        return Err(format!(
            "{}: expected {ROWS}x{COLS}, got {rows}x{cols}",
            path.display()
        ));
    }
    let body = &raw[16..];
    let want = n * PIXELS;
    if body.len() < want {
        return Err(format!(
            "{}: truncated (need {want} pixels, have {})",
            path.display(),
            body.len()
        ));
    }
    // u8 → f32 in [-1, 1].
    let mut out = Vec::with_capacity(want);
    for &b in &body[..want] {
        out.push((b as f32 / 255.0) * 2.0 - 1.0);
    }
    Ok(out)
}

fn read_labels(path: &Path) -> Result<Vec<f32>, String> {
    let raw = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if raw.len() < 8 {
        return Err(format!("{}: header too short", path.display()));
    }
    let magic = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    if magic != 0x00000801 {
        return Err(format!(
            "{}: bad magic 0x{magic:08x} (expected 0x00000801)",
            path.display()
        ));
    }
    let n = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    if raw.len() < 8 + n {
        return Err(format!("{}: truncated label body", path.display()));
    }
    Ok(raw[8..8 + n].iter().map(|&b| b as f32).collect())
}
