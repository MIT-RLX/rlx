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

use anyhow::{Context, Result};

/// Max absolute difference allowed between ORT and RLX outputs.
pub const DEFAULT_ATOL: f32 = 1e-4;

pub struct ConformanceResult {
    pub op: String,
    pub max_abs_diff: f32,
    pub passed: bool,
}

pub fn compare_tensors(a: &[f32], b: &[f32], atol: f32) -> (f32, bool) {
    if a.len() != b.len() {
        return (f32::INFINITY, false);
    }
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    (max_diff, max_diff <= atol)
}

#[cfg(not(any(target_os = "ios", target_os = "android")))]
pub struct OrtSession {
    session: ort::session::Session,
}

#[cfg(not(any(target_os = "ios", target_os = "android")))]
impl OrtSession {
    pub fn from_bytes(model: &[u8]) -> Result<Self> {
        let session = ort::session::Session::builder()
            .context("ort session builder")?
            .commit_from_memory(model)
            .context("load ort model")?;
        Ok(Self { session })
    }

    /// Run a single f32 input and return one f32 output tensor (flattened).
    pub fn run_one_f32_input(
        &mut self,
        input_name: &str,
        input: &[f32],
        input_shape: &[i64],
        output_index: usize,
    ) -> Result<Vec<f32>> {
        use ort::session::{SessionInputValue, SessionInputs};
        use ort::value::Tensor;
        let owned: Vec<f32> = input.to_vec();
        let shape_usize: Vec<usize> = input_shape.iter().map(|&d| d as usize).collect();
        let tensor = Tensor::from_array((shape_usize.as_slice(), owned))
            .context("ort input tensor")?
            .into_dyn();
        let feeds = vec![(input_name.to_string(), SessionInputValue::Owned(tensor))];
        let outputs = self
            .session
            .run(SessionInputs::from(feeds))
            .context("ort run")?;
        let (out_name, out_val) = outputs
            .iter()
            .nth(output_index)
            .with_context(|| format!("ort output index {output_index}"))?;
        let (_shape, data) = out_val
            .try_extract_tensor::<f32>()
            .with_context(|| format!("extract ort output {out_name}"))?;
        Ok(data.to_vec())
    }

    /// Run a zero-input model and return one f32 output tensor (flattened).
    pub fn run_no_inputs(&mut self, output_index: usize) -> Result<Vec<f32>> {
        use ort::session::{SessionInputValue, SessionInputs};
        let empty: Vec<(String, SessionInputValue)> = Vec::new();
        let outputs = self
            .session
            .run(SessionInputs::from(empty))
            .context("ort run")?;
        let (out_name, out_val) = outputs
            .iter()
            .nth(output_index)
            .with_context(|| format!("ort output index {output_index}"))?;
        let (_shape, data) = out_val
            .try_extract_tensor::<f32>()
            .with_context(|| format!("extract ort output {out_name}"))?;
        Ok(data.to_vec())
    }
}

#[cfg(any(target_os = "ios", target_os = "android"))]
pub struct OrtSession;

#[cfg(any(target_os = "ios", target_os = "android"))]
impl OrtSession {
    pub fn from_bytes(_model: &[u8]) -> Result<Self> {
        anyhow::bail!("ORT conformance harness unavailable on this target")
    }
}
