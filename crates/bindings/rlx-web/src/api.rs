// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Developer-facing wasm API for vision benchmarks and browser runtime.

use wasm_bindgen::prelude::*;

use crate::exec;
use crate::vision::{self, ModelMeta, VisionModelId};

fn grad_slices(outs: &[Vec<f32>]) -> Vec<&[f32]> {
    outs.iter().skip(1).map(|v| v.as_slice()).collect()
}

/// List vision model slugs available in the browser bundle.
#[wasm_bindgen]
pub fn list_vision_models() -> Vec<String> {
    VisionModelId::all()
        .iter()
        .map(|id| id.slug().to_string())
        .collect()
}

/// Metadata for a vision model (shapes, param layout, WebGL train support).
#[wasm_bindgen]
pub struct VisionModelInfo {
    slug: String,
    title: String,
    input_dims: Vec<usize>,
    num_classes: usize,
    param_names: Vec<String>,
    param_sizes: Vec<usize>,
    webgl_train: bool,
}

#[wasm_bindgen]
impl VisionModelInfo {
    #[wasm_bindgen(getter)]
    pub fn slug(&self) -> String {
        self.slug.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn title(&self) -> String {
        self.title.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn input_dims(&self) -> Vec<usize> {
        self.input_dims.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    #[wasm_bindgen(getter)]
    pub fn param_names(&self) -> Vec<String> {
        self.param_names.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn param_sizes(&self) -> Vec<usize> {
        self.param_sizes.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn webgl_train(&self) -> bool {
        self.webgl_train
    }

    #[wasm_bindgen(getter)]
    pub fn input_len(&self) -> usize {
        self.input_dims.iter().product()
    }

    #[wasm_bindgen(getter)]
    pub fn param_flat_len(&self) -> usize {
        self.param_sizes.iter().sum()
    }
}

fn info_from_meta(meta: &ModelMeta) -> VisionModelInfo {
    VisionModelInfo {
        slug: meta.id.slug().to_string(),
        title: meta.id.title().to_string(),
        input_dims: meta.input_shape.to_vec(),
        num_classes: meta.num_classes,
        param_names: meta.param_names().into_iter().map(str::to_string).collect(),
        param_sizes: meta.param_sizes(),
        webgl_train: meta.id.webgl_train(),
    }
}

/// Look up model metadata by slug (`mnist-cnn`, `mnist-mlp`, `cifar-cnn`, `resnet`).
#[wasm_bindgen]
pub fn vision_model_info(slug: &str) -> Result<VisionModelInfo, JsValue> {
    let id = VisionModelId::from_slug(slug).ok_or_else(|| {
        JsValue::from_str(&format!(
            "unknown vision model {slug:?}; use list_vision_models()"
        ))
    })?;
    Ok(info_from_meta(&vision::model_meta(id)))
}

/// Vision benchmark handle — forward, backward, and SGD on CPU / WebGPU / WebGL.
#[wasm_bindgen]
pub struct VisionBench {
    id: VisionModelId,
    meta: ModelMeta,
}

#[wasm_bindgen]
impl VisionBench {
    #[wasm_bindgen(constructor)]
    pub fn new(slug: &str) -> Result<VisionBench, JsValue> {
        let id = VisionModelId::from_slug(slug)
            .ok_or_else(|| JsValue::from_str(&format!("unknown vision model {slug:?}")))?;
        let meta = vision::model_meta(id);
        Ok(Self { id, meta })
    }

    pub fn info(&self) -> VisionModelInfo {
        info_from_meta(&self.meta)
    }

    /// Deterministic Kaiming-style init; biases are zero.
    pub fn init_params(&self, seed: u32) -> Vec<f32> {
        vision::init_params(&self.meta, seed)
    }

    /// Synthetic normalized input + integer class label.
    pub fn synthetic_batch(&self, seed: u32) -> Vec<f32> {
        let (x, label) = vision::synthetic_input(&self.meta, seed);
        let mut out = x;
        out.push(label);
        out
    }

    /// Forward pass on CPU — returns logits (`num_classes` floats).
    pub fn forward_cpu(&self, x: &[f32], params: &[f32]) -> Result<Vec<f32>, JsValue> {
        exec::run_forward_cpu(self.id, &self.meta, x, params).map_err(|e| JsValue::from_str(&e))
    }

    /// One training step on CPU. Returns `[loss, updated_params_flat…]`.
    pub fn train_step_cpu(
        &self,
        x: &[f32],
        label: f32,
        params: &[f32],
        lr: f32,
    ) -> Result<Vec<f32>, JsValue> {
        let (loss, outs) = exec::run_train_cpu(self.id, &self.meta, x, label, params)
            .map_err(|e| JsValue::from_str(&e))?;
        let grads = grad_slices(&outs);
        let updated = vision::train_step_params(&self.meta, params, &grads, lr);
        let mut flat = vec![loss];
        flat.extend(updated);
        Ok(flat)
    }

    /// Run `steps` SGD iterations on CPU with synthetic data. Returns
    /// `[initial_loss, final_loss, steps_per_sec]`.
    pub fn bench_cpu(&self, steps: u32, seed: u32, lr: f32) -> Result<Vec<f32>, JsValue> {
        let mut params = vision::init_params(&self.meta, seed);
        let (x, label) = vision::synthetic_input(&self.meta, seed);
        let l0 = exec::run_train_cpu(self.id, &self.meta, &x, label, &params)
            .map_err(|e| JsValue::from_str(&e))?
            .0;

        let t0 = bench_now();
        for step in 0..steps {
            let (loss, outs) = exec::run_train_cpu(self.id, &self.meta, &x, label, &params)
                .map_err(|e| JsValue::from_str(&e))?;
            let grads = grad_slices(&outs);
            params = vision::train_step_params(&self.meta, &params, &grads, lr);
            let _ = (step, loss);
        }
        let elapsed = bench_now() - t0;
        let l1 = exec::run_train_cpu(self.id, &self.meta, &x, label, &params)
            .map_err(|e| JsValue::from_str(&e))?
            .0;
        let sps = if elapsed > 0.0 {
            steps as f32 / elapsed
        } else {
            0.0
        };
        Ok(vec![l0, l1, sps])
    }

    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    pub async fn forward_webgpu(&self, x: Vec<f32>, params: Vec<f32>) -> Result<Vec<f32>, JsValue> {
        exec::run_forward_webgpu(self.id, &self.meta, &x, &params)
            .await
            .map_err(|e| JsValue::from_str(&e))
    }

    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    pub async fn train_step_webgpu(
        &self,
        x: Vec<f32>,
        label: f32,
        params: Vec<f32>,
        lr: f32,
    ) -> Result<Vec<f32>, JsValue> {
        let (loss, outs) = exec::run_train_webgpu(self.id, &self.meta, &x, label, &params)
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        let grads = grad_slices(&outs);
        let updated = vision::train_step_params(&self.meta, &params, &grads, lr);
        let mut flat = vec![loss];
        flat.extend(updated);
        Ok(flat)
    }

    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    pub fn forward_webgl(&self, x: &[f32], params: &[f32]) -> Result<Vec<f32>, JsValue> {
        exec::run_forward_webgl(self.id, &self.meta, x, params).map_err(|e| JsValue::from_str(&e))
    }

    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    pub fn train_step_webgl(
        &self,
        x: &[f32],
        label: f32,
        params: &[f32],
        lr: f32,
    ) -> Result<Vec<f32>, JsValue> {
        let (loss, outs) = exec::run_train_webgl(self.id, &self.meta, x, label, params)
            .map_err(|e| JsValue::from_str(&e))?;
        let grads = grad_slices(&outs);
        let updated = vision::train_step_params(&self.meta, params, &grads, lr);
        let mut flat = vec![loss];
        flat.extend(updated);
        Ok(flat)
    }
}

/// Parse backend name: `auto`, `cpu`, `webgpu`, `webgl`.
#[wasm_bindgen]
pub fn parse_backend(name: &str) -> String {
    exec::ExecBackend::parse(name).label().to_string()
}

/// Highest-priority available browser backend label (`cpu`, `webgpu`, `webgl`, or `none`).
#[wasm_bindgen]
pub fn preferred_backend() -> String {
    rlx_runtime::preferred_browser_device()
        .map(|d| match d {
            rlx_runtime::Device::WebGpu => "webgpu".to_string(),
            rlx_runtime::Device::OpenGl => "webgl".to_string(),
            rlx_runtime::Device::Cpu => "cpu".to_string(),
            other => format!("{other:?}"),
        })
        .unwrap_or_else(|| "none".to_string())
}

fn bench_now() -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now() as f32 / 1000.0)
            .unwrap_or(0.0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::Instant;
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let start = START.get_or_init(Instant::now);
        start.elapsed().as_secs_f32()
    }
}

#[cfg(test)]
mod api_tests {
    use crate::{VisionBench, list_vision_models};

    #[test]
    fn lists_four_models() {
        assert_eq!(list_vision_models().len(), 4);
    }

    #[test]
    fn bench_cpu_runs() {
        let bench = VisionBench::new("mnist-mlp").unwrap();
        let out = bench.bench_cpu(5, 7, 0.01).unwrap();
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn all_slugs_bench_decreases_loss() {
        for slug in list_vision_models() {
            let bench = VisionBench::new(&slug).unwrap();
            let out = bench.bench_cpu(15, 11, 0.05).unwrap();
            let (l0, l1, _sps) = (out[0], out[1], out[2]);
            assert!(
                l0.is_finite() && l1.is_finite(),
                "{slug}: non-finite loss {l0} -> {l1}"
            );
            assert!(
                l1 <= l0 + 1e-3,
                "{slug}: expected loss not to rise: {l0} -> {l1}"
            );
        }
    }
}
