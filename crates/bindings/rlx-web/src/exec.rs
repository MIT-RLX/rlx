// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-backend compile/run helpers for vision graphs.

use rlx_ir::Graph;
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::vision::{self, BATCH, ModelMeta, VisionModelId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecBackend {
    Cpu,
    WebGpu,
    WebGl,
    Auto,
}

impl ExecBackend {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Self::Cpu,
            "webgpu" | "gpu" | "wgpu" => Self::WebGpu,
            "webgl" | "opengl" | "gl" => Self::WebGl,
            _ => Self::Auto,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::WebGpu => "webgpu",
            Self::WebGl => "webgl",
            Self::Auto => "auto",
        }
    }
}

#[allow(dead_code)] // used by browser feature paths / future API
pub fn resolve_device(backend: ExecBackend) -> Device {
    match backend {
        ExecBackend::Cpu => Device::Cpu,
        ExecBackend::WebGpu => Device::WebGpu,
        ExecBackend::WebGl => Device::OpenGl,
        ExecBackend::Auto => rlx_runtime::preferred_browser_device().unwrap_or(Device::Cpu),
    }
}

pub fn run_forward_cpu(
    id: VisionModelId,
    meta: &ModelMeta,
    x: &[f32],
    params: &[f32],
) -> Result<Vec<f32>, String> {
    let mut compiled = Session::new(Device::Cpu).compile(vision::build_forward(id, BATCH));
    vision::apply_params(&mut compiled, meta, params);
    compiled
        .run(&[("x", x)])
        .into_iter()
        .next()
        .ok_or_else(|| "forward produced no outputs".into())
}

pub fn run_train_cpu(
    id: VisionModelId,
    meta: &ModelMeta,
    x: &[f32],
    label: f32,
    params: &[f32],
) -> Result<(f32, Vec<Vec<f32>>), String> {
    let mut compiled = Session::new(Device::Cpu).compile(vision::build_train(id, BATCH));
    vision::apply_params(&mut compiled, meta, params);
    let outs = compiled.run(&[("x", x), ("labels", &[label]), ("d_output", &[1.0])]);
    let loss = outs
        .first()
        .and_then(|o| o.first().copied())
        .ok_or_else(|| "train produced no loss".to_string())?;
    Ok((loss, outs))
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub async fn run_forward_webgpu(
    id: VisionModelId,
    meta: &ModelMeta,
    x: &[f32],
    params: &[f32],
) -> Result<Vec<f32>, String> {
    let mut compiled = BrowserSession::new(Device::WebGpu)
        .map_err(|e| e.to_string())?
        .compile(vision::build_forward(id, BATCH))
        .map_err(|e| e.to_string())?;
    vision::apply_browser_params(&mut compiled, meta, params);
    compiled
        .run_async(&[("x", x)])
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "forward produced no outputs".into())
}

#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub async fn run_train_webgpu(
    id: VisionModelId,
    meta: &ModelMeta,
    x: &[f32],
    label: f32,
    params: &[f32],
) -> Result<(f32, Vec<Vec<f32>>), String> {
    let mut compiled = BrowserSession::new(Device::WebGpu)
        .map_err(|e| e.to_string())?
        .compile(vision::build_train(id, BATCH))
        .map_err(|e| e.to_string())?;
    vision::apply_browser_params(&mut compiled, meta, params);
    let outs = compiled
        .run_async(&[("x", x), ("labels", &[label]), ("d_output", &[1.0])])
        .await
        .map_err(|e| e.to_string())?;
    let loss = outs
        .first()
        .and_then(|o| o.first().copied())
        .ok_or_else(|| "train produced no loss".to_string())?;
    Ok((loss, outs))
}

#[cfg(all(feature = "webgl", target_arch = "wasm32"))]
pub fn run_forward_webgl(
    id: VisionModelId,
    meta: &ModelMeta,
    x: &[f32],
    params: &[f32],
) -> Result<Vec<f32>, String> {
    let mut compiled = BrowserSession::new(Device::OpenGl)
        .map_err(|e| e.to_string())?
        .compile(vision::build_forward(id, BATCH))
        .map_err(|e| e.to_string())?;
    vision::apply_browser_params(&mut compiled, meta, params);
    compiled
        .run(&[("x", x)])
        .into_iter()
        .next()
        .ok_or_else(|| "forward produced no outputs".into())
}

#[cfg(all(feature = "webgl", target_arch = "wasm32"))]
pub fn run_train_webgl(
    id: VisionModelId,
    meta: &ModelMeta,
    x: &[f32],
    label: f32,
    params: &[f32],
) -> Result<(f32, Vec<Vec<f32>>), String> {
    if !id.webgl_train() {
        return Err(format!(
            "model {} does not support WebGL training (conv backward unavailable)",
            id.slug()
        ));
    }
    let mut compiled = BrowserSession::new(Device::OpenGl)
        .map_err(|e| e.to_string())?
        .compile(vision::build_train(id, BATCH))
        .map_err(|e| e.to_string())?;
    vision::apply_browser_params(&mut compiled, meta, params);
    let outs = compiled.run(&[("x", x), ("labels", &[label]), ("d_output", &[1.0])]);
    let loss = outs
        .first()
        .and_then(|o| o.first().copied())
        .ok_or_else(|| "train produced no loss".to_string())?;
    Ok((loss, outs))
}

/// Compile-only check that a graph is runnable on a browser backend.
#[allow(dead_code)] // wasm / browser selection helpers
pub fn browser_supports_train(id: VisionModelId, device: Device) -> bool {
    if device == Device::OpenGl && !id.webgl_train() {
        return false;
    }
    let g = vision::build_train(id, BATCH);
    rlx_runtime::browser_support::select_browser_device_for_graph(&g) == Some(device)
}

#[allow(dead_code)]
pub fn compile_cpu(graph: Graph) -> CompiledGraph {
    Session::new(Device::Cpu).compile(graph)
}
