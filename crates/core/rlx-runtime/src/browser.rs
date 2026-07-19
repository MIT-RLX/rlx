// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! Browser runtime — unified WebGPU (async) + WebGL fallback + CPU.
//!
//! Call [`init_webgpu`] once at startup on wasm, then use [`BrowserSession`] to
//! compile and run graphs through the same backend registry as native code.

use crate::browser_support;
use crate::compiled::CompiledGraph;
use crate::device_ext::{self, BROWSER_DEVICE_PRIORITY};
use crate::session::Session;
use rlx_driver::Device;
use rlx_ir::Graph;

/// Error from browser compile/run preflight.
#[derive(Debug, Clone)]
pub struct BrowserError(pub String);

impl std::fmt::Display for BrowserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for BrowserError {}

impl From<String> for BrowserError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Initialize the WebGPU device on wasm. Idempotent; returns `true` when an
/// adapter is available. No-op on native (uses blocking device discovery).
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub async fn init_webgpu() -> bool {
    rlx_wgpu::device::init_wgpu_device().await
}

#[cfg(not(all(feature = "gpu", target_arch = "wasm32")))]
pub async fn init_webgpu() -> bool {
    #[cfg(feature = "gpu")]
    {
        rlx_wgpu::is_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Re-export browser device priority for bindings.
pub use device_ext::BROWSER_DEVICE_PRIORITY as browser_device_priority;

/// Available browser backends in priority order.
pub fn available_browser_devices() -> Vec<Device> {
    device_ext::available_browser_devices()
}

/// Highest-priority live browser backend.
pub fn preferred_browser_device() -> Option<Device> {
    device_ext::preferred_browser_device()
}

/// Is `graph` runnable on any available browser backend?
pub fn supports_browser_graph(graph: &Graph) -> bool {
    browser_support::select_browser_device_for_graph(graph).is_some()
}

/// Diagnostic when no browser backend can run `graph`.
pub fn browser_block_reason(graph: &Graph) -> Option<String> {
    if let Some(reason) = browser_support::collective_block_reason(graph) {
        return Some(reason);
    }
    for &device in BROWSER_DEVICE_PRIORITY {
        if !device_ext::is_available(device) {
            continue;
        }
        if let Some((_, op)) = browser_support::first_browser_unsupported_op(graph, device) {
            return Some(format!(
                "op {op:?} not supported on browser backend {device:?}"
            ));
        }
        if device_ext::supports_graph(device, graph) {
            return None;
        }
    }
    Some("no browser backend available — enable `webgpu` and/or `opengl` features".into())
}

/// Session targeting a browser backend (`WebGpu` preferred, then `OpenGl`, then `Cpu`).
pub struct BrowserSession {
    device: Device,
}

impl BrowserSession {
    /// Use the highest-priority available browser backend.
    pub fn new_preferred() -> Result<Self, BrowserError> {
        let device = preferred_browser_device().ok_or_else(|| {
            BrowserError(
                "no browser backend available — call init_webgpu() and/or enable opengl feature"
                    .into(),
            )
        })?;
        Ok(Self { device })
    }

    /// Target a specific browser backend.
    pub fn new(device: Device) -> Result<Self, BrowserError> {
        if !BROWSER_DEVICE_PRIORITY.contains(&device) {
            return Err(BrowserError(format!(
                "{device:?} is not a browser backend (use WebGpu, OpenGl, or Cpu)"
            )));
        }
        if !device_ext::is_available(device) {
            return Err(BrowserError(format!(
                "browser backend {device:?} is not available"
            )));
        }
        Ok(Self { device })
    }

    /// Compile for a specific graph, auto-selecting WebGPU vs WebGL fallback.
    pub fn compile_for_graph(graph: &Graph) -> Result<BrowserCompiledGraph, BrowserError> {
        if let Some(reason) = browser_support::collective_block_reason(graph) {
            return Err(BrowserError(reason));
        }
        let device = browser_support::select_browser_device_for_graph(graph).ok_or_else(|| {
            browser_block_reason(graph)
                .unwrap_or_else(|| "graph not supported on any browser backend".into())
        })?;
        BrowserSession { device }.compile(graph.clone())
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Compile `graph` on the selected browser backend.
    pub fn compile(&self, graph: Graph) -> Result<BrowserCompiledGraph, BrowserError> {
        if let Some(reason) = browser_support::collective_block_reason(&graph) {
            return Err(BrowserError(reason));
        }
        if !device_ext::supports_graph(self.device, &graph) {
            return Err(BrowserError(browser_block_reason(&graph).unwrap_or_else(
                || format!("graph not supported on {:?}", self.device),
            )));
        }
        let compiled = Session::new(self.device).compile(graph);
        Ok(BrowserCompiledGraph {
            inner: compiled,
            device: self.device,
        })
    }
}

/// A graph compiled for a browser backend.
pub struct BrowserCompiledGraph {
    inner: CompiledGraph,
    device: Device,
}

impl BrowserCompiledGraph {
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.inner.set_param(name, data);
    }

    /// Synchronous run (WebGL, CPU, or native WebGPU).
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.inner.run(inputs)
    }

    /// Async run on wasm WebGPU; sync fallback for WebGL/CPU.
    #[cfg(target_arch = "wasm32")]
    pub async fn run_async(
        &mut self,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>, BrowserError> {
        match self.device {
            Device::WebGpu | Device::Gpu => {
                self.inner.run_async(inputs).await.map_err(BrowserError)
            }
            _ => Ok(self.inner.run(inputs)),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn run_async(
        &mut self,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>, BrowserError> {
        Ok(self.inner.run(inputs))
    }
}
