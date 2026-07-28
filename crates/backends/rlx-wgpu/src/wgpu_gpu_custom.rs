// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw-GPU custom-kernel registry for `Op::Custom` on wgpu.
//!
//! Companion to the host-delegate path (`custom_host.rs`, which stages operands
//! off-GPU and runs an `rlx-cpu` reference kernel). A `WgpuGpuKernel` instead
//! dispatches a **real WGSL compute kernel directly against the arena buffer**,
//! with no D2H/H2D roundtrip — the wgpu analogue of `rlx_metal`'s
//! `MetalGpuKernel`. A registered GPU kernel takes precedence over a host one,
//! and (being pure-GPU) it also runs on browser WebGPU, unlike `Step::CustomHost`.
//!
//! ## Binding convention (fixed)
//!
//! The executor binds a single storage window covering the op's operands, plus a
//! storage params buffer. A downstream kernel's WGSL must declare exactly:
//!
//! ```wgsl
//! @group(0) @binding(0) var<storage, read_write> arena: array<f32>;
//! @group(0) @binding(1) var<storage, read>       params: array<u32>;
//! // params = [ out_off, out_len, n_inputs, _pad,
//! //            in0_off, in0_len, in1_off, in1_len, ... ]   (f32-element offsets
//! //            into the bound `arena` window)
//! ```
//!
//! Index the output at `arena[params[0] + i]` and input `j` at
//! `arena[params[4 + 2*j] + i]`. Offsets are element (f32) offsets relative to
//! the bound window (the executor rebases them for you).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::kernels::Kernel;

/// A raw-GPU wgpu custom kernel: a WGSL compute shader dispatched straight
/// against the arena buffer, no host roundtrip. Register under the same `name`
/// used in `Op::Custom` / `OpExtension::name`. See the module docs for the fixed
/// binding convention the WGSL must follow.
pub trait WgpuGpuKernel: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;

    /// WGSL source. Must bind `arena` (storage rw) @0 and `params` (storage
    /// read) @1 exactly as documented on this module.
    fn wgsl(&self) -> &str;

    /// Compute-shader entry point (default `"main"`).
    fn entry_point(&self) -> &str {
        "main"
    }

    /// Workgroup grid given the output element count. Default: 1-D, 64/group.
    fn workgroups(&self, out_elems: u32) -> (u32, u32, u32) {
        (out_elems.div_ceil(64).max(1), 1, 1)
    }
}

struct Registry {
    kernels: RwLock<HashMap<String, Arc<dyn WgpuGpuKernel>>>,
    pipelines: RwLock<HashMap<String, Arc<Kernel>>>,
}

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Registry {
        kernels: RwLock::new(HashMap::new()),
        pipelines: RwLock::new(HashMap::new()),
    })
}

/// Register a raw-GPU wgpu custom kernel (takes precedence over a host-delegate
/// kernel of the same name).
pub fn register_wgpu_gpu_kernel(k: Arc<dyn WgpuGpuKernel>) {
    let name = k.name().to_string();
    let mut g = registry().kernels.write().unwrap();
    if g.contains_key(&name) {
        eprintln!("rlx-wgpu: WgpuGpuKernel '{name}' was already registered — replacing");
    }
    g.insert(name, k);
}

/// Whether a raw-GPU wgpu kernel is registered for `name`.
pub fn has_gpu_kernel(name: &str) -> bool {
    registry().kernels.read().unwrap().contains_key(name)
}

/// Look up a registered kernel by name.
pub fn lookup(name: &str) -> Option<Arc<dyn WgpuGpuKernel>> {
    registry().kernels.read().unwrap().get(name).cloned()
}

/// Get (compiling + caching on first use) the compute pipeline for `name`.
pub fn get_or_build_pipeline(device: &wgpu::Device, k: &dyn WgpuGpuKernel) -> Arc<Kernel> {
    let name = k.name();
    if let Some(p) = registry().pipelines.read().unwrap().get(name) {
        return Arc::clone(p);
    }
    let built = Arc::new(build_custom_kernel(device, k.wgsl(), k.entry_point()));
    registry()
        .pipelines
        .write()
        .unwrap()
        .insert(name.to_string(), Arc::clone(&built));
    built
}

/// Build a [`Kernel`] with the fixed `{storage rw @0, storage read @1}` layout.
/// Mirrors `kernels::build_kernel` but with a storage (not uniform) params
/// binding so params carry a tight `array<u32>` with no 16-byte stride padding.
fn build_custom_kernel(device: &wgpu::Device, wgsl: &str, entry_point: &str) -> Kernel {
    let label = "rlx-wgpu custom gpu";
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

/// Create a storage buffer holding `data` (params), written at creation.
pub fn make_params_buffer(device: &wgpu::Device, data: &[u32]) -> wgpu::Buffer {
    // `data` is `[u32]`, so its byte length is always a multiple of 4 and ≥ 16
    // (the fixed header) — size the buffer exactly so the mapped view length
    // matches for `copy_from_slice`.
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu custom gpu params"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        // `BufferViewMut` derefs to `[u8]` for method calls (not the index op).
        let mut view = buf
            .slice(..)
            .get_mapped_range_mut()
            .expect("params buffer mapped at creation");
        view.copy_from_slice(bytes);
    }
    buf.unmap();
    buf
}
