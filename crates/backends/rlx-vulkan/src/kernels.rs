// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Per-kernel compute-pipeline cache.
//!
//! Every kernel shares one descriptor-set layout: binding 0 = activations,
//! binding 1 = weights (a tiny dummy buffer when unsplit). A 128-byte push-
//! constant range carries per-op offsets (weight slots tagged with bit 31).

use crate::device::{VulkanDevice, vulkan_device};
use crate::shaders;
use ash::vk;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Push-constant block size in bytes (Vulkan guarantees ≥ 128).
pub const PUSH_CONSTANT_BYTES: u32 = 128;

pub struct Kernels {
    dev: &'static VulkanDevice,
    pub dsl: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    cache: Mutex<HashMap<&'static str, vk::Pipeline>>,
    modules: Mutex<Vec<vk::ShaderModule>>,
}

unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

static KERNELS: OnceLock<Option<Kernels>> = OnceLock::new();

/// The process-wide kernel cache, or `None` if no device.
pub fn kernels() -> Option<&'static Kernels> {
    KERNELS
        .get_or_init(|| vulkan_device().map(Kernels::new))
        .as_ref()
}

impl Kernels {
    fn new(dev: &'static VulkanDevice) -> Self {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl = unsafe {
            dev.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .expect("vk descriptor_set_layout");

        let set_layouts = [dsl];
        let pc_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES)];
        let pipeline_layout = unsafe {
            dev.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&pc_ranges),
                None,
            )
        }
        .expect("vk pipeline_layout");

        Self {
            dev,
            dsl,
            pipeline_layout,
            cache: Mutex::new(HashMap::new()),
            modules: Mutex::new(Vec::new()),
        }
    }

    /// Get (compiling on first use) the compute pipeline for kernel `name`.
    pub fn pipeline(&self, name: &'static str) -> vk::Pipeline {
        if let Some(p) = self.cache.lock().unwrap().get(name) {
            return *p;
        }
        let blob = shaders::blob(name)
            .unwrap_or_else(|| panic!("rlx-vulkan: no embedded SPIR-V for kernel '{name}'"));
        let words = shaders::words(blob);
        let module = unsafe {
            self.dev
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .unwrap_or_else(|e| panic!("vk shader_module '{name}': {e}"));

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let create = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout);
        let pipeline = unsafe {
            self.dev
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create], None)
        }
        .map_err(|(_, e)| e)
        .unwrap_or_else(|e| panic!("vk compute_pipeline '{name}': {e}"))[0];

        self.modules.lock().unwrap().push(module);
        self.cache.lock().unwrap().insert(name, pipeline);
        pipeline
    }
}
