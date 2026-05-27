// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Three-step model compile pipeline (template → specialize → backend).
//!
//! Host code loads symbolic HIR once, specializes per [`ModelVariant`] /
//! [`DimBinding`], then lowers to a device executable. Pair with
//! [`BindingManifest`] for parameter-block style binding.

use std::collections::VecDeque;

use rlx_ir::hir::HirModule;
use rlx_ir::{BindingManifest, DimBinding, ModelComponent};
use rlx_opt::CompileResult;

use crate::stages;
use crate::{CompileOptions, CompiledGraph, Device};

/// Compile-once / specialize-per-variant pipeline with optional FIFO cache.
pub struct ModelCompilePipeline {
    device: Device,
    capacity: usize,
    template: Option<CompileResult>,
    entries: Vec<(u64, CompiledGraph)>,
    order: VecDeque<u64>,
}

impl ModelCompilePipeline {
    pub fn new(device: Device) -> Self {
        Self::with_capacity(device, 8)
    }

    pub fn with_capacity(device: Device, capacity: usize) -> Self {
        assert!(capacity > 0, "ModelCompilePipeline capacity must be ≥ 1");
        Self {
            device,
            capacity,
            template: None,
            entries: Vec::new(),
            order: VecDeque::new(),
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn has_template(&self) -> bool {
        self.template.is_some()
    }

    /// **Step 1** — run fusion pipeline on symbolic HIR (dynamic dims allowed).
    pub fn build_template<F>(
        &mut self,
        build_hir: F,
        options: &CompileOptions,
    ) -> Result<&CompileResult, rlx_ir::hir::LowerError>
    where
        F: FnOnce() -> HirModule,
    {
        if self.template.is_none() {
            let pipe = stages::pipeline_for(self.device, options);
            self.template = Some(pipe.compile_hir(build_hir())?);
        }
        Ok(self.template.as_ref().expect("template set"))
    }

    pub fn template_binding_manifest(&self) -> BindingManifest {
        let template = self.template.as_ref().expect("call build_template first");
        BindingManifest::from_lir(&template.lir)
    }

    /// **Step 2** — bind symbolic dims and replan buffers.
    pub fn specialize_template(
        &self,
        binding: &DimBinding,
        options: &CompileOptions,
    ) -> CompileResult {
        let template = self
            .template
            .as_ref()
            .expect("call build_template before specialize_template");
        let pipe = stages::pipeline_for(self.device, options);
        template.specialize(&pipe, binding)
    }

    /// **Step 3** — backend executable from specialized LIR.
    pub fn compile_lir(
        &self,
        specialized: CompileResult,
        options: &CompileOptions,
    ) -> CompiledGraph {
        let backend = crate::registry::backend_for(self.device).expect("backend registered");
        let executable = backend.compile_lir(specialized.lir, options);
        CompiledGraph::new(executable, self.device)
    }

    /// Full pipeline: template (once) → specialize → compile; cached by `key`.
    pub fn get_or_compile<F>(
        &mut self,
        key: u64,
        binding: &DimBinding,
        build_hir: F,
        options: &CompileOptions,
    ) -> Result<&mut CompiledGraph, rlx_ir::hir::LowerError>
    where
        F: FnOnce() -> HirModule,
    {
        if let Some(idx) = self.entries.iter().position(|(k, _)| *k == key) {
            return Ok(&mut self.entries[idx].1);
        }
        let mut template_opts = options.clone();
        template_opts.dim_binding = None;
        self.build_template(build_hir, &template_opts)?;
        let specialized = self.specialize_template(binding, &template_opts);
        let mut compile_opts = options.clone();
        compile_opts.dim_binding = None;
        let compiled = self.compile_lir(specialized, &compile_opts);

        if self.entries.len() >= self.capacity
            && let Some(evict) = self.order.pop_front()
        {
            self.entries.retain(|(k, _)| *k != evict);
        }
        self.entries.push((key, compiled));
        self.order.push_back(key);
        Ok(&mut self.entries.last_mut().unwrap().1)
    }

    /// Manifest for a variant without storing specialized LIR in the cache.
    pub fn binding_manifest_for_binding(
        &self,
        binding: &DimBinding,
        options: &CompileOptions,
    ) -> BindingManifest {
        let specialized = self.specialize_template(binding, options);
        BindingManifest::from_lir(&specialized.lir)
    }

    /// Layout for a full [`ModelComponent`] (specialized parameter block).
    pub fn binding_manifest_for_component(
        &self,
        component: &ModelComponent,
        options: &CompileOptions,
    ) -> BindingManifest {
        self.binding_manifest_for_binding(&component.dim_binding(), options)
    }

    /// Template → specialize → compile; keyed by [`ModelComponent::cache_key`].
    pub fn get_or_compile_component<F>(
        &mut self,
        component: &ModelComponent,
        build_hir: F,
        options: &CompileOptions,
    ) -> Result<(&mut CompiledGraph, BindingManifest), rlx_ir::hir::LowerError>
    where
        F: FnOnce() -> HirModule,
    {
        let key = component.cache_key();
        let binding = component.dim_binding();
        let manifest = self.binding_manifest_for_component(component, options);
        let compiled = self.get_or_compile(key, &binding, build_hir, options)?;
        Ok((compiled, manifest))
    }

    pub fn contains(&self, key: u64) -> bool {
        self.entries.iter().any(|(k, _)| *k == key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Symbolic template from [`Self::build_template`] / [`Self::get_or_compile`].
    pub fn template_result(&self) -> Option<&CompileResult> {
        self.template.as_ref()
    }

    /// Build the symbolic template once (no specialization).
    pub fn ensure_template<F: FnOnce() -> HirModule>(
        &mut self,
        build_hir: F,
        options: &CompileOptions,
    ) -> Result<&CompileResult, rlx_ir::hir::LowerError> {
        self.build_template(build_hir, options)
    }

    /// Disk-backed specialize ([`CompilationMode::Aot`]); caches by `key`.
    pub fn get_or_specialize_aot<F: FnOnce() -> HirModule>(
        &mut self,
        aot: &crate::AotCache,
        disk_base: &str,
        key: u64,
        binding: &DimBinding,
        build_hir: F,
        options: &CompileOptions,
    ) -> Result<&mut CompiledGraph, crate::AotCacheError> {
        if let Some(idx) = self.entries.iter().position(|(k, _)| *k == key) {
            return Ok(&mut self.entries[idx].1);
        }
        let device = self.device;
        let template = self.ensure_template(build_hir, options)?;
        let compiled = aot.specialize_cached(disk_base, binding, device, template, options)?;
        if self.entries.len() >= self.capacity
            && let Some(evict_key) = self.order.pop_front()
        {
            self.entries.retain(|(k, _)| *k != evict_key);
        }
        self.entries.push((key, compiled));
        self.order.push_back(key);
        Ok(&mut self.entries.last_mut().unwrap().1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::hir::HirMut;
    use rlx_ir::{DType, HirModule, Shape};

    #[test]
    fn template_specialize_compile_smoke() {
        let device = Device::Cpu;
        let mut pipe = ModelCompilePipeline::new(device);
        let opts = CompileOptions::new();

        let build = || {
            let mut hir = HirModule::new("dyn");
            let mut gb = HirMut::new(&mut hir);
            let x = gb.input("x", Shape::new(&[1, 8, 4], DType::F32));
            let w = gb.param("w", Shape::new(&[4, 2], DType::F32));
            let y = hir.linear(x, w, None, None, Shape::new(&[1, 8, 2], DType::F32));
            hir.set_outputs(vec![y]);
            hir
        };

        pipe.build_template(build, &opts).unwrap();
        let binding = rlx_ir::DimBinding::new();
        let spec = pipe.specialize_template(&binding, &opts);
        let manifest = BindingManifest::from_lir(&spec.lir);
        assert_eq!(manifest.params[0].name, "w");
        let mut compiled = pipe.compile_lir(spec, &opts);
        compiled.set_param("w", &[0.0f32; 8]);
        let outs = compiled.run(&[("x", &[0.0f32; 32])]);
        assert_eq!(outs.len(), 1);
    }
}
