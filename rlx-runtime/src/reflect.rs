// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Model reflection services (Slang compiler/runtime API §5).
//!
//! Introspect unspecialized templates and specialized layouts across eager/lazy/AOT
//! while preserving the HIR → MIR → LIR pipeline.

use rlx_ir::hir::HirModule;
use rlx_ir::{
    apply_hir_extensions, BindingManifest, HirReflection, ManifestDiff, MirReflection,
    ModelComponent, layout_from_lir,
};
use rlx_opt::CompileResult;

use crate::model_pipeline::ModelCompilePipeline;
use crate::options::CompileOptions;
use crate::stages;
use crate::Device;

/// Loaded template + HIR reflection (front-end load).
pub struct ModelReflection {
    pub hir: HirReflection,
    template: Option<CompileResult>,
}

impl ModelReflection {
    /// Build HIR reflection only (no compile).
    pub fn from_hir(hir: &HirModule) -> Self {
        Self {
            hir: HirReflection::from_hir(hir),
            template: None,
        }
    }

    /// Compile symbolic template on `device` and retain for specialize/layout.
    pub fn load_hir_template(
        device: Device,
        hir: HirModule,
        options: &CompileOptions,
    ) -> Result<Self, rlx_ir::hir::LowerError> {
        let mut opts = options.clone();
        opts.dim_binding = None;
        let hir_ref = HirReflection::from_hir(&hir);
        let pipe = stages::pipeline_for(device, &opts);
        let template = pipe.compile_hir(hir)?;
        Ok(Self {
            hir: hir_ref,
            template: Some(template),
        })
    }

    pub fn has_template(&self) -> bool {
        self.template.is_some()
    }

    pub fn mir_summary(&self) -> Option<MirReflection> {
        self.template
            .as_ref()
            .map(|t| MirReflection::from_mir(&t.lir.mir))
    }

    /// Template layout (symbolic dims may be unresolved in arena sizes).
    pub fn template_layout(&self) -> Option<BindingManifest> {
        self.template.as_ref().map(|t| layout_from_lir(&t.lir))
    }

    /// Specialized layout for a [`ModelComponent`] (getTypeLayout after specialize).
    pub fn layout_for_component(
        &self,
        component: &ModelComponent,
        device: Device,
        options: &CompileOptions,
    ) -> Option<BindingManifest> {
        let template = self.template.as_ref()?;
        let mut opts = options.clone();
        opts.dim_binding = None;
        let pipe = stages::pipeline_for(device, &opts);
        let specialized = template.specialize(&pipe, &component.dim_binding());
        Some(layout_from_lir(&specialized.lir))
    }

    pub fn manifest_diff_for_component(
        &self,
        component: &ModelComponent,
        device: Device,
        options: &CompileOptions,
    ) -> Option<ManifestDiff> {
        let t = self.template_layout()?;
        let s = self.layout_for_component(component, device, options)?;
        Some(ManifestDiff::compare(&t, &s))
    }
}

/// Full specialize + compile entry (specializeEntryPoint analogue).
pub fn specialize_entry<'a>(
    pipeline: &'a mut ModelCompilePipeline,
    component: &ModelComponent,
    build_hir: impl FnOnce() -> HirModule,
    options: &CompileOptions,
) -> Result<&'a mut crate::CompiledGraph, rlx_ir::hir::LowerError> {
    let key = component.cache_key();
    let binding = component.dim_binding();
    pipeline.get_or_compile(key, &binding, build_hir, options)
}

/// Apply HIR extensions then load template.
pub fn load_hir_template_with_extensions(
    device: Device,
    mut hir: HirModule,
    options: &CompileOptions,
) -> Result<ModelReflection, rlx_ir::hir::LowerError> {
    apply_hir_extensions(&mut hir);
    ModelReflection::load_hir_template(device, hir, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::hir::HirMut;
    use rlx_ir::{DType, HirModule, ModelVariant, Shape};

    #[test]
    fn reflection_loads_template_on_cpu() {
        let device = Device::Cpu;
        let hir = || {
            let mut hir = HirModule::new("refl");
            let mut gb = HirMut::new(&mut hir);
            let x = gb.input("x", Shape::new(&[1, 4], DType::F32));
            let w = gb.param("w", Shape::new(&[4, 2], DType::F32));
            let y = hir.linear(x, w, None, None, Shape::new(&[1, 2], DType::F32));
            hir.set_outputs(vec![y]);
            hir
        };
        let refl = ModelReflection::load_hir_template(device, hir(), &CompileOptions::new())
            .unwrap();
        assert!(refl.has_template());
        let layout = refl.template_layout().unwrap();
        assert_eq!(layout.params[0].name, "w");
        let comp = ModelComponent::new(ModelVariant::prefill(1, 4));
        let spec_layout = refl
            .layout_for_component(&comp, device, &CompileOptions::new())
            .unwrap();
        assert_eq!(spec_layout.params[0].name, "w");
    }
}
