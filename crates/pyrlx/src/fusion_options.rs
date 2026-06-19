// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! `pyrlx.FusionOptions` - FKL / elementwise fusion toggles for `Session.compile_with`.

use pyo3::prelude::*;
use rlx_ir::logical_kernel::KernelDispatchPolicy;
use rlx_opt::FusionOptions;
use rlx_runtime::{CompileOptions, Precision};

use pyo3::exceptions::PyValueError;

pub(crate) fn parse_kernel_dispatch(s: &str) -> PyResult<KernelDispatchPolicy> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "native" | "force_native" => KernelDispatchPolicy::ForceNative,
        "common" | "force_common" => KernelDispatchPolicy::ForceCommon,
        "prefer_native" | "prefer" | "default" => KernelDispatchPolicy::PreferNative,
        other => Err(PyValueError::new_err(format!(
            "unknown kernel_dispatch '{other}' (native, common, prefer_native)"
        )))?,
    })
}

pub(crate) fn build_compile_options(
    precision: Precision,
    fusion_options: Option<PyRef<PyFusionOptions>>,
    kernel_dispatch: Option<&str>,
) -> PyResult<CompileOptions> {
    let mut opts = CompileOptions::new().precision(precision);
    if let Some(fo) = fusion_options {
        opts.fusion_opts = fo.to_rust();
    }
    if let Some(kd) = kernel_dispatch {
        opts.kernel_dispatch.policy = parse_kernel_dispatch(kd)?;
    }
    Ok(opts)
}

/// Per-compile fusion controls (mirrors `rlx_opt::FusionOptions`).
///
/// Env vars such as `RLX_NATIVE_FK_REGIONS=1`, `RLX_NO_NATIVE_FK_REGIONS=1`, and
/// `RLX_FK_BATCH_SINGLE_KERNEL=1` (CUDA/ROCm/Metal/wgpu batch one-launch; TPU uses per-slice HLO)
/// are merged at compile time. GPU-class targets enable native FKL regions by default unless opted out.
#[pyclass(name = "FusionOptions", module = "pyrlx._pyrlx")]
#[derive(Clone, Copy)]
pub(crate) struct PyFusionOptions {
    #[pyo3(get, set)]
    pub skip_fusion: bool,
    #[pyo3(get, set)]
    pub unfuse_elementwise_regions: bool,
    #[pyo3(get, set)]
    pub keep_elementwise_regions: bool,
    #[pyo3(get, set)]
    pub decompose_fusion_regions: bool,
    #[pyo3(get, set)]
    pub fk_fusion: bool,
    #[pyo3(get, set)]
    pub fuse_region_prologue: bool,
    #[pyo3(get, set)]
    pub fuse_batch_preprocess: bool,
    #[pyo3(get, set)]
    pub native_fk_regions: bool,
}

#[pymethods]
impl PyFusionOptions {
    #[new]
    #[pyo3(signature = (
        skip_fusion = false,
        unfuse_elementwise_regions = false,
        keep_elementwise_regions = false,
        decompose_fusion_regions = false,
        fk_fusion = true,
        fuse_region_prologue = true,
        fuse_batch_preprocess = true,
        native_fk_regions = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        skip_fusion: bool,
        unfuse_elementwise_regions: bool,
        keep_elementwise_regions: bool,
        decompose_fusion_regions: bool,
        fk_fusion: bool,
        fuse_region_prologue: bool,
        fuse_batch_preprocess: bool,
        native_fk_regions: bool,
    ) -> Self {
        Self {
            skip_fusion,
            unfuse_elementwise_regions,
            keep_elementwise_regions,
            decompose_fusion_regions,
            fk_fusion,
            fuse_region_prologue,
            fuse_batch_preprocess,
            native_fk_regions,
        }
    }

    /// Preset: keep `BatchElementwiseRegion` / `TransformRegion` in MIR.
    #[staticmethod]
    fn native_fk() -> Self {
        Self {
            skip_fusion: false,
            unfuse_elementwise_regions: false,
            keep_elementwise_regions: false,
            decompose_fusion_regions: false,
            fk_fusion: true,
            fuse_region_prologue: true,
            fuse_batch_preprocess: true,
            native_fk_regions: true,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyrlx.FusionOptions native_fk_regions={} fk_fusion={} fuse_region_prologue={} \
             fuse_batch_preprocess={} keep_elementwise_regions={}>",
            self.native_fk_regions,
            self.fk_fusion,
            self.fuse_region_prologue,
            self.fuse_batch_preprocess,
            self.keep_elementwise_regions
        )
    }
}

impl PyFusionOptions {
    pub(crate) fn to_rust(self) -> FusionOptions {
        FusionOptions {
            skip_fusion: self.skip_fusion,
            unfuse_elementwise_regions: self.unfuse_elementwise_regions,
            keep_elementwise_regions: self.keep_elementwise_regions,
            decompose_fusion_regions: self.decompose_fusion_regions,
            fk_fusion: self.fk_fusion,
            fuse_region_prologue: self.fuse_region_prologue,
            fuse_batch_preprocess: self.fuse_batch_preprocess,
            native_fk_regions: self.native_fk_regions,
            fusion_limits: rlx_opt::FusionLimits::default(),
        }
    }
}
