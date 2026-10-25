// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `pyrlx.FusionOptions` - FKL / elementwise fusion toggles for `Session.compile_with`.

use pyo3::prelude::*;
use rlx_bind::runtime::CompileRequest;
use rlx_opt::FusionOptions;

/// The overrides a `compile_with` call passes, as the core's request.
pub(crate) fn compile_request(
    fusion_options: Option<PyRef<PyFusionOptions>>,
    kernel_dispatch: Option<&str>,
    policy: Option<rlx_runtime::PrecisionPolicy>,
) -> PyResult<CompileRequest> {
    let kernel_dispatch = kernel_dispatch
        .map(rlx_bind::runtime::session::kernel_dispatch)
        .transpose()
        .map_err(crate::bind::to_py)?;
    Ok(CompileRequest {
        fusion: fusion_options.map(|fo| fo.to_rust()),
        kernel_dispatch,
        policy,
    })
}

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
    /// Keep conv + bias + activation as separate ops instead of
    /// `FusedConvBiasAct` — for A/B-ing the fused kernel.
    #[pyo3(get, set)]
    pub disable_conv_bias_act_fusion: bool,
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
        disable_conv_bias_act_fusion = false,
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
        disable_conv_bias_act_fusion: bool,
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
            disable_conv_bias_act_fusion,
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
            disable_conv_bias_act_fusion: false,
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
            disable_conv_bias_act_fusion: self.disable_conv_bias_act_fusion,
            no_io_peaks_output: false,
            elementwise_only: false,
            no_weight_concat: false,
        }
    }
}
