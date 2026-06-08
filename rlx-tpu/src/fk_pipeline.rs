// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! FKL fusion passes for [`crate::TpuExecutable::compile`] (after `MarkElementwiseRegions`).

use rlx_compile::fusion_pipeline::{
    FusionOptions, FusionTarget, fk_passes_after_elementwise_regions, supported_for_target,
};
use rlx_fusion::run_passes;
use rlx_ir::Graph;

/// Run FKL passes (batch slice / transform / prologue / batch preprocess) using the same
/// options and supported-op claims as the Session compile pipeline for TPU.
pub fn apply_fk_passes(graph: Graph) -> Graph {
    let opts = FusionOptions::default()
        .merge_env()
        .apply_native_fk_defaults(FusionTarget::Tpu);
    let passes = fk_passes_after_elementwise_regions(supported_for_target(FusionTarget::Tpu), opts);
    run_passes(graph, &passes, false)
}
