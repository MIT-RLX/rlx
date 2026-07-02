// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! IR passes shared by [`crate::TpuExecutable::compile`] and orchestrated HLO segments.

use rlx_ir::Graph;
use rlx_opt::pass::Pass as _;

/// Run the TPU pre-HLO pipeline (DCE, tier-2 fusions, elementwise regions, FKL, unfuse).
pub fn prepare_graph_for_hlo(graph: Graph) -> Graph {
    let graph = rlx_opt::DeadCodeElimination.run(graph);
    let graph = rlx_opt::ConstantFolding.run(graph);
    let graph = rlx_opt::FuseResidualLN.run(graph);
    let graph = rlx_opt::FuseResidualRmsNorm.run(graph);
    let graph = rlx_opt::FuseRmsNormReshape.run(graph);
    let graph = rlx_opt::FuseMatMulBiasAct.run(graph);
    let graph = rlx_opt::LegalizeBroadcast.run(graph);
    let graph = rlx_opt::MarkElementwiseRegions.run(graph);
    let graph = crate::fk_pipeline::apply_fk_passes(graph);
    crate::unfuse::unfuse(graph)
}
