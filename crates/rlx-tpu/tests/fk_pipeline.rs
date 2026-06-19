// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! FKL pass wiring for direct `TpuExecutable::compile`.

use rlx_fusion::fk_graphs::batch_narrow_relu_primitive_graph;
use rlx_ir::Op;
use rlx_tpu::fk_pipeline::apply_fk_passes;
use rlx_tpu::ir_passes::prepare_graph_for_hlo;
use rlx_tpu::lower::lower_graph;

#[test]
fn apply_fk_passes_fuses_batch_narrow_relu_primitive() {
    let g = batch_narrow_relu_primitive_graph("tpu_fk", 2, 3, 4, 4);
    let out = apply_fk_passes(g);
    assert!(
        out.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. })),
        "TPU FK pipeline should produce BatchElementwiseRegion"
    );
    assert!(
        !out.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Concat { .. })),
        "standalone concat should be folded away"
    );
}

#[test]
fn prepare_graph_for_hlo_lowers_fused_batch_primitive() {
    let g = batch_narrow_relu_primitive_graph("tpu_full", 2, 3, 4, 4);
    let prepared = prepare_graph_for_hlo(g);
    assert!(
        prepared
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
    );
    let module = lower_graph(&prepared);
    let bytes = &module.bytes;
    assert!(bytes.windows(11).any(|w| w == b"concatenate"));
}
