// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! IO-gated output selection — prefer peaks-only readback when fusion wins.

use rlx_fusion::pass::Pass;
use rlx_ir::{Graph, NodeId, Op};

use crate::fusion_benefit::profile_graph_io_outputs;
use crate::fusion_pipeline::{FusionTarget, should_fuse_with_target};
use crate::fusion_target::active_fusion_target;

/// When an FFT spectrum is a graph output but `Op::WelchPeaks` consumes it,
/// drop spectrum from outputs if the per-target IO gate favors peaks-only readback.
pub struct SelectPeaksOnlyOutputs;

impl Pass for SelectPeaksOnlyOutputs {
    fn name(&self) -> &str {
        "select_peaks_only_outputs"
    }

    fn run(&self, mut graph: Graph) -> Graph {
        if rlx_ir::env::flag("RLX_NO_IO_PEAKS_OUTPUT") {
            return graph;
        }
        let Some(target) = active_fusion_target() else {
            return graph;
        };
        if target == FusionTarget::Cpu {
            return graph;
        }

        let pairs = fft_welch_peaks_pairs(&graph);
        if pairs.is_empty() {
            return graph;
        }

        let mut outputs: Vec<NodeId> = graph.outputs.clone();
        let mut changed = false;

        for (fft_id, peaks_id) in pairs {
            if !outputs.contains(&fft_id) {
                continue;
            }
            let peaks_already_out = outputs.contains(&peaks_id);
            let mut after_outputs: Vec<NodeId> =
                outputs.iter().copied().filter(|&id| id != fft_id).collect();
            if !peaks_already_out {
                after_outputs.push(peaks_id);
            }
            if peaks_already_out {
                // Redundant spectrum readback when peaks are already materialized.
                outputs = after_outputs;
                changed = true;
                continue;
            }
            let before = profile_graph_io_outputs(&graph, &outputs);
            let after = profile_graph_io_outputs(&graph, &after_outputs);
            if should_fuse_with_target(target, &before, &after) {
                outputs = after_outputs;
                changed = true;
            }
        }

        if changed {
            graph.set_outputs(outputs);
        }
        graph
    }
}

fn fft_welch_peaks_pairs(graph: &Graph) -> Vec<(NodeId, NodeId)> {
    let mut pairs = Vec::new();
    for node in graph.nodes() {
        let Op::WelchPeaks { .. } = node.op else {
            continue;
        };
        if node.inputs.is_empty() {
            continue;
        }
        let spec_id = node.inputs[0];
        if matches!(graph.node(spec_id).op, Op::Fft { .. }) {
            pairs.push((spec_id, node.id));
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion_benefit::profile_graph_io;
    use crate::fusion_target::with_fusion_target;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::{DType, Shape};

    fn fused_graph_with_spectrum_output(batch: usize, n_fft: usize, k: usize) -> Graph {
        let mut g = Graph::new("peaks_gate");
        let segs = g.input("segs", Shape::new(&[batch * 2, n_fft], DType::F32));
        let zeros = g.sub(segs, segs);
        let block = g.concat_(vec![segs, zeros], 1);
        let spec = g.fft(block, false);
        let peaks = g.welch_peaks(spec, k, 2);
        g.set_outputs(vec![spec, peaks]);
        g
    }

    #[test]
    fn metal_gate_drops_spectrum_output() {
        let g = fused_graph_with_spectrum_output(1024, 256, 16);
        let before = profile_graph_io(&g);
        assert!(before.host_output_bytes > 1_000_000);

        let out = with_fusion_target(FusionTarget::Metal, || SelectPeaksOnlyOutputs.run(g));
        let after = profile_graph_io(&out);
        assert_eq!(out.outputs.len(), 1);
        assert!(matches!(out.node(out.outputs[0]).op, Op::WelchPeaks { .. }));
        assert!(after.host_output_bytes < before.host_output_bytes);
    }

    #[test]
    fn dual_output_always_drops_redundant_spectrum() {
        let g = fused_graph_with_spectrum_output(8192, 256, 16);
        let out = with_fusion_target(FusionTarget::Wgpu, || SelectPeaksOnlyOutputs.run(g));
        assert_eq!(out.outputs.len(), 1);
        assert!(matches!(out.node(out.outputs[0]).op, Op::WelchPeaks { .. }));
    }
}
