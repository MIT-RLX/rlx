// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-level Graph → SystemVerilog export.

use std::io;
use std::path::{Path, PathBuf};

use rlx_ir::Graph;

use crate::codegen::{collect_artifacts_opt, emit_with_config};
use crate::estimate::{Estimate, estimate};
use crate::export_config::{FpgaExportConfig, OutputKind};
use crate::legalize::prepare_model;
use crate::model::{Layer, Model};
use crate::passes::{optimize, summary};
use crate::sideband::resolve_graph_sidebands;

pub use crate::export_config::{HwTarget, OutputKind as ExportOutputKind};
pub use crate::from_graph::FromGraphOptions;
pub use crate::legalize::{ExportQuantMode, LegalizeOptions};

/// Result of a successful FPGA export.
#[derive(Debug, Clone)]
pub struct ExportResult {
    pub out_dir: PathBuf,
    pub model_name: String,
    pub estimate: Estimate,
    pub optimize_summary: String,
    pub files: Vec<String>,
}

/// Lower `graph` → FPGA [`Model`] → optimize → emit SystemVerilog under `out_dir`.
///
/// Runs [`prepare_model`] first (pass-through for Q* graphs, PTQ for f32).
pub fn export_graph(
    graph: &Graph,
    cfg: &FpgaExportConfig,
    out_dir: &Path,
) -> Result<ExportResult, String> {
    let mut cfg = cfg.clone();
    // Io bind is authoritative when set on FpgaExportConfig.io.
    if cfg.io.bind.input.is_some()
        || !cfg.io.bind.outputs.is_empty()
        || !cfg.io.bind.sideband_inputs.is_empty()
    {
        cfg.from_graph.bind = cfg.io.bind.clone();
    }
    resolve_graph_sidebands(graph, &mut cfg.io)?;
    let mut model = prepare_model(graph, cfg.quant_mode, &cfg.legalize, &cfg.from_graph)?;
    apply_output_kind(&mut model, cfg.output_kind);
    export_model(&model, &cfg, out_dir).map_err(|e| e.to_string())
}

fn apply_output_kind(model: &mut Model, kind: OutputKind) {
    match kind {
        OutputKind::Argmax => {}
        OutputKind::Logits => {
            while matches!(model.layers.last(), Some(Layer::Argmax { .. })) {
                model.layers.pop();
            }
        }
    }
}

/// Emit an already-built [`Model`] with full export configuration.
pub fn export_model(
    model: &Model,
    cfg: &FpgaExportConfig,
    out_dir: &Path,
) -> io::Result<ExportResult> {
    let opt = optimize(model, &cfg.tune);
    let est = estimate(&opt);
    let optimize_summary = summary(&opt);
    emit_with_config(model, cfg, out_dir)?;

    let files: Vec<String> = collect_artifacts_opt(&opt)
        .into_iter()
        .map(|a| a.rel_path)
        .chain(
            crate::codegen::synth::collect_synth_artifacts(model, cfg)
                .into_iter()
                .map(|a| a.rel_path),
        )
        .chain(
            crate::codegen::board_shell::collect_board_shell(model, &cfg.hw_target, &cfg.io)
                .into_iter()
                .map(|a| a.rel_path),
        )
        .collect();

    Ok(ExportResult {
        out_dir: out_dir.to_path_buf(),
        model_name: model.name.clone(),
        estimate: est,
        optimize_summary,
        files,
    })
}
