// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Pipeline inspection — dump every HIR / MIR / LIR stage as text.

use std::fmt::Write as _;

use rlx_ir::hir::{HirModule, LowerError};
use rlx_ir::{inspect_graph_diff, inspect_hir, inspect_lir, inspect_mir, inspect_mir_stats};

use crate::compiler::{CompilePipeline, CompileResult};
use rlx_fusion::fusion_report::FusionReport;

/// Text dump of each compiler pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineInspect {
    pub hir: String,
    pub mir_lowered: String,
    pub mir_diff: String,
    pub mir_optimized: String,
    pub lir: String,
    pub fusion: String,
}

impl std::fmt::Display for PipelineInspect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln_section(f, "HIR", &self.hir)?;
        writeln_section(f, "MIR (lowered)", &self.mir_lowered)?;
        if !self.mir_diff.is_empty() {
            writeln_section(f, "MIR (fusion diff)", &self.mir_diff)?;
        }
        writeln_section(f, "MIR (optimized)", &self.mir_optimized)?;
        writeln_section(f, "FUSION", &self.fusion)?;
        writeln_section(f, "LIR", &self.lir)
    }
}

fn writeln_section(f: &mut std::fmt::Formatter<'_>, title: &str, body: &str) -> std::fmt::Result {
    let mut header = String::new();
    banner(&mut header, title);
    write!(f, "{header}{body}")?;
    if !body.ends_with('\n') {
        writeln!(f)?;
    }
    Ok(())
}

/// Inspect every lowering stage for `hir` through `pipeline`.
pub fn inspect_pipeline(
    pipeline: &CompilePipeline,
    hir: HirModule,
) -> Result<PipelineInspect, LowerError> {
    let hir_text = inspect_hir(&hir);
    let mir_raw = CompilePipeline::lower_hir(hir)?;
    let mir_lowered = inspect_mir(&mir_raw);
    let mir_before = mir_raw.clone();
    let (mir_opt, fusion) = pipeline.optimize_with_report(mir_raw);
    let mir_diff = inspect_graph_diff(mir_before.as_graph(), mir_opt.as_graph());
    let fusion_text = format!("{}\n{}", fusion, inspect_mir_stats(&mir_opt));
    let lir = pipeline.plan_lir(mir_opt);
    Ok(PipelineInspect {
        hir: hir_text,
        mir_lowered,
        mir_diff,
        mir_optimized: inspect_mir(&lir.mir),
        lir: inspect_lir(&lir),
        fusion: fusion_text,
    })
}

/// Inspect a completed [`CompileResult`] plus the original HIR text.
pub fn inspect_compiled(hir_text: &str, result: &CompileResult) -> PipelineInspect {
    PipelineInspect {
        hir: hir_text.to_string(),
        mir_lowered: String::new(),
        mir_diff: String::new(),
        mir_optimized: inspect_mir(&result.lir.mir),
        lir: inspect_lir(&result.lir),
        fusion: format!("{}", result.fusion),
    }
}

/// Write a full pipeline dump when `RLX_IR_DUMP` is set (path prefix or directory).
pub fn maybe_dump_pipeline(dump: &PipelineInspect, module_name: &str) {
    let Some(path) = rlx_ir::env::var("RLX_IR_DUMP") else {
        return;
    };
    let target = if path.ends_with('/') || path.ends_with('\\') {
        format!("{path}{module_name}.ir.txt")
    } else {
        path
    };
    if let Err(e) = std::fs::write(&target, dump.to_string()) {
        eprintln!("[rlx] RLX_IR_DUMP write failed ({target}): {e}");
    } else {
        eprintln!("[rlx] wrote IR dump to {target}");
    }
}

/// Fusion report only (post-optimize diagnostics).
pub fn inspect_fusion(report: &FusionReport) -> String {
    format!("{report}")
}

fn banner(out: &mut String, title: &str) {
    let line = "═".repeat(title.len() + 4);
    writeln!(out, "{line}").unwrap();
    writeln!(out, "══ {title} ══").unwrap();
    writeln!(out, "{line}").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;
    use rlx_ir::Shape;

    fn f32_shape(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn inspect_pipeline_covers_all_stages() {
        let mut hir = HirModule::new("probe");
        let x = hir.input("x", f32_shape(&[2, 64]));
        let w = hir.param("w", f32_shape(&[64, 64]));
        let h = hir.linear(x, w, None, None, f32_shape(&[2, 64]));
        hir.outputs = vec![h];

        let pipe = CompilePipeline::default();
        let dump = inspect_pipeline(&pipe, hir).expect("inspect");
        assert!(dump.hir.contains("hir @probe"));
        assert!(dump.mir_lowered.contains("mir @probe"));
        assert!(dump.mir_optimized.contains("mir @probe"));
        assert!(dump.lir.contains("lir @probe"));
        assert!(dump.fusion.contains("nodes="));
        let full = dump.to_string();
        assert!(full.contains("══ HIR ══"));
        assert!(full.contains("══ LIR ══"));
    }
}
