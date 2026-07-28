// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! First-class hardware **export** (artifacts), parallel to [`crate::Backend`] (run).
//!
//! ```text
//! Session / Backend::compile  →  ExecutableGraph::run()   (µs–ms)
//! ExportSession / export_graph →  SystemVerilog / …       (offline)
//! ```
//!
//! There is **no** `Device::Fpga` — FPGA export never goes through
//! `Session::compile`. Use [`ExportSession`] instead.

#[cfg(feature = "fpga")]
use std::path::Path;
use std::path::PathBuf;

use rlx_ir::Graph;

/// Offline artifact targets (not runtime devices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportTarget {
    /// IR → SystemVerilog datapath (`rlx-fpga`).
    #[cfg(feature = "fpga")]
    Fpga,
}

impl ExportTarget {
    pub fn name(self) -> &'static str {
        match self {
            #[cfg(feature = "fpga")]
            Self::Fpga => "fpga",
        }
    }
}

/// Options for [`export_graph`].
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Destination directory for artifacts.
    pub out_dir: PathBuf,
    /// FPGA-specific configuration (ignored for other targets).
    #[cfg(feature = "fpga")]
    pub fpga: rlx_fpga::FpgaExportConfig,
}

impl ExportOptions {
    pub fn new(out_dir: impl Into<PathBuf>) -> Self {
        Self {
            out_dir: out_dir.into(),
            #[cfg(feature = "fpga")]
            fpga: rlx_fpga::FpgaExportConfig::default(),
        }
    }

    #[cfg(feature = "fpga")]
    pub fn with_fpga(mut self, cfg: rlx_fpga::FpgaExportConfig) -> Self {
        self.fpga = cfg;
        self
    }

    #[cfg(feature = "fpga")]
    pub fn fpga_generic(out_dir: impl Into<PathBuf>) -> Self {
        Self::new(out_dir).with_fpga(
            rlx_fpga::FpgaExportConfig::default().with_hw_target(rlx_fpga::HwTarget::Generic),
        )
    }
}

/// Summary of written artifacts.
#[derive(Debug, Clone)]
pub struct ExportedArtifacts {
    pub target: ExportTarget,
    pub out_dir: PathBuf,
    pub files: Vec<String>,
    pub estimate_summary: Option<String>,
    pub optimize_summary: Option<String>,
}

/// Builder-style offline export API (counterpart to [`crate::Session`]).
///
/// ```ignore
/// let exp = ExportSession::fpga("hw/out")
///     .quant_mode(ExportQuantMode::Int4)
///     .hw_target(HwTarget::Generic);
/// exp.export(&graph)?;
/// ```
#[cfg(feature = "fpga")]
pub struct ExportSession {
    opts: ExportOptions,
}

#[cfg(feature = "fpga")]
impl ExportSession {
    /// Target-agnostic SystemVerilog under `out_dir`.
    pub fn fpga(out_dir: impl Into<PathBuf>) -> Self {
        Self {
            opts: ExportOptions::fpga_generic(out_dir),
        }
    }

    pub fn with_config(mut self, cfg: rlx_fpga::FpgaExportConfig) -> Self {
        self.opts.fpga = cfg;
        self
    }

    pub fn quant_mode(mut self, mode: rlx_fpga::ExportQuantMode) -> Self {
        self.opts.fpga = self.opts.fpga.with_quant_mode(mode);
        self
    }

    pub fn hw_target(mut self, hw: rlx_fpga::HwTarget) -> Self {
        self.opts.fpga = self.opts.fpga.with_hw_target(hw);
        self
    }

    pub fn output_kind(mut self, kind: rlx_fpga::OutputKind) -> Self {
        self.opts.fpga = self.opts.fpga.with_output_kind(kind);
        self
    }

    pub fn io(mut self, io: rlx_fpga::IoConfig) -> Self {
        self.opts.fpga = self.opts.fpga.with_io(io);
        self
    }

    pub fn bind_input(mut self, name: impl Into<String>) -> Self {
        self.opts.fpga = self.opts.fpga.bind_input(name);
        self
    }

    pub fn bind_outputs<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.opts.fpga = self.opts.fpga.bind_outputs(names);
        self
    }

    pub fn bind_sideband_inputs<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.opts.fpga = self.opts.fpga.bind_sideband_inputs(names);
        self
    }

    pub fn sideband(mut self, spec: rlx_fpga::SidebandSpec) -> Self {
        self.opts.fpga = self.opts.fpga.sideband(spec);
        self
    }

    pub fn legalize(mut self, opts: rlx_fpga::LegalizeOptions) -> Self {
        self.opts.fpga = self.opts.fpga.with_legalize(opts);
        self
    }

    pub fn export(&self, graph: &Graph) -> Result<ExportedArtifacts, String> {
        export_graph(graph, ExportTarget::Fpga, &self.opts)
    }

    pub fn export_model(&self, model: &rlx_fpga::Model) -> Result<ExportedArtifacts, String> {
        let result = rlx_fpga::export_model(model, &self.opts.fpga, &self.opts.out_dir)
            .map_err(|e| e.to_string())?;
        Ok(ExportedArtifacts {
            target: ExportTarget::Fpga,
            out_dir: result.out_dir,
            files: result.files,
            estimate_summary: Some(result.estimate.summary()),
            optimize_summary: Some(result.optimize_summary),
        })
    }
}

/// Export `graph` to hardware artifacts for `target`.
pub fn export_graph(
    graph: &Graph,
    target: ExportTarget,
    opts: &ExportOptions,
) -> Result<ExportedArtifacts, String> {
    #[cfg(feature = "fpga")]
    {
        match target {
            ExportTarget::Fpga => export_fpga(graph, opts),
        }
    }
    #[cfg(not(feature = "fpga"))]
    {
        let _ = (graph, target, opts);
        Err("no export targets enabled; enable the `fpga` feature".into())
    }
}

#[cfg(feature = "fpga")]
fn export_fpga(graph: &Graph, opts: &ExportOptions) -> Result<ExportedArtifacts, String> {
    let result = rlx_fpga::export_graph(graph, &opts.fpga, &opts.out_dir)?;
    Ok(ExportedArtifacts {
        target: ExportTarget::Fpga,
        out_dir: result.out_dir,
        files: result.files,
        estimate_summary: Some(result.estimate.summary()),
        optimize_summary: Some(result.optimize_summary),
    })
}

/// Convenience: export TinyConv-MNIST (cortexm weights) as target-agnostic Verilog.
#[cfg(feature = "fpga")]
pub fn export_tinyconv_mnist(
    out_dir: &Path,
    cfg: &rlx_fpga::FpgaExportConfig,
) -> Result<ExportedArtifacts, String> {
    let model = rlx_fpga::tinyconv_mnist_from_cortexm();
    let result = rlx_fpga::export_model(&model, cfg, out_dir).map_err(|e| e.to_string())?;
    Ok(ExportedArtifacts {
        target: ExportTarget::Fpga,
        out_dir: result.out_dir,
        files: result.files,
        estimate_summary: Some(result.estimate.summary()),
        optimize_summary: Some(result.optimize_summary),
    })
}

#[cfg(all(test, feature = "fpga"))]
mod tests {
    use super::*;
    use rlx_fpga::ir::to_graph;
    use rlx_fpga::tinyconv_mnist_from_cortexm;

    #[test]
    fn export_session_tinyconv_generic() {
        let model = tinyconv_mnist_from_cortexm();
        let ir = to_graph(&model);
        let dir = tempfile::tempdir().unwrap();
        let arts = ExportSession::fpga(dir.path())
            .quant_mode(rlx_fpga::ExportQuantMode::Int8)
            .export(&ir.graph)
            .expect("export");
        assert!(dir.path().join("top.sv").is_file());
        assert!(dir.path().join("synth.sh").is_file());
        assert!(arts.files.iter().any(|f| f == "top.sv"));
    }

    #[test]
    fn export_ecp5_emits_board_shell() {
        let dir = tempfile::tempdir().unwrap();
        let cfg =
            rlx_fpga::FpgaExportConfig::default().with_hw_target(rlx_fpga::HwTarget::ecp5_25k());
        export_tinyconv_mnist(dir.path(), &cfg).unwrap();
        assert!(dir.path().join("board_top.sv").is_file());
        assert!(dir.path().join("constraints.lpf").is_file());
    }
}
