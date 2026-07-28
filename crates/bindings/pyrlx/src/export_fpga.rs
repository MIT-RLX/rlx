// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Offline FPGA / SystemVerilog export (feature `fpga`).
//!
//! Not a runtime Device — counterpart to Rust [`rlx_runtime::ExportSession`].

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::graph::PyGraph;

fn consumed() -> PyErr {
    PyRuntimeError::new_err("graph already consumed")
}

/// Export an INT8 (or legalizable f32) graph to SystemVerilog under `out_dir`.
///
/// Args:
///   graph: pyrlx Graph (not consumed)
///   out_dir: output directory
///   quant: "int8" | "int4" | "fp4" (default "int8")
///   hw: "generic" | "ecp5" | "ice40" | "xilinx7:PART" (default "generic")
///   in_iface: "memory" | "stream" | "both" (default "memory")
///   out_iface: "scalar" | "memory" | "stream" | "scalar+memory" | "both"
///   bind_in: optional Graph input name
///   bind_out: optional comma-separated output / layer names
///   sideband: optional comma-separated `name[:bits[:signed]]` soft ports
///   bind_sideband: optional comma-separated Graph Input names (scalar)
#[pyfunction]
#[pyo3(signature = (
    graph,
    out_dir,
    quant="int8",
    hw="generic",
    in_iface="memory",
    out_iface="scalar",
    bind_in=None,
    bind_out=None,
    sideband=None,
    bind_sideband=None,
))]
pub fn export_fpga(
    graph: &Bound<'_, PyGraph>,
    out_dir: &str,
    quant: &str,
    hw: &str,
    in_iface: &str,
    out_iface: &str,
    bind_in: Option<&str>,
    bind_out: Option<&str>,
    sideband: Option<&str>,
    bind_sideband: Option<&str>,
) -> PyResult<Vec<String>> {
    let g = graph.borrow();
    let inner = g.inner.as_ref().ok_or_else(consumed)?;
    let mode = rlx_fpga::ExportQuantMode::parse(quant).map_err(PyRuntimeError::new_err)?;
    let hw_t = rlx_fpga::HwTarget::parse(hw).map_err(PyRuntimeError::new_err)?;
    let in_t = rlx_fpga::InputIface::parse(in_iface).map_err(PyRuntimeError::new_err)?;
    let out_t = rlx_fpga::OutputIface::parse(out_iface).map_err(PyRuntimeError::new_err)?;
    let mut io = rlx_fpga::IoConfig::default()
        .with_input(in_t)
        .with_output(out_t);
    if let Some(list) = sideband {
        for part in list.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let spec = rlx_fpga::SidebandSpec::parse(part).map_err(PyRuntimeError::new_err)?;
            io = io.sideband(spec);
        }
    }
    if let Some(list) = bind_sideband {
        let names: Vec<_> = list
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        io.bind.sideband_inputs = names;
    }
    let mut sess = rlx_runtime::ExportSession::fpga(out_dir)
        .quant_mode(mode)
        .hw_target(hw_t)
        .io(io);
    if let Some(name) = bind_in {
        sess = sess.bind_input(name);
    }
    if let Some(list) = bind_out {
        let names: Vec<_> = list
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        sess = sess.bind_outputs(names);
    }
    let arts = sess.export(inner).map_err(PyRuntimeError::new_err)?;
    Ok(arts.files)
}
