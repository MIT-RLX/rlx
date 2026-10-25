// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Offline FPGA / SystemVerilog export (feature `fpga`) — not a runtime
//! device, the binding form of `rlx_runtime::ExportSession::fpga`.

use rlx_ir::Graph;
use serde_json::Value;

use super::opts::Opts;
use crate::{ApiError, ApiResult};

/// A list option given as a list of strings or one comma-separated string.
fn names(o: &Opts<'_>, name: &str) -> ApiResult<Option<Vec<String>>> {
    let Some(v) = o.get(name) else {
        return Ok(None);
    };
    let items: Vec<String> = match v {
        Value::String(s) => s.split(',').map(|p| p.trim().to_string()).collect(),
        Value::Array(a) => a
            .iter()
            .map(|x| {
                x.as_str().map(str::to_string).ok_or_else(|| {
                    ApiError::type_(format!("fpga option '{name}': expected strings"))
                })
            })
            .collect::<ApiResult<_>>()?,
        _ => {
            return Err(ApiError::type_(format!(
                "fpga option '{name}': expected a string or a list of strings"
            )));
        }
    };
    Ok(Some(items.into_iter().filter(|s| !s.is_empty()).collect()))
}

/// Export `graph` as SystemVerilog under `out_dir`; returns the files written.
///
/// Options: `quant` (`"int8"` | `"int4"` | `"fp4"` | `"ternary"`), `hw`
/// (`"generic"` | `"ecp5"` | `"ice40"` | `"xilinx7:PART"` | `"xc7:BOARD"`),
/// `in_iface` (`"memory"` | `"stream"` | `"both"`), `out_iface` (`"scalar"` |
/// `"memory"` | `"stream"` | `"scalar+memory"` | `"both"`), `bind_in`,
/// `bind_out`, `sideband` (`name[:bits[:signed]]`), `bind_sideband`.
pub fn export(graph: &Graph, out_dir: &str, opts: &Value) -> ApiResult<Vec<String>> {
    let o = Opts::new("fpga", opts)?;
    let runtime = |e: String| ApiError::runtime(e);
    let mode =
        rlx_fpga::ExportQuantMode::parse(o.str("quant")?.unwrap_or("int8")).map_err(runtime)?;
    let hw = rlx_fpga::HwTarget::parse(o.str("hw")?.unwrap_or("generic")).map_err(runtime)?;
    let input =
        rlx_fpga::InputIface::parse(o.str("in_iface")?.unwrap_or("memory")).map_err(runtime)?;
    let output =
        rlx_fpga::OutputIface::parse(o.str("out_iface")?.unwrap_or("scalar")).map_err(runtime)?;
    let mut io = rlx_fpga::IoConfig::default()
        .with_input(input)
        .with_output(output);
    for part in names(&o, "sideband")?.unwrap_or_default() {
        io = io.sideband(rlx_fpga::SidebandSpec::parse(&part).map_err(runtime)?);
    }
    if let Some(list) = names(&o, "bind_sideband")? {
        io.bind.sideband_inputs = list;
    }
    let bind_in = o.str("bind_in")?;
    let bind_out = names(&o, "bind_out")?;
    o.finish()?;
    let mut sess = rlx_runtime::ExportSession::fpga(out_dir)
        .quant_mode(mode)
        .hw_target(hw)
        .io(io);
    if let Some(name) = bind_in {
        sess = sess.bind_input(name);
    }
    if let Some(list) = bind_out {
        sess = sess.bind_outputs(list);
    }
    let arts = crate::panics::catch("export_fpga", || sess.export(graph))?.map_err(runtime)?;
    Ok(arts.files)
}
