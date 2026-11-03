// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ship a weight-space program: bake the weights into a batch-1 graph and
//! hand it to rlx's offline paths.
//!
//! Written programs are often tiny and sparse (Claude's Bounce program used
//! 400 nonzero weights), which suits int8 MCU and FPGA targets. `export`
//! writes:
//!
//! - `model.rlxmodel` — the graph with every param baked into
//!   `Op::Constant` (stable IR text, loadable anywhere rlx runs)
//! - `weights.json` — the params, row-major
//! - `export.json` — size / sparsity per tensor and the held-out score with
//!   int8 per-tensor symmetric weights next to the f32 score
//! - `fpga/` — SystemVerilog via `rlx_runtime::export` (feature `fpga`)

use crate::space::{Tensor, WeightSpace};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

/// The network at batch 1 with params baked as constants.
pub fn baked_graph(ws: &WeightSpace) -> Result<rlx_ir::Graph, String> {
    let mut g = match &ws.state.graph {
        Some(text) => rlx_ir::rlxmodel::from_text(text).map_err(|e| e.to_string())?,
        None => ws.state.mlp.build_graph(1),
    };
    for n in g.nodes_mut() {
        if let rlx_ir::Op::Param { name } = &n.op {
            let t = ws.param(name)?;
            let data: Vec<u8> = t.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            n.op = rlx_ir::Op::Constant { data };
        }
    }
    Ok(g)
}

/// Per-tensor symmetric int8, dequantised back to f32.
pub fn int8_params(params: &BTreeMap<String, Tensor>) -> BTreeMap<String, Tensor> {
    params
        .iter()
        .map(|(n, t)| {
            let max = t.data.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
            let data = t
                .data
                .iter()
                .map(|v| (v / scale).round().clamp(-127.0, 127.0) * scale)
                .collect();
            (
                n.clone(),
                Tensor {
                    rows: t.rows,
                    cols: t.cols,
                    data,
                },
            )
        })
        .collect()
}

/// Write the artifacts under `out`; `fpga` adds SystemVerilog (needs the
/// `fpga` feature). Returns the `export.json` document.
pub fn export(
    ws: &mut WeightSpace,
    out: &Path,
    fpga: bool,
    episodes: usize,
) -> Result<Value, String> {
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let g = baked_graph(ws)?;
    let text = rlx_ir::rlxmodel::to_text(&g).map_err(|e| e.to_string())?;
    std::fs::write(out.join("model.rlxmodel"), text).map_err(|e| e.to_string())?;
    let weights: BTreeMap<&String, &Vec<f32>> =
        ws.state.params.iter().map(|(n, t)| (n, &t.data)).collect();
    std::fs::write(
        out.join("weights.json"),
        serde_json::to_string(&weights).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let tensors: Vec<Value> = ws
        .state
        .params
        .iter()
        .map(|(n, t)| {
            let mut distinct: Vec<u32> = t.data.iter().map(|v| v.to_bits()).collect();
            distinct.sort_unstable();
            distinct.dedup();
            json!({
                "name": n, "rows": t.rows, "cols": t.cols,
                "nonzero": t.data.iter().filter(|v| **v != 0.0).count(),
                "distinct_values": distinct.len(),
            })
        })
        .collect();
    let total: usize = ws.state.params.values().map(|t| t.data.len()).sum();
    let nonzero: usize = ws
        .state
        .params
        .values()
        .map(|t| t.data.iter().filter(|v| **v != 0.0).count())
        .sum();

    let scored = ws.state.graph.is_none();
    let (f32_score, int8_score) = if scored {
        let live = ws.state.params.clone();
        let f = ws.held_out(episodes)?;
        ws.set_params(int8_params(&live));
        let q = ws.held_out(episodes);
        ws.set_params(live);
        (f, q?)
    } else {
        (Value::Null, Value::Null)
    };

    let fpga_result = if fpga {
        export_fpga(&g, &out.join("fpga"))
    } else {
        Value::Null
    };
    let doc = json!({
        "files": ["model.rlxmodel", "weights.json", "export.json"],
        "weights": { "total": total, "nonzero": nonzero, "sparsity": 1.0 - nonzero as f64 / total.max(1) as f64 },
        "tensors": tensors,
        "held_out": { "f32": f32_score, "int8_per_tensor": int8_score },
        "fpga": fpga_result,
    });
    std::fs::write(out.join("export.json"), format!("{doc:#}\n")).map_err(|e| e.to_string())?;
    Ok(doc)
}

#[cfg(feature = "fpga")]
fn export_fpga(g: &rlx_ir::Graph, dir: &Path) -> Value {
    use rlx_runtime::export::{ExportOptions, ExportTarget, export_graph};
    match export_graph(g, ExportTarget::Fpga, &ExportOptions::fpga_generic(dir)) {
        Ok(a) => json!({
            "dir": a.out_dir, "files": a.files,
            "estimate": a.estimate_summary, "optimize": a.optimize_summary,
        }),
        Err(e) => json!({ "error": e }),
    }
}

#[cfg(not(feature = "fpga"))]
fn export_fpga(_: &rlx_ir::Graph, _: &Path) -> Value {
    json!({ "error": "rebuild rlx-wsp with `--features fpga` for SystemVerilog export" })
}
