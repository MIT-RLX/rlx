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

//! Soft-port list builder for `top.sv` from [`IoConfig`].

use crate::codegen::relu::bits_for;
use crate::export_config::{IoConfig, PortNames, SidebandSpec};
use crate::model::Model;

/// Build the `top` module port list for the configured I/O mode.
pub fn top_ports(model: &Model, io: &IoConfig) -> Vec<String> {
    let n = &io.names;
    let in_addr_bits = bits_for(model.input_len.max(1));
    let out_len = primary_out_len(model);
    let out_addr_bits = bits_for(out_len.max(1));
    let mut ports = vec![
        format!("input  logic                       {}", n.clk),
        format!("input  logic                       {}", n.rst),
        format!("input  logic                       {}", n.start),
        format!("output logic                       {}", n.done),
    ];
    if io.input.wants_memory() {
        ports.push(format!(
            "input  logic [{}:0]              {}",
            in_addr_bits - 1,
            n.in_addr
        ));
        ports.push(format!("input  logic                       {}", n.in_we));
        ports.push(format!("input  logic signed [7:0]          {}", n.in_din));
    }
    if io.input.wants_stream() {
        let b = io.input.beat_elems() as usize;
        let w = 8 * b;
        ports.push(format!("input  logic                       {}", n.in_valid));
        ports.push(format!("output logic                       {}", n.in_ready));
        ports.push(format!(
            "input  logic [{}:0]              {}",
            w - 1,
            n.in_data
        ));
    }
    for sb in &io.sidebands {
        ports.extend(sideband_ports(sb));
    }
    if io.output.wants_pred_port() {
        ports.push(format!("output logic signed [7:0]          {}", n.pred));
    }
    if io.output.wants_memory() {
        ports.push(format!(
            "input  logic [{}:0]              {}",
            out_addr_bits - 1,
            n.out_addr
        ));
        ports.push(format!("input  logic                       {}", n.out_re));
        ports.push(format!("output logic signed [7:0]          {}", n.out_dout));
    }
    if io.output.wants_stream() {
        let b = io.output.beat_elems() as usize;
        let w = 8 * b;
        ports.push(format!(
            "output logic                       {}",
            n.out_valid
        ));
        ports.push(format!(
            "input  logic                       {}",
            n.out_ready
        ));
        ports.push(format!(
            "output logic [{}:0]              {}",
            w - 1,
            n.out_data
        ));
    }
    // Extra named readout banks (bind.outputs[1..]).
    for (i, extra) in model.extra_outputs.iter().enumerate() {
        let ab = bits_for(extra.len.max(1));
        let stem = sanitize_port(&extra.name);
        ports.push(format!(
            "input  logic [{}:0]              {stem}_addr /* extra[{i}] */",
            ab - 1
        ));
        ports.push(format!("input  logic                       {stem}_re"));
        ports.push(format!("output logic signed [7:0]          {stem}_dout"));
    }
    ports
}

fn sideband_ports(sb: &SidebandSpec) -> Vec<String> {
    let name = sanitize_port(&sb.name);
    let w = sb.bits.max(1) as usize;
    let signed = if sb.signed { "signed " } else { "" };
    let mut out = vec![format!(
        "input  logic {signed}[{}:0]          {name}",
        w - 1
    )];
    if sb.echo {
        out.push(format!(
            "output logic {signed}[{}:0]         {name}_q",
            w - 1
        ));
    }
    out
}

pub fn primary_out_len(model: &Model) -> usize {
    model.layers.last().map(|l| l.out_len()).unwrap_or(1).max(1)
}

pub fn sanitize_port(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() || s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        s = format!("o_{s}");
    }
    s
}

/// Markdown port table rows for EXPORT.md.
pub fn port_table_md(io: &IoConfig, model: &Model) -> String {
    let n = &io.names;
    let mut rows = vec![
        format!("| `{}` | in | clock |", n.clk),
        format!("| `{}` | in | synchronous reset |", n.rst),
        format!("| `{}` | in | pulse to begin inference |", n.start),
        format!("| `{}` | out | inference complete |", n.done),
    ];
    if io.input.wants_memory() {
        rows.push(format!(
            "| `{}` / `{}` / `{}` | in | write input activation bytes |",
            n.in_addr, n.in_we, n.in_din
        ));
    }
    if io.input.wants_stream() {
        rows.push(format!(
            "| `{}` / `{}` / `{}` | in/out | stream input (valid/ready/data) |",
            n.in_valid, n.in_ready, n.in_data
        ));
    }
    for sb in &io.sidebands {
        let name = sanitize_port(&sb.name);
        let signed = if sb.signed { "signed " } else { "" };
        rows.push(format!(
            "| `{name}` | in | sideband {signed}{}-bit (sampled at start) |",
            sb.bits
        ));
        if sb.echo {
            rows.push(format!("| `{name}_q` | out | registered sideband echo |"));
        }
    }
    if io.output.wants_pred_port() {
        rows.push(format!(
            "| `{}` | out | argmax / output byte (addr 0 peek) |",
            n.pred
        ));
    }
    if io.output.wants_memory() {
        rows.push(format!(
            "| `{}` / `{}` / `{}` | in/out | read last-layer buffer (len={}) |",
            n.out_addr,
            n.out_re,
            n.out_dout,
            primary_out_len(model)
        ));
    }
    if io.output.wants_stream() {
        rows.push(format!(
            "| `{}` / `{}` / `{}` | out/in | stream last-layer buffer |",
            n.out_valid, n.out_ready, n.out_data
        ));
    }
    for extra in &model.extra_outputs {
        let stem = sanitize_port(&extra.name);
        rows.push(format!(
            "| `{stem}_addr` / `{stem}_re` / `{stem}_dout` | in/out | extra readout `{}` (len={}) |",
            extra.name, extra.len
        ));
    }
    rows.join("\n")
}

pub fn default_names() -> PortNames {
    PortNames::default()
}
