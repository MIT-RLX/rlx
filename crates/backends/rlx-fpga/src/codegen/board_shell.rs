// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Board shell wrapping soft-port `top` for a concrete FPGA family.
//!
//! Soft RTL stays target-agnostic; this module emits an optional
//! `board_top.sv` that maps board pins → `top` ports. Pin names are
//! parameters so a real board PCF/LPF can override them.

use crate::codegen::Artifact;
use crate::codegen::io_ports::sanitize_port;
use crate::export_config::{HwTarget, IoConfig};
use crate::model::Model;
use crate::verilog::V;

/// Emit `board_top.sv` when the hardware target is board-specific.
///
/// Assumes default soft port names (`clk`/`rst`/`start`/`done`/`in_*`/`pred`).
/// Scalar sidebands are tied to zero (override in a custom shell for live
/// sensors).
pub fn collect_board_shell(model: &Model, hw: &HwTarget, io: &IoConfig) -> Option<Artifact> {
    match hw {
        HwTarget::Generic => None,
        HwTarget::Ecp5 { .. } => Some(Artifact {
            rel_path: "board_top.sv".into(),
            content: emit_ecp5_shell(&model.name, io),
        }),
        HwTarget::Ice40 { .. } => Some(Artifact {
            rel_path: "board_top.sv".into(),
            content: emit_ice40_shell(&model.name, io),
        }),
        HwTarget::Xilinx7 { .. } => Some(Artifact {
            rel_path: "board_top.sv".into(),
            content: emit_xilinx7_shell(&model.name, io),
        }),
    }
}

fn emit_sideband_tieoffs(v: &mut V, io: &IoConfig) {
    if io.sidebands.is_empty() {
        return;
    }
    v.comment("Scalar sidebands — tied off in the stock board shell.");
    for sb in &io.sidebands {
        let name = sanitize_port(&sb.name);
        let w = sb.bits.max(1) as usize;
        let signed = if sb.signed { "signed " } else { "" };
        v.line(&format!("logic {signed}[{}:0] {name};", w - 1));
        v.line(&format!("assign {name} = '0;"));
        if sb.echo {
            v.line(&format!("logic {signed}[{}:0] {name}_q;", w - 1));
        }
    }
}

fn sideband_port_conns(io: &IoConfig) -> String {
    let mut parts = Vec::new();
    for sb in &io.sidebands {
        let name = sanitize_port(&sb.name);
        parts.push(format!(".{name}({name})"));
        if sb.echo {
            parts.push(format!(".{name}_q({name}_q)"));
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(", {}", parts.join(", "))
    }
}

fn emit_ecp5_shell(model_name: &str, io: &IoConfig) -> String {
    let mut v = V::new();
    v.banner(&format!(
        "board_top — ECP5 / ULX3S-class shell over soft `top` ({model_name})"
    ));
    v.comment("Map board pins to soft ports. Override LOCATE in constraints.lpf.");
    v.blank();
    v.module(
        "board_top",
        &[],
        &[
            "input  logic clk_25mhz".into(),
            "input  logic btn_rst_n".into(),
            "input  logic btn_start".into(),
            "output logic led_done".into(),
            "output logic led_pred0".into(),
            "output logic led_pred1".into(),
            "output logic led_pred2".into(),
            "output logic led_pred3".into(),
        ],
        |v| {
            v.line("logic rst;");
            v.line("assign rst = ~btn_rst_n;");
            v.blank();
            v.line("logic        start;");
            v.line("logic        done;");
            v.line("logic [9:0]  in_addr;");
            v.line("logic        in_we;");
            v.line("logic signed [7:0] in_din;");
            v.line("logic signed [7:0] pred;");
            emit_sideband_tieoffs(v, io);
            v.blank();
            v.comment("Host load path is left open — drive in_* from a UART/SPI bridge.");
            v.line("assign start   = btn_start;");
            v.line("assign in_addr = '0;");
            v.line("assign in_we   = 1'b0;");
            v.line("assign in_din  = '0;");
            v.blank();
            let sb = sideband_port_conns(io);
            v.line("top u_top (");
            v.block(|v| {
                v.line(".clk(clk_25mhz), .rst(rst), .start(start), .done(done),");
                v.line(&format!(
                    ".in_addr(in_addr), .in_we(in_we), .in_din(in_din), .pred(pred){sb}"
                ));
            });
            v.line(");");
            v.blank();
            v.line("assign led_done  = done;");
            v.line("assign led_pred0 = pred[0];");
            v.line("assign led_pred1 = pred[1];");
            v.line("assign led_pred2 = pred[2];");
            v.line("assign led_pred3 = pred[3];");
        },
    );
    v.into_string()
}

fn emit_ice40_shell(model_name: &str, io: &IoConfig) -> String {
    let mut v = V::new();
    v.banner(&format!(
        "board_top — iCE40 shell over soft `top` ({model_name})"
    ));
    v.module(
        "board_top",
        &[],
        &[
            "input  logic clk".into(),
            "input  logic rst_n".into(),
            "input  logic start_btn".into(),
            "output logic done_led".into(),
            "output logic [3:0] pred_nibble".into(),
        ],
        |v| {
            v.line("logic rst; assign rst = ~rst_n;");
            v.line("logic done;");
            v.line("logic signed [7:0] pred;");
            v.line("logic [9:0] in_addr; logic in_we; logic signed [7:0] in_din;");
            v.line("assign in_addr = '0; assign in_we = 1'b0; assign in_din = '0;");
            emit_sideband_tieoffs(v, io);
            let sb = sideband_port_conns(io);
            v.line("top u_top (.clk(clk), .rst(rst), .start(start_btn), .done(done),");
            v.line(&format!(
                "           .in_addr(in_addr), .in_we(in_we), .in_din(in_din), .pred(pred){sb});"
            ));
            v.line("assign done_led = done;");
            v.line("assign pred_nibble = pred[3:0];");
        },
    );
    v.into_string()
}

fn emit_xilinx7_shell(model_name: &str, io: &IoConfig) -> String {
    let mut v = V::new();
    v.banner(&format!(
        "board_top — Xilinx 7-series shell over soft `top` ({model_name})"
    ));
    v.module(
        "board_top",
        &[],
        &[
            "input  logic clk".into(),
            "input  logic rst".into(),
            "input  logic start".into(),
            "output logic done".into(),
            "output logic [7:0] pred".into(),
        ],
        |v| {
            v.line("logic [9:0] in_addr; logic in_we; logic signed [7:0] in_din;");
            v.line("assign in_addr = '0; assign in_we = 1'b0; assign in_din = '0;");
            v.line("logic signed [7:0] pred_s;");
            emit_sideband_tieoffs(v, io);
            let sb = sideband_port_conns(io);
            v.line("top u_top (.clk(clk), .rst(rst), .start(start), .done(done),");
            v.line(&format!(
                "           .in_addr(in_addr), .in_we(in_we), .in_din(in_din), .pred(pred_s){sb});"
            ));
            v.line("assign pred = pred_s;");
        },
    );
    v.into_string()
}
