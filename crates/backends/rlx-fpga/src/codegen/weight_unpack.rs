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

//! Combinational weight unpack: byte + lane → 32-bit signed value.
//!
//! * `ENCODING=0` (default): two's-complement INT2/4/8 (cortexm layout).
//! * `ENCODING=1` + `BITS=4`: OCP FP4 E2M1 nibble → fixed-point via LUT
//!   (`round(decode(code) * 2)` so 0.5→1); fold the ×2 into requant scales.

use rlx_ir::ScaledFormat;

use crate::verilog::V;

pub fn emit() -> String {
    let mut v = V::new();
    v.banner("weight_unpack — extract one logical weight from a packed byte");
    v.comment("BITS ∈ {2, 4, 8}.  ENCODING: 0=signed-int, 1=FP4-E2M1 (BITS=4).");
    v.comment("  BITS=2  →  lane ∈ {0,1,2,3}  (4 crumbs per byte, LSB first)");
    v.comment("  BITS=4  →  lane ∈ {0,1}      (low nibble, then high nibble)");
    v.comment("  BITS=8  →  lane is ignored");
    v.blank();

    let fp4_lut = fp4_e2m1_lut_lines();

    v.module(
        "weight_unpack",
        &[
            "parameter int BITS = 8".into(),
            "parameter int ENCODING = 0".into(),
        ],
        &[
            "input  logic        [7:0]  byte_in".into(),
            "input  logic        [1:0]  lane".into(),
            "output logic signed [31:0] w_out".into(),
        ],
        |v| {
            v.line("generate");
            v.block(|v| {
                v.line("if (ENCODING == 1 && BITS == 4) begin : g_fp4");
                v.block(|v| {
                    v.line("logic [3:0] nib;");
                    v.always_comb(|v| {
                        v.line("if (lane[0]) nib = byte_in[7:4];");
                        v.line("else         nib = byte_in[3:0];");
                    });
                    v.comment("F4E2M1 decode × 2 → signed int for the MAC");
                    v.always_comb(|v| {
                        v.line("unique case (nib)");
                        v.block(|v| {
                            for line in &fp4_lut {
                                v.line(line);
                            }
                            v.line("default: w_out = 32'sd0;");
                        });
                        v.line("endcase");
                    });
                });
                v.line("end else if (BITS == 8) begin : g_b8");
                v.block(|v| v.line("assign w_out = $signed({{24{byte_in[7]}}, byte_in});"));
                v.line("end else if (BITS == 4) begin : g_b4");
                v.block(|v| {
                    v.line("logic [3:0] nib;");
                    v.always_comb(|v| {
                        v.line("if (lane[0]) nib = byte_in[7:4];");
                        v.line("else         nib = byte_in[3:0];");
                    });
                    v.line("assign w_out = $signed({{28{nib[3]}}, nib});");
                });
                v.line("end else if (BITS == 2) begin : g_b2");
                v.block(|v| {
                    v.line("logic [1:0] crumb;");
                    v.always_comb(|v| {
                        v.line("unique case (lane)");
                        v.block(|v| {
                            v.line("2'd0: crumb = byte_in[1:0];");
                            v.line("2'd1: crumb = byte_in[3:2];");
                            v.line("2'd2: crumb = byte_in[5:4];");
                            v.line("2'd3: crumb = byte_in[7:6];");
                        });
                        v.line("endcase");
                    });
                    v.line("assign w_out = $signed({{30{crumb[1]}}, crumb});");
                });
                v.line("end else begin : g_bad");
                v.block(|v| v.line("assign w_out = 32'sd0;"));
                v.line("end");
            });
            v.line("endgenerate");
        },
    );
    v.into_string()
}

fn fp4_e2m1_lut_lines() -> Vec<String> {
    let fmt = ScaledFormat::F4E2M1;
    (0u8..16)
        .map(|code| {
            let f = fmt.decode(code);
            let q = if f.is_finite() {
                (f * 2.0).round() as i32
            } else {
                0
            };
            format!("4'h{code:X}: w_out = 32'sd{q};")
        })
        .collect()
}
