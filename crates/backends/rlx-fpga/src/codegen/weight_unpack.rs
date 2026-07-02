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
//! Mirrors `rlx_cortexm::quant::read_weight` for `BITS ∈ {2, 4, 8}`. For
//! ternary (`BITS = 2`), the trainer emits values in `{-1, 0, 1}`; the
//! unpack still sign-extends the unused `-2` codepoint correctly so
//! malformed weight files don't silently produce wrong arithmetic.

use crate::verilog::V;

pub fn emit() -> String {
    let mut v = V::new();
    v.banner("weight_unpack — extract one logical weight from a packed byte");
    v.comment("BITS ∈ {2, 4, 8}.  `lane` selects the sub-byte slot:");
    v.comment("  BITS=2  →  lane ∈ {0,1,2,3}  (4 crumbs per byte, LSB first)");
    v.comment("  BITS=4  →  lane ∈ {0,1}      (low nibble, then high nibble)");
    v.comment("  BITS=8  →  lane is ignored");
    v.comment("Output is sign-extended to 32 bits to feed the i32 MAC path.");
    v.blank();

    v.module(
        "weight_unpack",
        &["parameter int BITS = 8".into()],
        &[
            "input  logic        [7:0]  byte_in".into(),
            "input  logic        [1:0]  lane".into(),
            "output logic signed [31:0] w_out".into(),
        ],
        |v| {
            v.line("generate");
            v.block(|v| {
                v.line("if (BITS == 8) begin : g_b8");
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
                v.block(|v| {
                    v.line(
                        "// $error must be inside an initial; for now compile-time fallback to 0",
                    )
                });
                v.block(|v| v.line("assign w_out = 32'sd0;"));
                v.line("end");
            });
            v.line("endgenerate");
        },
    );
    v.into_string()
}
