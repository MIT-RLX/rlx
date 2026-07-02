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

//! Argmax kernel: scan an i8 buffer, return the index of the largest
//! element. Output buffer length is 1 (the index, saturating-cast to i8
//! — works for ≤127 classes).

use super::{Artifact, LayerArtifacts};
use crate::model::Layer;
use crate::verilog::V;

pub fn emit(layer: &Layer) -> LayerArtifacts {
    let (name, len) = match layer {
        Layer::Argmax { name, len } => (*name, *len),
        _ => unreachable!("argmax::emit called with non-Argmax layer"),
    };
    let module_name = format!("{name}_kernel");
    let instance_name = format!("u_{name}");

    let mut v = V::new();
    v.banner(&format!(
        "{module_name} — argmax over LEN={len} i8 elements"
    ));
    v.blank();

    let addr_bits = super::relu::bits_for(len);
    let idx_bits = super::relu::bits_for(len);

    v.module(
        &module_name,
        &[],
        &[
            "input  logic                       clk".into(),
            "input  logic                       rst".into(),
            "input  logic                       start".into(),
            "output logic                       done".into(),
            format!("output logic [{}:0]              x_addr", addr_bits - 1),
            "input  logic signed [7:0]          x_dout".into(),
            "output logic [0:0]                 y_addr".into(),
            "output logic                       y_we".into(),
            "output logic signed [7:0]          y_din".into(),
        ],
        |v| {
            v.line(&format!("localparam int LEN = {len};"));
            v.blank();

            v.line("typedef enum logic [2:0] {");
            v.block(|v| v.line("S_IDLE, S_READ, S_WAIT, S_UPDATE, S_WRITE, S_DONE"));
            v.line("} state_t;");
            v.line("state_t state, next;");
            v.line(&format!("logic [{}:0] i;", addr_bits - 1));
            v.line(&format!("logic [{}:0] best_idx;", idx_bits - 1));
            v.line("logic signed [7:0] best_val;");
            v.blank();

            v.always_ff(|v| {
                v.line("if (rst) begin");
                v.block(|v| {
                    v.line("state    <= S_IDLE;");
                    v.line("i        <= '0;");
                    v.line("best_idx <= '0;");
                    v.line("best_val <= 8'sh80;");
                });
                v.line("end else begin");
                v.block(|v| {
                    v.line("state <= next;");
                    v.line("if (state == S_IDLE && start) begin");
                    v.block(|v| {
                        v.line("i        <= '0;");
                        v.line("best_idx <= '0;");
                        v.line("best_val <= 8'sh80;  // i8::MIN");
                    });
                    v.line("end");
                    v.line("if (state == S_UPDATE) begin");
                    v.block(|v| {
                        v.line("if (x_dout > best_val) begin");
                        v.block(|v| {
                            v.line("best_val <= x_dout;");
                            v.line("best_idx <= i;");
                        });
                        v.line("end");
                        v.line("i <= i + 1;");
                    });
                    v.line("end");
                });
                v.line("end");
            });
            v.blank();

            v.always_comb(|v| {
                v.line("next   = state;");
                v.line("x_addr = i;");
                v.line("y_addr = 1'b0;");
                v.line("y_we   = 1'b0;");
                v.line("y_din  = best_idx[7:0];");
                v.line("done   = 1'b0;");
                v.line("unique case (state)");
                v.block(|v| {
                    v.line("S_IDLE   : if (start) next = S_READ;");
                    v.line("S_READ   : next = S_WAIT;");
                    v.line("S_WAIT   : next = S_UPDATE;");
                    v.line("S_UPDATE : next = (i == LEN - 1) ? S_WRITE : S_READ;");
                    v.line("S_WRITE  : begin y_we = 1'b1; next = S_DONE; end");
                    v.line("S_DONE   : begin done = 1'b1; if (!start) next = S_IDLE; end");
                });
                v.line("endcase");
            });
        },
    );

    LayerArtifacts {
        module_name,
        instance_name,
        out_len: 1,
        sv: Artifact {
            rel_path: format!("layers/{name}.sv"),
            content: v.into_string(),
        },
        mems: Vec::new(),
    }
}
