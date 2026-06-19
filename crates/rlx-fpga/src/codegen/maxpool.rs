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

//! INT8 max-pool kernel, NHWC, no padding (matches the only case
//! TinyConv-MNIST and `rlx-cortexm::maxpool` actually use).
//!
//! Loop nest: `oh, ow, c, kh, kw`. For each output element we read
//! `KH·KW` input values and keep the running max. With 1-cycle BRAM
//! read latency we use a small {READ, WAIT, UPDATE} cycle per input
//! sample, then a WRITE state at the end of each window.

use super::{Artifact, LayerArtifacts};
use crate::codegen::relu::bits_for;
use crate::model::Layer;
use crate::verilog::V;

pub fn emit(layer: &Layer) -> LayerArtifacts {
    let (name, h_in, w_in, c, kh, kw, sh, sw) = match layer {
        Layer::MaxPool2d {
            name,
            h_in,
            w_in,
            c,
            kh,
            kw,
            stride_h,
            stride_w,
        } => (*name, *h_in, *w_in, *c, *kh, *kw, *stride_h, *stride_w),
        _ => unreachable!("maxpool::emit called with non-MaxPool2d layer"),
    };
    let h_out = (h_in - kh) / sh + 1;
    let w_out = (w_in - kw) / sw + 1;
    let in_len = h_in * w_in * c;
    let out_len = h_out * w_out * c;

    let module_name = format!("{name}_kernel");
    let instance_name = format!("u_{name}");

    let in_addr_bits = bits_for(in_len);
    let out_addr_bits = bits_for(out_len);

    let mut v = V::new();
    v.banner(&format!(
        "{module_name} — INT8 maxpool {kh}x{kw} stride {sh}x{sw} on [{h_in}x{w_in}x{c}] → [{h_out}x{w_out}x{c}]"
    ));
    v.blank();

    v.module(
        &module_name,
        &[],
        &[
            "input  logic                       clk".into(),
            "input  logic                       rst".into(),
            "input  logic                       start".into(),
            "output logic                       done".into(),
            format!("output logic [{}:0]              x_addr", in_addr_bits - 1),
            "input  logic signed [7:0]          x_dout".into(),
            format!("output logic [{}:0]              y_addr", out_addr_bits - 1),
            "output logic                       y_we".into(),
            "output logic signed [7:0]          y_din".into(),
        ],
        |v| {
            v.line(&format!("localparam int H_IN={h_in}, W_IN={w_in}, C={c};"));
            v.line(&format!("localparam int H_OUT={h_out}, W_OUT={w_out};"));
            v.line(&format!("localparam int KH={kh}, KW={kw}, SH={sh}, SW={sw};"));
            v.blank();

            v.line("typedef enum logic [2:0] {");
            v.block(|v| v.line("S_IDLE, S_READ, S_WAIT, S_UPDATE, S_WRITE, S_DONE"));
            v.line("} state_t;");
            v.line("state_t state, next;");
            v.blank();
            v.line(&format!("logic [{}:0] oh, ow, oc;", bits_for(h_out.max(w_out).max(c)) - 1));
            v.line(&format!("logic [{}:0] kh_i, kw_i;",  bits_for(kh.max(kw)) - 1));
            v.line("logic signed [7:0] best;");
            v.blank();

            v.comment("Input address = ((oh*SH + kh_i) * W_IN + (ow*SW + kw_i)) * C + oc");
            v.comment("Output addr   = (oh * W_OUT + ow) * C + oc");
            v.line(&format!("logic [{}:0] in_idx;",  in_addr_bits - 1));
            v.line(&format!("logic [{}:0] out_idx;", out_addr_bits - 1));
            v.always_comb(|v| {
                v.line("in_idx  = ((oh*SH + kh_i) * W_IN + (ow*SW + kw_i)) * C + oc;");
                v.line("out_idx = (oh * W_OUT + ow) * C + oc;");
            });
            v.blank();

            v.always_ff(|v| {
                v.line("if (rst) begin");
                v.block(|v| {
                    v.line("state <= S_IDLE;");
                    v.line("oh    <= '0;  ow <= '0;  oc <= '0;");
                    v.line("kh_i  <= '0;  kw_i <= '0;");
                    v.line("best  <= 8'sh80;");
                });
                v.line("end else begin");
                v.block(|v| {
                    v.line("state <= next;");
                    v.line("if (state == S_IDLE && start) begin");
                    v.block(|v| {
                        v.line("oh <= '0; ow <= '0; oc <= '0;");
                        v.line("kh_i <= '0; kw_i <= '0;");
                        v.line("best <= 8'sh80;");
                    });
                    v.line("end");
                    v.line("if (state == S_UPDATE) begin");
                    v.block(|v| {
                        v.line("if (x_dout > best) best <= x_dout;");
                        v.comment("advance kw_i, kh_i — the inner pool window");
                        v.line("if (kw_i == KW - 1) begin");
                        v.block(|v| {
                            v.line("kw_i <= '0;");
                            v.line("if (kh_i == KH - 1) begin");
                            v.block(|v| v.line("kh_i <= '0;  // window done; S_WRITE next"));
                            v.line("end else kh_i <= kh_i + 1;");
                        });
                        v.line("end else kw_i <= kw_i + 1;");
                    });
                    v.line("end");
                    v.line("if (state == S_WRITE) begin");
                    v.block(|v| {
                        v.line("best <= 8'sh80;");
                        v.comment("advance oc, ow, oh");
                        v.line("if (oc == C - 1) begin");
                        v.block(|v| {
                            v.line("oc <= '0;");
                            v.line("if (ow == W_OUT - 1) begin");
                            v.block(|v| {
                                v.line("ow <= '0;");
                                v.line("oh <= oh + 1;  // overflow caught by FSM transition");
                            });
                            v.line("end else ow <= ow + 1;");
                        });
                        v.line("end else oc <= oc + 1;");
                    });
                    v.line("end");
                });
                v.line("end");
            });
            v.blank();

            v.always_comb(|v| {
                v.line("next   = state;");
                v.line("x_addr = in_idx;");
                v.line("y_addr = out_idx;");
                v.line("y_we   = 1'b0;");
                v.line("y_din  = best;");
                v.line("done   = 1'b0;");
                v.line("unique case (state)");
                v.block(|v| {
                    v.line("S_IDLE   : if (start) next = S_READ;");
                    v.line("S_READ   : next = S_WAIT;");
                    v.line("S_WAIT   : next = S_UPDATE;");
                    v.line("S_UPDATE : next = (kh_i == KH - 1 && kw_i == KW - 1) ? S_WRITE : S_READ;");
                    v.line("S_WRITE  : begin");
                    v.block(|v| {
                        v.line("y_we = 1'b1;");
                        v.line("next = (oh == H_OUT - 1 && ow == W_OUT - 1 && oc == C - 1) ? S_DONE : S_READ;");
                    });
                    v.line("end");
                    v.line("S_DONE   : begin done = 1'b1; if (!start) next = S_IDLE; end");
                });
                v.line("endcase");
            });
        },
    );

    LayerArtifacts {
        module_name,
        instance_name,
        out_len,
        sv: Artifact {
            rel_path: format!("layers/{name}.sv"),
            content: v.into_string(),
        },
        mems: Vec::new(),
    }
}
