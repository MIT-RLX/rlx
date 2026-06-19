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

//! INT8 fully-connected kernel with per-row Q0.31 / Q0.15 requantize.
//!
//! Loop nest (matches `rlx-cortexm::dense::dense_i8` line-for-line).
//! Honors the same `Hints` knobs as `conv2d`: `fast_mac`,
//! `ternary_fast_path`, `shared_requant`, `requant_precision`.

use super::{Artifact, LayerArtifacts};
use crate::codegen::conv2d::emit_byte_addr_lane;
use crate::codegen::relu::bits_for;
use crate::model::Layer;
use crate::pack::{packed_byte_len, weights_per_byte};
use crate::passes::Hints;
use crate::quant::q31_to_q15;
use crate::tune::RequantPrecision;
use crate::verilog::{V, mem_hex_bytes, mem_hex_words_i32};

pub fn emit(layer: &Layer, hints: &Hints) -> LayerArtifacts {
    let (name, in_f, out_f, x_zp, w_zp, out_zp, weight_bits, requant, weights, bias) = match layer {
        Layer::Dense {
            name,
            in_features,
            out_features,
            x_zp,
            w_zp,
            out_zp,
            weight_bits,
            requant,
            weights,
            bias,
        } => (
            *name,
            *in_features,
            *out_features,
            *x_zp,
            *w_zp,
            *out_zp,
            *weight_bits,
            requant.clone(),
            weights.clone(),
            bias.clone(),
        ),
        _ => unreachable!("dense::emit called with non-Dense layer"),
    };
    assert!(
        matches!(weight_bits, 2 | 4 | 8),
        "dense: weight_bits must be 2, 4 or 8 (got {weight_bits})"
    );
    assert!(
        !(hints.ternary_fast_path && weight_bits != 2),
        "ternary_fast_path requires weight_bits == 2"
    );
    let w_logical_len = in_f * out_f;
    let w_byte_len = packed_byte_len(w_logical_len, weight_bits);

    let module_name = format!("{name}_kernel");
    let instance_name = format!("u_{name}");

    let q15 = matches!(hints.requant_precision, RequantPrecision::Q0_15);
    let m0_width: u32 = if q15 { 16 } else { 32 };

    let mut mems = Vec::new();
    mems.push(Artifact {
        rel_path: format!("weights/{name}_w.mem"),
        content: mem_hex_bytes(&weights),
    });
    if let Some(b) = &bias {
        mems.push(Artifact {
            rel_path: format!("weights/{name}_b.mem"),
            content: mem_hex_words_i32(b, 32),
        });
    }
    if hints.shared_requant.is_none() {
        let m0_words: Vec<i32> = if q15 {
            requant
                .iter()
                .map(|(m, s)| q31_to_q15(*m, *s).0 as i32)
                .collect()
        } else {
            requant.iter().map(|(m, _)| *m).collect()
        };
        let shift_words: Vec<i32> = if q15 {
            requant.iter().map(|(m, s)| q31_to_q15(*m, *s).1).collect()
        } else {
            requant.iter().map(|(_, s)| *s).collect()
        };
        mems.push(Artifact {
            rel_path: format!("weights/{name}_m0.mem"),
            content: mem_hex_words_i32(&m0_words, m0_width),
        });
        mems.push(Artifact {
            rel_path: format!("weights/{name}_sh.mem"),
            content: mem_hex_words_i32(&shift_words, 8),
        });
    }

    let in_addr_bits = bits_for(in_f);
    let out_addr_bits = bits_for(out_f);
    let w_byte_addr_bits = bits_for(w_byte_len);
    let w_log_bits = bits_for(w_logical_len);
    let r_addr_bits = bits_for(out_f);
    let per_byte = weights_per_byte(weight_bits);
    let log2_per_byte = (per_byte.trailing_zeros()) as usize;

    let mut v = V::new();
    v.banner(&format!(
        "{module_name} — INT8 dense {in_f} → {out_f} (w{weight_bits}), x_zp={x_zp} w_zp={w_zp} out_zp={out_zp}"
    ));
    let tags: Vec<&'static str> = [
        ("fast_mac", hints.fast_mac),
        ("ternary_fast_path", hints.ternary_fast_path),
        ("shared_requant", hints.shared_requant.is_some()),
        ("Q0_15", q15),
    ]
    .iter()
    .filter(|(_, on)| *on)
    .map(|(t, _)| *t)
    .collect();
    if !tags.is_empty() {
        v.comment(&format!("hints: {}", tags.join(" + ")));
    }
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
            v.line(&format!("localparam int IN_F={in_f}, OUT_F={out_f};"));
            v.line(&format!("localparam int X_ZP={x_zp}, W_ZP={w_zp}, OUT_ZP={out_zp};"));
            v.line(&format!("localparam int W_BITS={weight_bits};"));
            v.line(&format!("localparam int W_LOG_LEN={w_logical_len};"));
            v.line(&format!("localparam int W_BYTE_LEN={w_byte_len};"));
            v.blank();

            v.comment("Weight ROM (byte-addressed)");
            v.line(&format!("logic [{}:0] w_addr;", w_byte_addr_bits - 1));
            v.line("logic        [7:0] w_byte;");
            v.line(&format!("block_rom #(.WIDTH(8), .DEPTH(W_BYTE_LEN), .INIT_FILE(\"weights/{name}_w.mem\"))"));
            v.line("u_w_rom (.clk(clk), .addr(w_addr), .dout(w_byte));");
            v.blank();

            v.line("logic [1:0] w_lane_q;");
            if hints.ternary_fast_path {
                v.comment("Ternary fast path — direct crumb extraction (no DSP multiplier)");
                v.line("logic [1:0] w_crumb;");
                v.always_comb(|v| {
                    v.line("unique case (w_lane_q)");
                    v.block(|v| {
                        v.line("2'd0: w_crumb = w_byte[1:0];");
                        v.line("2'd1: w_crumb = w_byte[3:2];");
                        v.line("2'd2: w_crumb = w_byte[5:4];");
                        v.line("2'd3: w_crumb = w_byte[7:6];");
                    });
                    v.line("endcase");
                });
            } else {
                v.comment("Combinational weight unpack");
                v.line("logic signed [31:0] w_val;");
                v.line("weight_unpack #(.BITS(W_BITS)) u_w_unpack (");
                v.block(|v| {
                    v.line(".byte_in(w_byte),");
                    v.line(".lane(w_lane_q),");
                    v.line(".w_out(w_val)");
                });
                v.line(");");
            }
            v.blank();

            v.comment("Bias ROM (i32)");
            v.line(&format!("logic [{}:0] b_addr;", r_addr_bits - 1));
            v.line("logic signed [31:0] b_dout;");
            if bias.is_some() {
                v.line(&format!("block_rom #(.WIDTH(32), .DEPTH({out_f}), .INIT_FILE(\"weights/{name}_b.mem\"))"));
                v.line("u_b_rom (.clk(clk), .addr(b_addr), .dout(b_dout));");
            } else {
                v.line("assign b_dout = 32'sd0;");
            }
            v.blank();

            // Requant table source: ROM or localparam
            if let Some((m0_q31, shift_v)) = hints.shared_requant {
                v.comment("Requant — shared (uniform across rows)");
                let (m0_lit, sh_lit): (i64, i32) = if q15 {
                    let (m0, s) = q31_to_q15(m0_q31, shift_v);
                    (m0 as i64, s)
                } else {
                    (m0_q31 as i64, shift_v)
                };
                v.line(&format!("localparam logic signed [{}:0] M0_VAL = {}'sd{};",
                                m0_width - 1, m0_width, m0_lit));
                v.line(&format!("localparam logic        [7:0]  SHIFT_VAL = 8'd{sh_lit};"));
                v.line(&format!("logic signed [{}:0] m0_q;", m0_width - 1));
                v.line("logic        [7:0]  sh_q;");
                v.line("assign m0_q = M0_VAL;");
                v.line("assign sh_q = SHIFT_VAL;");
            } else {
                v.comment("Requant ROMs (per-row)");
                v.line(&format!("logic [{}:0] r_addr;", r_addr_bits - 1));
                v.line(&format!("logic signed [{}:0] m0_q;", m0_width - 1));
                v.line("logic        [7:0]  sh_q;");
                v.line(&format!("block_rom #(.WIDTH({m0_width}), .DEPTH({out_f}), .INIT_FILE(\"weights/{name}_m0.mem\"))"));
                v.line("u_m0_rom (.clk(clk), .addr(r_addr), .dout(m0_q));");
                v.line(&format!("block_rom #(.WIDTH(8),  .DEPTH({out_f}), .INIT_FILE(\"weights/{name}_sh.mem\"))"));
                v.line("u_sh_rom (.clk(clk), .addr(r_addr), .dout(sh_q));");
                v.line("always_comb r_addr = m_i;");
            }
            v.blank();

            v.comment("Combinational requantize");
            v.line("logic signed [7:0] q_out;");
            if q15 {
                v.line("requant_q15 u_requant (");
                v.block(|v| {
                    v.line(".acc(acc),");
                    v.line(".m0(m0_q[15:0]),");
                    v.line(".shift(sh_q[3:0]),");
                    v.line(".out_zp(OUT_ZP),");
                    v.line(".q(q_out)");
                });
                v.line(");");
            } else {
                v.line("requant_q31 u_requant (");
                v.block(|v| {
                    v.line(".acc(acc),");
                    v.line(".m0(m0_q),");
                    v.line(".shift(sh_q[4:0]),");
                    v.line(".out_zp(OUT_ZP),");
                    v.line(".q(q_out)");
                });
                v.line(");");
            }
            v.blank();

            v.line(&format!("logic [{}:0] m_i;", out_addr_bits - 1));
            v.line(&format!("logic [{}:0] k_i;", in_addr_bits - 1));
            v.line("logic signed [31:0] acc;");
            v.blank();

            v.comment("Address derivation");
            v.line(&format!("logic [{}:0] w_idx_logical;", w_log_bits - 1));
            v.always_comb(|v| {
                v.line("w_idx_logical = m_i * IN_F + k_i;");
                emit_byte_addr_lane(v, "w_addr", "w_lane_q", "w_idx_logical", w_log_bits, log2_per_byte);
            });
            v.blank();

            // MAC delta — same logic as conv2d (kept inline here to avoid
            // a public helper traffic across modules).
            v.comment("Per-cycle MAC delta");
            v.line("logic signed [31:0] xv_corrected;");
            if hints.fast_mac {
                v.line("assign xv_corrected = $signed({{24{x_dout[7]}}, x_dout});");
            } else {
                v.line("assign xv_corrected = $signed({{24{x_dout[7]}}, x_dout}) - X_ZP;");
            }
            v.line("logic signed [31:0] mac_delta;");
            if hints.ternary_fast_path {
                v.always_comb(|v| {
                    v.line("unique case (w_crumb)");
                    v.block(|v| {
                        v.line("2'b00: mac_delta =  32'sd0;");
                        v.line("2'b01: mac_delta =  xv_corrected;");
                        v.line("2'b10: mac_delta = -(xv_corrected <<< 1);");
                        v.line("2'b11: mac_delta = -xv_corrected;");
                    });
                    v.line("endcase");
                });
            } else if hints.fast_mac && w_zp == 0 {
                v.line("assign mac_delta = xv_corrected * w_val;");
            } else {
                v.line("assign mac_delta = xv_corrected * (w_val - W_ZP);");
            }
            v.blank();

            v.comment("Pipelined FSM — 1 MAC/cycle in S_PIPE");
            v.line("typedef enum logic [2:0] {");
            v.block(|v| v.line(
                "S_IDLE, S_LOAD_BIAS, S_BIAS_WAIT, S_INIT_ACC, \
                 S_PIPE, S_WRITE, S_DONE"
            ));
            v.line("} state_t;");
            v.line("state_t state, next;");
            v.line("logic prev_valid;");
            v.line("logic done_issuing;");
            v.blank();

            v.always_ff(|v| {
                v.line("if (rst) begin");
                v.block(|v| {
                    v.line("state <= S_IDLE;");
                    v.line("m_i<='0; k_i<='0; acc<='0;");
                    v.line("prev_valid <= 1'b0;");
                    v.line("done_issuing <= 1'b0;");
                });
                v.line("end else begin");
                v.block(|v| {
                    v.line("state <= next;");
                    v.line("if (state == S_IDLE && start) begin m_i<='0; end");
                    v.line("if (state == S_INIT_ACC) begin");
                    v.block(|v| {
                        v.line("acc <= b_dout;");
                        v.line("k_i <= '0;");
                        v.line("prev_valid <= 1'b0;");
                        v.line("done_issuing <= 1'b0;");
                    });
                    v.line("end");
                    v.line("if (state == S_PIPE) begin");
                    v.block(|v| {
                        v.line("if (prev_valid) acc <= acc + mac_delta;");
                        v.line("if (!done_issuing) begin");
                        v.block(|v| {
                            v.line("if (k_i == IN_F - 1) begin");
                            v.block(|v| {
                                v.line("k_i <= '0;");
                                v.line("done_issuing <= 1'b1;");
                            });
                            v.line("end else k_i <= k_i + 1;");
                            v.line("prev_valid <= 1'b1;");
                        });
                        v.line("end else begin");
                        v.block(|v| v.line("prev_valid <= 1'b0;"));
                        v.line("end");
                    });
                    v.line("end");
                    v.line("if (state == S_WRITE) m_i <= m_i + 1;");
                });
                v.line("end");
            });
            v.blank();

            v.always_comb(|v| {
                v.line("next   = state;");
                v.line("x_addr = k_i;");
                v.line("b_addr = m_i;");
                v.line("y_addr = m_i;");
                v.line("y_we   = 1'b0;");
                v.line("y_din  = q_out;");
                v.line("done   = 1'b0;");
                v.line("unique case (state)");
                v.block(|v| {
                    v.line("S_IDLE      : if (start) next = S_LOAD_BIAS;");
                    v.line("S_LOAD_BIAS : next = S_BIAS_WAIT;");
                    v.line("S_BIAS_WAIT : next = S_INIT_ACC;");
                    v.line("S_INIT_ACC  : next = S_PIPE;");
                    v.line("S_PIPE      : next = (done_issuing && prev_valid) ? S_WRITE : S_PIPE;");
                    v.line("S_WRITE     : begin");
                    v.block(|v| {
                        v.line("y_we = 1'b1;");
                        v.line("next = (m_i == OUT_F - 1) ? S_DONE : S_LOAD_BIAS;");
                    });
                    v.line("end");
                    v.line("S_DONE      : begin done = 1'b1; if (!start) next = S_IDLE; end");
                });
                v.line("endcase");
            });
        },
    );

    LayerArtifacts {
        module_name,
        instance_name,
        out_len: out_f,
        sv: Artifact {
            rel_path: format!("layers/{name}.sv"),
            content: v.into_string(),
        },
        mems,
    }
}
