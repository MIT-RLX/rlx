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

//! INT8 conv2d kernel, NHWC, per-output-channel Q0.31 / Q0.15 requant.
//!
//! Loop nest (matches `rlx-cortexm::conv2d::conv2d_i8` line-for-line):
//!
//! ```text
//!   for oh, ow, oc:
//!     acc = bias[oc]
//!     for kh, kw, ic:
//!       acc += (x[ih,iw,ic] - x_zp) * (w[oc,kh,kw,ic] - w_zp)
//!     out[oh,ow,oc] = requant(acc, m0[oc], shift[oc], out_zp)
//! ```
//!
//! Tunable codegen (driven by [`Hints`]):
//!
//! * `fast_mac` — drop the `- X_ZP` / `- W_ZP` subs (saves ~32 LUT/MAC)
//! * `ternary_fast_path` — replace the multiplier with a 4-way mux on
//!   the weight crumb (`{-1,0,1,-2}` → `{-x, 0, +x, -2x}`); drops the
//!   DSP slice for the MAC
//! * `shared_requant` — when every per-channel `(M0, shift)` is equal,
//!   replace the M0 / shift ROMs with `localparam`s (saves 2 BRAMs)
//! * `requant_precision` — Q0.31 (default, bit-exact) or Q0.15 (smaller)

use super::{Artifact, LayerArtifacts};
use crate::codegen::relu::bits_for;
use crate::model::Layer;
use crate::pack::{packed_byte_len, weights_per_byte};
use crate::passes::Hints;
use crate::quant::q31_to_q15;
use crate::tune::RequantPrecision;
use crate::verilog::{V, mem_hex_bytes, mem_hex_words_i32};

pub fn emit(layer: &Layer, hints: &Hints) -> LayerArtifacts {
    let (
        name,
        h_in,
        w_in,
        c_in,
        c_out,
        kh,
        kw,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        x_zp,
        w_zp,
        out_zp,
        weight_bits,
        requant,
        weights,
        bias,
    ) = match layer {
        Layer::Conv2d {
            name,
            h_in,
            w_in,
            c_in,
            c_out,
            kh,
            kw,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            x_zp,
            w_zp,
            out_zp,
            weight_bits,
            requant,
            weights,
            bias,
        } => (
            *name,
            *h_in,
            *w_in,
            *c_in,
            *c_out,
            *kh,
            *kw,
            *pad_h,
            *pad_w,
            *stride_h,
            *stride_w,
            *x_zp,
            *w_zp,
            *out_zp,
            *weight_bits,
            requant.clone(),
            weights.clone(),
            bias.clone(),
        ),
        _ => unreachable!("conv2d::emit called with non-Conv2d layer"),
    };
    assert!(
        matches!(weight_bits, 2 | 4 | 8),
        "conv2d: weight_bits must be 2, 4 or 8 (got {weight_bits})"
    );
    assert_eq!(pad_h, 0, "conv2d: only pad_h=0 supported in first cut");
    assert_eq!(pad_w, 0, "conv2d: only pad_w=0 supported in first cut");
    assert!(
        !(hints.ternary_fast_path && weight_bits != 2),
        "ternary_fast_path requires weight_bits == 2"
    );

    let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
    let in_len = h_in * w_in * c_in;
    let out_len = h_out * w_out * c_out;
    let w_logical_len = c_out * kh * kw * c_in;
    let w_byte_len = packed_byte_len(w_logical_len, weight_bits);

    let module_name = format!("{name}_kernel");
    let instance_name = format!("u_{name}");

    // ── weight + bias + requant .mem files ──────────────────────────
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
    let q15 = matches!(hints.requant_precision, RequantPrecision::Q0_15);
    let m0_width: u32 = if q15 { 16 } else { 32 };
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

    // ── module body ─────────────────────────────────────────────────
    let in_addr_bits = bits_for(in_len);
    let out_addr_bits = bits_for(out_len);
    let w_byte_addr_bits = bits_for(w_byte_len);
    let w_log_bits = bits_for(w_logical_len);
    let b_addr_bits = bits_for(c_out);
    let per_byte = weights_per_byte(weight_bits);
    let log2_per_byte = (per_byte.trailing_zeros()) as usize;

    let mut v = V::new();
    v.banner(&format!(
        "{module_name} — INT8 conv2d {kh}x{kw} stride {stride_h}x{stride_w} on [{h_in}x{w_in}x{c_in}] → [{h_out}x{w_out}x{c_out}] (w{weight_bits})"
    ));
    v.comment(&format!(
        "x_zp={x_zp} w_zp={w_zp} out_zp={out_zp}; weights={w_byte_len} bytes ({w_logical_len} logical, {weight_bits}-bit); requant per-OC ({c_out} entries)."
    ));
    let tags: Vec<&'static str> = [
        ("fast_mac", hints.fast_mac),
        ("ternary_fast_path", hints.ternary_fast_path),
        ("shared_requant", hints.shared_requant.is_some()),
        ("Q0_15", q15),
        ("fuses_relu", hints.fuses_relu),
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
            v.line(&format!(
                "localparam int H_IN={h_in}, W_IN={w_in}, C_IN={c_in};"
            ));
            v.line(&format!(
                "localparam int H_OUT={h_out}, W_OUT={w_out}, C_OUT={c_out};"
            ));
            v.line(&format!(
                "localparam int KH={kh}, KW={kw}, SH={stride_h}, SW={stride_w};"
            ));
            v.line(&format!(
                "localparam int X_ZP={x_zp}, W_ZP={w_zp}, OUT_ZP={out_zp};"
            ));
            v.line(&format!("localparam int W_BITS={weight_bits};"));
            v.line(&format!("localparam int W_LOG_LEN={w_logical_len};"));
            v.line(&format!("localparam int W_BYTE_LEN={w_byte_len};"));
            v.blank();

            v.comment(&format!(
                "Weight ROM (byte-addressed; logical idx → byte idx via >>{log2_per_byte})"
            ));
            v.line(&format!("logic [{}:0] w_addr;", w_byte_addr_bits - 1));
            v.line("logic        [7:0] w_byte;");
            v.line("block_rom #(.WIDTH(8), .DEPTH(W_BYTE_LEN),");
            v.line(&format!("  .INIT_FILE(\"weights/{name}_w.mem\"))"));
            v.line("u_w_rom (.clk(clk), .addr(w_addr), .dout(w_byte));");
            v.blank();

            // Lane signal is needed by either the ternary fast path or
            // weight_unpack (or both, if we go that route in future).
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
            v.line(&format!("logic [{}:0] b_addr;", b_addr_bits - 1));
            v.line("logic signed [31:0] b_dout;");
            if bias.is_some() {
                v.line(&format!("block_rom #(.WIDTH(32), .DEPTH({c_out}),"));
                v.line(&format!("  .INIT_FILE(\"weights/{name}_b.mem\"))"));
                v.line("u_b_rom (.clk(clk), .addr(b_addr), .dout(b_dout));");
            } else {
                v.line("assign b_dout = 32'sd0;");
            }
            v.blank();

            // ── Requant table: ROM (per-channel) or localparam (shared) ──
            emit_requant_source(v, name, &requant, hints, c_out, b_addr_bits);
            v.blank();

            // ── Requant epilogue instantiation ──
            v.comment("Requantize epilogue (combinational)");
            v.line("logic signed [7:0] q_raw;");
            v.line("logic signed [7:0] q_out;");
            if q15 {
                v.line("requant_q15 u_requant (");
                v.block(|v| {
                    v.line(".acc(acc),");
                    v.line(".m0(m0_q[15:0]),");
                    v.line(".shift(sh_q[3:0]),");
                    v.line(".out_zp(OUT_ZP),");
                    v.line(".q(q_raw)");
                });
                v.line(");");
            } else {
                v.line("requant_q31 u_requant (");
                v.block(|v| {
                    v.line(".acc(acc),");
                    v.line(".m0(m0_q),");
                    v.line(".shift(sh_q[4:0]),");
                    v.line(".out_zp(OUT_ZP),");
                    v.line(".q(q_raw)");
                });
                v.line(");");
            }
            if hints.fuses_relu {
                v.comment("fuses_relu: clamp at OUT_ZP (= relu zero_point)");
                v.line("assign q_out = (q_raw < OUT_ZP[7:0]) ? OUT_ZP[7:0] : q_raw;");
            } else {
                v.line("assign q_out = q_raw;");
            }
            v.blank();

            v.comment("Counters");
            v.line(&format!(
                "logic [{}:0] oh, ow, oc;",
                bits_for(h_out.max(w_out).max(c_out)) - 1
            ));
            v.line(&format!(
                "logic [{}:0] kh_i, kw_i;",
                bits_for(kh.max(kw)) - 1
            ));
            v.line(&format!("logic [{}:0] ic;", bits_for(c_in) - 1));
            v.line("logic signed [31:0] acc;");
            v.blank();

            v.comment("Address derivation");
            v.line(&format!("logic [{}:0] in_idx;", in_addr_bits - 1));
            v.line(&format!("logic [{}:0] out_idx;", out_addr_bits - 1));
            v.line(&format!("logic [{}:0] w_idx_logical;", w_log_bits - 1));
            v.always_comb(|v| {
                v.line("in_idx        = ((oh*SH + kh_i) * W_IN + (ow*SW + kw_i)) * C_IN + ic;");
                v.line("out_idx       = (oh * W_OUT + ow) * C_OUT + oc;");
                v.line("w_idx_logical = ((oc * KH + kh_i) * KW + kw_i) * C_IN + ic;");
                emit_byte_addr_lane(
                    v,
                    "w_addr",
                    "w_lane_q",
                    "w_idx_logical",
                    w_log_bits,
                    log2_per_byte,
                );
            });
            v.blank();

            // ── Per-cycle MAC delta ──
            emit_mac_delta(v, hints, x_zp != 0, w_zp);
            v.blank();

            v.comment("Pipelined FSM — 1 MAC/cycle in S_PIPE");
            v.comment("  prev_valid    : last cycle issued a real read (data is valid this cycle)");
            v.comment(
                "  done_issuing  : all addresses driven; remaining cycles drain the pipeline",
            );
            v.line("typedef enum logic [2:0] {");
            v.block(|v| {
                v.line(
                    "S_IDLE, S_LOAD_BIAS, S_BIAS_WAIT, S_INIT_ACC, \
                 S_PIPE, S_WRITE, S_DONE",
                )
            });
            v.line("} state_t;");
            v.line("state_t state, next;");
            v.line("logic prev_valid;");
            v.line("logic done_issuing;");
            v.blank();

            v.always_ff(|v| {
                v.line("if (rst) begin");
                v.block(|v| {
                    v.line("state <= S_IDLE;");
                    v.line("oh<='0; ow<='0; oc<='0; kh_i<='0; kw_i<='0; ic<='0;");
                    v.line("acc<='0;");
                    v.line("prev_valid <= 1'b0;");
                    v.line("done_issuing <= 1'b0;");
                });
                v.line("end else begin");
                v.block(|v| {
                    v.line("state <= next;");
                    v.line("if (state == S_IDLE && start) begin");
                    v.block(|v| {
                        v.line("oh<='0; ow<='0; oc<='0;");
                    });
                    v.line("end");
                    v.line("if (state == S_INIT_ACC) begin");
                    v.block(|v| {
                        v.line("acc <= b_dout;");
                        v.line("kh_i <= '0; kw_i <= '0; ic <= '0;");
                        v.line("prev_valid <= 1'b0;");
                        v.line("done_issuing <= 1'b0;");
                    });
                    v.line("end");
                    v.line("if (state == S_PIPE) begin");
                    v.block(|v| {
                        v.comment("(1) Accumulate the previous cycle's MAC if its read was real.");
                        v.line("if (prev_valid) acc <= acc + mac_delta;");
                        v.comment("(2) If still issuing, advance counters and mark issued.");
                        v.line("if (!done_issuing) begin");
                        v.block(|v| {
                            v.line("if (ic == C_IN - 1) begin");
                            v.block(|v| {
                                v.line("ic <= '0;");
                                v.line("if (kw_i == KW - 1) begin");
                                v.block(|v| {
                                    v.line("kw_i <= '0;");
                                    v.line("if (kh_i == KH - 1) begin");
                                    v.block(|v| {
                                        v.line("kh_i <= '0;");
                                        v.line("done_issuing <= 1'b1;  // last address driven");
                                    });
                                    v.line("end else kh_i <= kh_i + 1;");
                                });
                                v.line("end else kw_i <= kw_i + 1;");
                            });
                            v.line("end else ic <= ic + 1;");
                            v.line("prev_valid <= 1'b1;");
                        });
                        v.line("end else begin");
                        v.block(|v| v.line("prev_valid <= 1'b0;"));
                        v.line("end");
                    });
                    v.line("end");
                    v.line("if (state == S_WRITE) begin");
                    v.block(|v| {
                        v.line("if (oc == C_OUT - 1) begin");
                        v.block(|v| {
                            v.line("oc <= '0;");
                            v.line("if (ow == W_OUT - 1) begin");
                            v.block(|v| {
                                v.line("ow <= '0;");
                                v.line("oh <= oh + 1;");
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
                v.line("next    = state;");
                v.line("x_addr  = in_idx;");
                v.line("b_addr  = oc;");
                v.line("y_addr  = out_idx;");
                v.line("y_we    = 1'b0;");
                v.line("y_din   = q_out;");
                v.line("done    = 1'b0;");
                v.line("unique case (state)");
                v.block(|v| {
                    v.line("S_IDLE      : if (start) next = S_LOAD_BIAS;");
                    v.line("S_LOAD_BIAS : next = S_BIAS_WAIT;");
                    v.line("S_BIAS_WAIT : next = S_INIT_ACC;");
                    v.line("S_INIT_ACC  : next = S_PIPE;");
                    v.comment("S_PIPE exits when the last MAC has been accumulated:");
                    v.comment("  done_issuing means all reads have been issued; prev_valid means");
                    v.comment("  the data from the very last read is being consumed THIS cycle.");
                    v.line("S_PIPE      : next = (done_issuing && prev_valid) ? S_WRITE : S_PIPE;");
                    v.line("S_WRITE     : begin");
                    v.block(|v| {
                        v.line("y_we = 1'b1;");
                        v.line("next = (oh == H_OUT - 1 && ow == W_OUT - 1 && oc == C_OUT - 1)");
                        v.line("       ? S_DONE : S_LOAD_BIAS;");
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
        out_len,
        sv: Artifact {
            rel_path: format!("layers/{name}.sv"),
            content: v.into_string(),
        },
        mems,
    }
}

/// Emit either two M0/shift ROMs (per-channel) or two `localparam`s
/// (when shared). Defines the `m0_q` and `sh_q` signals the requant
/// epilogue connects to.
fn emit_requant_source(
    v: &mut V,
    name: &str,
    requant: &[(i32, i32)],
    hints: &Hints,
    c_out: usize,
    addr_bits: usize,
) {
    let q15 = matches!(hints.requant_precision, RequantPrecision::Q0_15);
    let m0_width = if q15 { 16 } else { 32 };

    if let Some((m0_q31, shift)) = hints.shared_requant {
        v.comment("Requant table — shared (uniform across channels)");
        let (m0_lit, sh_lit): (i64, i32) = if q15 {
            let (m0, s) = q31_to_q15(m0_q31, shift);
            (m0 as i64, s)
        } else {
            (m0_q31 as i64, shift)
        };
        v.line(&format!(
            "localparam logic signed [{}:0] M0_VAL = {}'sd{};",
            m0_width - 1,
            m0_width,
            m0_lit
        ));
        v.line(&format!(
            "localparam logic        [7:0]  SHIFT_VAL = 8'd{sh_lit};"
        ));
        v.line(&format!("logic signed [{}:0] m0_q;", m0_width - 1));
        v.line("logic        [7:0]  sh_q;");
        v.line("assign m0_q = M0_VAL;");
        v.line("assign sh_q = SHIFT_VAL;");
    } else {
        v.comment("Requant ROMs (per-OC)");
        v.line(&format!("logic [{}:0] r_addr;", addr_bits - 1));
        v.line(&format!("logic signed [{}:0] m0_q;", m0_width - 1));
        v.line("logic        [7:0]  sh_q;");
        v.line(&format!("block_rom #(.WIDTH({m0_width}), .DEPTH({c_out}),"));
        v.line(&format!("  .INIT_FILE(\"weights/{name}_m0.mem\"))"));
        v.line("u_m0_rom (.clk(clk), .addr(r_addr), .dout(m0_q));");
        v.line(&format!("block_rom #(.WIDTH(8),  .DEPTH({c_out}),"));
        v.line(&format!("  .INIT_FILE(\"weights/{name}_sh.mem\"))"));
        v.line("u_sh_rom (.clk(clk), .addr(r_addr), .dout(sh_q));");
        v.line("always_comb r_addr = oc;");
        let _ = requant; // kept for future per-channel codegen
    }
}

/// Emit the combinational `mac_delta` signal — the per-cycle increment
/// `acc <= acc + mac_delta` consumes. The shape depends on hints:
///
/// * `ternary_fast_path` → `case (w_crumb)` over `xv_corrected`
/// * else                → `xv_corrected * w_corrected`
///
/// `xv_corrected` is `$signed(x_dout)` (fast_mac) or `... - X_ZP` (else).
/// `w_corrected` is `w_val` (fast_mac) or `w_val - W_ZP` (else).
fn emit_mac_delta(v: &mut V, hints: &Hints, x_zp_nonzero: bool, w_zp: i32) {
    v.comment("Per-cycle MAC delta");
    v.line("logic signed [31:0] xv_corrected;");
    if hints.fast_mac {
        v.line("assign xv_corrected = $signed({{24{x_dout[7]}}, x_dout});");
    } else {
        let zp = if x_zp_nonzero { "X_ZP" } else { "32'sd0" };
        v.line(&format!(
            "assign xv_corrected = $signed({{{{24{{x_dout[7]}}}}, x_dout}}) - {zp};"
        ));
    }

    v.line("logic signed [31:0] mac_delta;");
    if hints.ternary_fast_path {
        v.always_comb(|v| {
            v.line("unique case (w_crumb)");
            v.block(|v| {
                v.line("2'b00: mac_delta =  32'sd0;            // weight = 0");
                v.line("2'b01: mac_delta =  xv_corrected;      // weight = +1");
                v.line(
                    "2'b10: mac_delta = -(xv_corrected <<< 1); // weight = -2 (unused codepoint)",
                );
                v.line("2'b11: mac_delta = -xv_corrected;      // weight = -1");
            });
            v.line("endcase");
        });
    } else if hints.fast_mac && w_zp == 0 {
        v.line("assign mac_delta = xv_corrected * w_val;");
    } else {
        let zp = if w_zp != 0 { "W_ZP" } else { "32'sd0" };
        v.line(&format!(
            "assign mac_delta = xv_corrected * (w_val - {zp});"
        ));
    }
}

/// Emit Verilog assignments that split a logical weight index into a
/// byte address + lane index.
pub(crate) fn emit_byte_addr_lane(
    v: &mut V,
    byte_addr_signal: &str,
    lane_signal: &str,
    log_signal: &str,
    log_bits: usize,
    shift: usize,
) {
    match shift {
        0 => {
            v.line(&format!("{byte_addr_signal} = {log_signal};"));
            v.line(&format!("{lane_signal} = 2'd0;"));
        }
        1 => {
            v.line(&format!(
                "{byte_addr_signal} = {log_signal}[{}:1];",
                log_bits - 1
            ));
            v.line(&format!("{lane_signal} = {{1'b0, {log_signal}[0]}};"));
        }
        2 => {
            v.line(&format!(
                "{byte_addr_signal} = {log_signal}[{}:2];",
                log_bits - 1
            ));
            v.line(&format!("{lane_signal} = {log_signal}[1:0];"));
        }
        _ => unreachable!("emit_byte_addr_lane: shift must be 0/1/2"),
    }
}
