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

//! Parallel (P > 1) conv2d kernel — output-channel parallelism.
//!
//! Each cycle in the inner loop:
//!
//! ```text
//!     1 activation read       (BRAM, 8 bits)
//!     P weight reads          (P banked weight ROMs, 8 bits each)
//!     P parallel multiplies   (×P MACs/cycle into P accumulators)
//! ```
//!
//! Cycles drop from `H·W·C_OUT·KH·KW·C_IN` (P=1) to roughly
//! `H·W·(C_OUT/P)·KH·KW·C_IN`, plus a sequential P-cycle requantize
//! epilogue per output pixel block.
//!
//! Weight / bias / requant tables are split into P lanes by `oc % P`
//! (see `pack::split_weights_by_oc_lane`), with one ROM per lane.
//! Layers where `C_OUT % P != 0` fall back to the scalar kernel — the
//! optimizer takes care of that in `passes::parallelism`.
//!
//! Restrictions in this first cut:
//! * `pad_h == pad_w == 0` (matches the scalar kernel).
//! * `weight_bits ∈ {2, 4, 8}` — handled by per-lane `weight_unpack`
//!   (or per-lane crumb-mux when `ternary_fast_path` is on).

use super::{Artifact, LayerArtifacts};
use crate::codegen::conv2d::emit_byte_addr_lane;
use crate::codegen::relu::bits_for;
use crate::model::Layer;
use crate::pack::{
    packed_byte_len, split_table_by_oc_lane, split_weights_by_oc_lane, weights_per_byte,
};
use crate::passes::Hints;
use crate::quant::q31_to_q15;
use crate::tune::RequantPrecision;
use crate::verilog::{V, mem_hex_bytes, mem_hex_words_i32};

pub fn emit(layer: &Layer, hints: &Hints) -> LayerArtifacts {
    let p = hints.parallelism.max(1) as usize;
    let p_ic_in = hints.ic_parallelism.max(1) as usize;
    assert!(
        p > 1 || p_ic_in > 1,
        "conv2d_parallel called with parallelism={p}, ic_parallelism={p_ic_in} \
             — use scalar emit instead"
    );

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
        _ => unreachable!("conv2d_parallel::emit on non-Conv2d layer"),
    };
    assert_eq!(pad_h, 0);
    assert_eq!(pad_w, 0);
    assert_eq!(
        c_out % p,
        0,
        "parallelism {p} must divide c_out {c_out} — optimizer should have caught this"
    );
    assert!(matches!(weight_bits, 2 | 4 | 8));
    assert!(!(hints.ternary_fast_path && weight_bits != 2));

    let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
    let inner = kh * kw * c_in; // logical weights per output channel
    let lane_logical_len = (c_out / p) * inner;
    let lane_byte_len = packed_byte_len(lane_logical_len, weight_bits);
    let in_len = h_in * w_in * c_in;
    let out_len = h_out * w_out * c_out;

    let module_name = format!("{name}_kernel");
    let instance_name = format!("u_{name}");

    // ── lane-split mem files ────────────────────────────────────────
    let mut mems = Vec::new();
    let weight_lanes = split_weights_by_oc_lane(&weights, c_out, inner, weight_bits, p);
    for (q, w) in weight_lanes.iter().enumerate() {
        debug_assert_eq!(w.len(), lane_byte_len);
        mems.push(Artifact {
            rel_path: format!("weights/{name}_w_l{q}.mem"),
            content: mem_hex_bytes(w),
        });
    }
    if let Some(b) = &bias {
        let lanes = split_table_by_oc_lane(b, p);
        for (q, lane_bias) in lanes.iter().enumerate() {
            mems.push(Artifact {
                rel_path: format!("weights/{name}_b_l{q}.mem"),
                content: mem_hex_words_i32(lane_bias, 32),
            });
        }
    }

    // requant lanes (Q0.31 raw, or convert to Q0.15 first)
    let q15 = matches!(hints.requant_precision, RequantPrecision::Q0_15);
    let m0_width: u32 = if q15 { 16 } else { 32 };
    if hints.shared_requant.is_none() {
        let m0_table: Vec<i32> = if q15 {
            requant
                .iter()
                .map(|(m, s)| q31_to_q15(*m, *s).0 as i32)
                .collect()
        } else {
            requant.iter().map(|(m, _)| *m).collect()
        };
        let sh_table: Vec<i32> = if q15 {
            requant.iter().map(|(m, s)| q31_to_q15(*m, *s).1).collect()
        } else {
            requant.iter().map(|(_, s)| *s).collect()
        };
        let m0_lanes = split_table_by_oc_lane(&m0_table, p);
        let sh_lanes = split_table_by_oc_lane(&sh_table, p);
        for q in 0..p {
            mems.push(Artifact {
                rel_path: format!("weights/{name}_m0_l{q}.mem"),
                content: mem_hex_words_i32(&m0_lanes[q], m0_width),
            });
            mems.push(Artifact {
                rel_path: format!("weights/{name}_sh_l{q}.mem"),
                content: mem_hex_words_i32(&sh_lanes[q], 8),
            });
        }
    }

    // ── module body ─────────────────────────────────────────────────
    let in_addr_bits = bits_for(in_len);
    let out_addr_bits = bits_for(out_len);
    let lane_byte_bits = bits_for(lane_byte_len);
    let lane_log_bits = bits_for(lane_logical_len);
    let oc_block_count = c_out / p;
    let oc_block_bits = bits_for(oc_block_count.max(1));
    let per_byte = weights_per_byte(weight_bits);
    let log2_per_byte = per_byte.trailing_zeros() as usize;
    let lane_bits = bits_for(p);

    // ic-parallelism: P_ic activations + crumbs consumed per cycle.
    // First-cut restriction: only enabled in the ternary fast path (so
    // each byte already holds P_ic crumbs, no ROM widening). Optimizer
    // should refuse non-ternary requests, but we re-assert here.
    let p_ic = hints.ic_parallelism.max(1) as usize;
    if p_ic > 1 {
        assert!(
            hints.ternary_fast_path,
            "ic_parallelism > 1 requires ternary_fast_path in this first cut"
        );
        assert!(
            weight_bits == 2,
            "ic_parallelism > 1 requires weight_bits = 2"
        );
        assert_eq!(
            c_in % p_ic,
            0,
            "c_in ({c_in}) must be divisible by p_ic ({p_ic})"
        );
        assert_eq!(
            p_ic, 4,
            "ic_parallelism currently only supports 4 (4 crumbs/byte)"
        );
    }

    let mut v = V::new();
    v.banner(&format!(
        "{module_name} — INT8 conv2d {kh}x{kw} stride {stride_h}x{stride_w} on [{h_in}x{w_in}x{c_in}] → [{h_out}x{w_out}x{c_out}] (w{weight_bits}) — P={p} parallel MACs"
    ));
    let tags: Vec<String> = std::iter::once(format!("P={p}"))
        .chain(
            [
                ("fast_mac", hints.fast_mac),
                ("ternary_fast_path", hints.ternary_fast_path),
                ("shared_requant", hints.shared_requant.is_some()),
                ("Q0_15", q15),
            ]
            .iter()
            .filter(|(_, on)| *on)
            .map(|(t, _)| t.to_string()),
        )
        .collect();
    v.comment(&format!("hints: {}", tags.join(" + ")));
    v.blank();

    v.module(
        &module_name,
        &[],
        &{
            let x_dout_decl = if p_ic > 1 {
                format!("input  logic [{}:0]              x_dout  // {p_ic} packed bytes",
                        8 * p_ic - 1)
            } else {
                "input  logic signed [7:0]          x_dout".into()
            };
            vec![
                "input  logic                       clk".into(),
                "input  logic                       rst".into(),
                "input  logic                       start".into(),
                "output logic                       done".into(),
                format!("output logic [{}:0]              x_addr", in_addr_bits - 1),
                x_dout_decl,
                format!("output logic [{}:0]              y_addr", out_addr_bits - 1),
                "output logic                       y_we".into(),
                "output logic signed [7:0]          y_din".into(),
            ]
        },
        |v| {
            v.line(&format!("localparam int H_IN={h_in}, W_IN={w_in}, C_IN={c_in};"));
            v.line(&format!("localparam int H_OUT={h_out}, W_OUT={w_out}, C_OUT={c_out};"));
            v.line(&format!("localparam int KH={kh}, KW={kw}, SH={stride_h}, SW={stride_w};"));
            v.line(&format!("localparam int X_ZP={x_zp}, W_ZP={w_zp}, OUT_ZP={out_zp};"));
            v.line(&format!("localparam int W_BITS={weight_bits};"));
            v.line(&format!("localparam int P={p};"));
            v.line(&format!("localparam int P_IC={p_ic};"));
            v.line(&format!("localparam int OC_BLOCKS={oc_block_count};"));
            v.line(&format!("localparam int LANE_BYTE_LEN={lane_byte_len};"));
            v.blank();

            v.comment("─── P weight ROMs (one per oc-lane) ───");
            v.line(&format!("logic [{}:0] w_addr;", lane_byte_bits.saturating_sub(1)));
            v.line("logic        [7:0] w_byte [0:P-1];");
            for q in 0..p {
                v.line("block_rom #(.WIDTH(8), .DEPTH(LANE_BYTE_LEN),");
                v.line(&format!("  .INIT_FILE(\"weights/{name}_w_l{q}.mem\"))"));
                v.line(&format!("u_w_rom_l{q} (.clk(clk), .addr(w_addr), .dout(w_byte[{q}]));"));
            }
            v.blank();

            // Lane index for sub-byte weight unpacking (shared across lanes — same lane_q value).
            v.line("logic [1:0] w_lane_q;");
            if hints.ternary_fast_path && p_ic > 1 {
                v.comment("Per-(oc, ic) crumb extraction — all 4 crumbs from each oc-lane's byte");
                v.line("logic [1:0] w_crumb [0:P-1][0:P_IC-1];");
                v.line("genvar gi, gj;");
                v.line("generate for (gi = 0; gi < P; gi = gi + 1) begin : g_crumb_oc");
                v.block(|v| {
                    v.line("for (gj = 0; gj < P_IC; gj = gj + 1) begin : g_crumb_ic");
                    v.block(|v| {
                        v.line("assign w_crumb[gi][gj] = w_byte[gi][2*gj +: 2];");
                    });
                    v.line("end");
                });
                v.line("end endgenerate");
            } else if hints.ternary_fast_path {
                v.comment("Per-lane crumb extraction (no DSP multiplier)");
                v.line("logic [1:0] w_crumb [0:P-1];");
                v.line("genvar gi;");
                v.line("generate for (gi = 0; gi < P; gi = gi + 1) begin : g_crumb");
                v.block(|v| {
                    v.always_comb(|v| {
                        v.line("unique case (w_lane_q)");
                        v.block(|v| {
                            v.line("2'd0: w_crumb[gi] = w_byte[gi][1:0];");
                            v.line("2'd1: w_crumb[gi] = w_byte[gi][3:2];");
                            v.line("2'd2: w_crumb[gi] = w_byte[gi][5:4];");
                            v.line("2'd3: w_crumb[gi] = w_byte[gi][7:6];");
                        });
                        v.line("endcase");
                    });
                });
                v.line("end endgenerate");
            } else {
                v.comment("Per-lane combinational weight unpack");
                v.line("logic signed [31:0] w_val [0:P-1];");
                v.line("genvar gi;");
                v.line("generate for (gi = 0; gi < P; gi = gi + 1) begin : g_unpack");
                v.block(|v| {
                    v.line("weight_unpack #(.BITS(W_BITS)) u_w_unpack (");
                    v.block(|v| {
                        v.line(".byte_in(w_byte[gi]),");
                        v.line(".lane(w_lane_q),");
                        v.line(".w_out(w_val[gi])");
                    });
                    v.line(");");
                });
                v.line("end endgenerate");
            }
            v.blank();

            // ── P bias ROMs (one per lane) ──
            v.comment("─── P bias ROMs ───");
            v.line(&format!("logic [{}:0] b_addr;", oc_block_bits.saturating_sub(1)));
            v.line("logic signed [31:0] b_dout [0:P-1];");
            if bias.is_some() {
                for q in 0..p {
                    v.line(&format!("block_rom #(.WIDTH(32), .DEPTH({oc_block_count}),"));
                    v.line(&format!("  .INIT_FILE(\"weights/{name}_b_l{q}.mem\"))"));
                    v.line(&format!("u_b_rom_l{q} (.clk(clk), .addr(b_addr), .dout(b_dout[{q}]));"));
                }
            } else {
                v.line("genvar gj;");
                v.line("generate for (gj = 0; gj < P; gj = gj + 1) begin : g_zero_bias");
                v.block(|v| v.line("assign b_dout[gj] = 32'sd0;"));
                v.line("end endgenerate");
            }
            v.blank();

            // ── Requant table sources ──
            v.comment("─── P requant ROMs (or shared localparam) ───");
            if let Some((m0_q31, shift_v)) = hints.shared_requant {
                let (m0_lit, sh_lit): (i64, i32) = if q15 {
                    let (m0, s) = q31_to_q15(m0_q31, shift_v);
                    (m0 as i64, s)
                } else {
                    (m0_q31 as i64, shift_v)
                };
                v.line(&format!("localparam logic signed [{}:0] M0_VAL = {}'sd{};",
                                m0_width - 1, m0_width, m0_lit));
                v.line(&format!("localparam logic        [7:0]  SHIFT_VAL = 8'd{sh_lit};"));
                v.line(&format!("logic signed [{}:0] m0_dout [0:P-1];", m0_width - 1));
                v.line("logic        [7:0]  sh_dout [0:P-1];");
                v.line("genvar gk;");
                v.line("generate for (gk = 0; gk < P; gk = gk + 1) begin : g_shared_req");
                v.block(|v| {
                    v.line("assign m0_dout[gk] = M0_VAL;");
                    v.line("assign sh_dout[gk] = SHIFT_VAL;");
                });
                v.line("end endgenerate");
            } else {
                v.line(&format!("logic [{}:0] r_addr;", oc_block_bits.saturating_sub(1)));
                v.line(&format!("logic signed [{}:0] m0_dout [0:P-1];", m0_width - 1));
                v.line("logic        [7:0]  sh_dout [0:P-1];");
                for q in 0..p {
                    v.line(&format!("block_rom #(.WIDTH({m0_width}), .DEPTH({oc_block_count}),"));
                    v.line(&format!("  .INIT_FILE(\"weights/{name}_m0_l{q}.mem\"))"));
                    v.line(&format!("u_m0_rom_l{q} (.clk(clk), .addr(r_addr), .dout(m0_dout[{q}]));"));
                    v.line(&format!("block_rom #(.WIDTH(8), .DEPTH({oc_block_count}),"));
                    v.line(&format!("  .INIT_FILE(\"weights/{name}_sh_l{q}.mem\"))"));
                    v.line(&format!("u_sh_rom_l{q} (.clk(clk), .addr(r_addr), .dout(sh_dout[{q}]));"));
                }
            }
            v.blank();

            // ── Sequential requantize epilogue: one requant unit muxed across lanes ──
            v.comment("Requantize epilogue (sequential through P lanes)");
            v.line(&format!("logic [{}:0] req_lane;", lane_bits.saturating_sub(1)));
            v.line("logic signed [31:0] acc_mux;");
            v.line(&format!("logic signed [{}:0] m0_mux;", m0_width - 1));
            v.line("logic        [7:0]  sh_mux;");
            v.always_comb(|v| {
                v.line("acc_mux = acc[req_lane];");
                v.line("m0_mux  = m0_dout[req_lane];");
                v.line("sh_mux  = sh_dout[req_lane];");
            });
            v.line("logic signed [7:0] q_raw;");
            v.line("logic signed [7:0] q_out;");
            if q15 {
                v.line("requant_q15 u_requant (");
                v.block(|v| {
                    v.line(".acc(acc_mux),");
                    v.line(".m0(m0_mux[15:0]),");
                    v.line(".shift(sh_mux[3:0]),");
                    v.line(".out_zp(OUT_ZP),");
                    v.line(".q(q_raw)");
                });
                v.line(");");
            } else {
                v.line("requant_q31 u_requant (");
                v.block(|v| {
                    v.line(".acc(acc_mux),");
                    v.line(".m0(m0_mux),");
                    v.line(".shift(sh_mux[4:0]),");
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

            // ── Counters ──
            v.line("logic signed [31:0] acc [0:P-1];");
            v.line(&format!("logic [{}:0] oh, ow;",
                            bits_for(h_out.max(w_out)) - 1));
            v.line(&format!("logic [{}:0] oc_block;",
                            oc_block_bits.saturating_sub(1)));
            v.line(&format!("logic [{}:0] kh_i, kw_i;", bits_for(kh.max(kw)) - 1));
            v.line(&format!("logic [{}:0] ic;", bits_for(c_in) - 1));
            v.blank();

            v.comment("Address derivation");
            v.line(&format!("logic [{}:0] in_idx;", in_addr_bits - 1));
            v.line(&format!("logic [{}:0] w_idx_logical;", lane_log_bits - 1));
            v.always_comb(|v| {
                v.line("in_idx = ((oh*SH + kh_i) * W_IN + (ow*SW + kw_i)) * C_IN + ic;");
                v.comment("Per-lane logical index (same for every lane — lane index lives outside)");
                v.line("w_idx_logical = (oc_block * KH + kh_i) * KW * C_IN + kw_i * C_IN + ic;");
                emit_byte_addr_lane(v, "w_addr", "w_lane_q", "w_idx_logical", lane_log_bits, log2_per_byte);
            });
            v.blank();

            // ── Per-lane MAC delta ──
            v.comment("Per-lane MAC delta (combinational)");
            if p_ic > 1 {
                // P_ic activations as bytes, sign-extended to i32
                v.line("logic signed [31:0] xv_corrected [0:P_IC-1];");
                v.line("genvar gx;");
                v.line("generate for (gx = 0; gx < P_IC; gx = gx + 1) begin : g_xv");
                v.block(|v| {
                    v.line("logic signed [7:0] xv_byte;");
                    v.line("assign xv_byte = $signed(x_dout[8*gx +: 8]);");
                    if hints.fast_mac {
                        v.line("assign xv_corrected[gx] = $signed({{24{xv_byte[7]}}, xv_byte});");
                    } else {
                        v.line("assign xv_corrected[gx] = $signed({{24{xv_byte[7]}}, xv_byte}) - X_ZP;");
                    }
                });
                v.line("end endgenerate");
                v.blank();
            } else {
                v.line("logic signed [31:0] xv_corrected;");
                if hints.fast_mac {
                    v.line("assign xv_corrected = $signed({{24{x_dout[7]}}, x_dout});");
                } else {
                    v.line("assign xv_corrected = $signed({{24{x_dout[7]}}, x_dout}) - X_ZP;");
                }
            }
            v.line("logic signed [31:0] mac_delta [0:P-1];");
            v.line("genvar gm;");
            v.line("generate for (gm = 0; gm < P; gm = gm + 1) begin : g_mac");
            v.block(|v| {
                if hints.ternary_fast_path && p_ic > 1 {
                    // P_ic partials per oc-lane, summed into one delta.
                    v.line("logic signed [31:0] partial [0:P_IC-1];");
                    v.line("genvar gp;");
                    v.line("generate for (gp = 0; gp < P_IC; gp = gp + 1) begin : g_partial");
                    v.block(|v| {
                        v.always_comb(|v| {
                            v.line("unique case (w_crumb[gm][gp])");
                            v.block(|v| {
                                v.line("2'b00: partial[gp] =  32'sd0;");
                                v.line("2'b01: partial[gp] =  xv_corrected[gp];");
                                v.line("2'b10: partial[gp] = -(xv_corrected[gp] <<< 1);");
                                v.line("2'b11: partial[gp] = -xv_corrected[gp];");
                            });
                            v.line("endcase");
                        });
                    });
                    v.line("end endgenerate");
                    v.line("// Reduction: sum all P_IC partials");
                    v.line("logic signed [31:0] sum_partial;");
                    v.always_comb(|v| {
                        v.line("sum_partial = 32'sd0;");
                        v.line("for (int s = 0; s < P_IC; s = s + 1) sum_partial = sum_partial + partial[s];");
                    });
                    v.line("assign mac_delta[gm] = sum_partial;");
                } else if hints.ternary_fast_path {
                    v.always_comb(|v| {
                        v.line("unique case (w_crumb[gm])");
                        v.block(|v| {
                            v.line("2'b00: mac_delta[gm] =  32'sd0;");
                            v.line("2'b01: mac_delta[gm] =  xv_corrected;");
                            v.line("2'b10: mac_delta[gm] = -(xv_corrected <<< 1);");
                            v.line("2'b11: mac_delta[gm] = -xv_corrected;");
                        });
                        v.line("endcase");
                    });
                } else if hints.fast_mac && w_zp == 0 {
                    v.line("assign mac_delta[gm] = xv_corrected * w_val[gm];");
                } else {
                    v.line("assign mac_delta[gm] = xv_corrected * (w_val[gm] - W_ZP);");
                }
            });
            v.line("end endgenerate");
            v.blank();

            // ── FSM ──
            v.comment("Pipelined FSM — 1 (P-wide) MAC/cycle in S_PIPE");
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
                    v.line("oh<='0; ow<='0; oc_block<='0;");
                    v.line("kh_i<='0; kw_i<='0; ic<='0;");
                    v.line("req_lane<='0;");
                    v.line("prev_valid <= 1'b0;");
                    v.line("done_issuing <= 1'b0;");
                    v.line("for (int i = 0; i < P; i = i + 1) acc[i] <= 32'sd0;");
                });
                v.line("end else begin");
                v.block(|v| {
                    v.line("state <= next;");
                    v.line("if (state == S_IDLE && start) begin");
                    v.block(|v| {
                        v.line("oh<='0; ow<='0; oc_block<='0;");
                        v.line("req_lane<='0;");
                    });
                    v.line("end");
                    v.line("if (state == S_INIT_ACC) begin");
                    v.block(|v| {
                        v.line("for (int i = 0; i < P; i = i + 1) acc[i] <= b_dout[i];");
                        v.line("kh_i <= '0; kw_i <= '0; ic <= '0;");
                        v.line("prev_valid <= 1'b0;");
                        v.line("done_issuing <= 1'b0;");
                    });
                    v.line("end");
                    v.line("if (state == S_PIPE) begin");
                    v.block(|v| {
                        v.line("if (prev_valid) for (int i = 0; i < P; i = i + 1)");
                        v.line("    acc[i] <= acc[i] + mac_delta[i];");
                        v.line("if (!done_issuing) begin");
                        v.block(|v| {
                            // ic advances by P_IC each cycle (P_IC=1 in the
                            // scalar / oc-only path, P_IC=4 in the ic-parallel
                            // ternary path).
                            v.line("if (ic == C_IN - P_IC) begin");
                            v.block(|v| {
                                v.line("ic <= '0;");
                                v.line("if (kw_i == KW - 1) begin");
                                v.block(|v| {
                                    v.line("kw_i <= '0;");
                                    v.line("if (kh_i == KH - 1) begin");
                                    v.block(|v| {
                                        v.line("kh_i <= '0;");
                                        v.line("done_issuing <= 1'b1;");
                                    });
                                    v.line("end else kh_i <= kh_i + 1;");
                                });
                                v.line("end else kw_i <= kw_i + 1;");
                            });
                            v.line("end else ic <= ic + P_IC;");
                            v.line("prev_valid <= 1'b1;");
                        });
                        v.line("end else begin");
                        v.block(|v| v.line("prev_valid <= 1'b0;"));
                        v.line("end");
                    });
                    v.line("end");
                    v.line("if (state == S_WRITE) begin");
                    v.block(|v| {
                        v.line("if (req_lane == P - 1) begin");
                        v.block(|v| {
                            v.line("req_lane <= '0;");
                            v.line("if (oc_block == OC_BLOCKS - 1) begin");
                            v.block(|v| {
                                v.line("oc_block <= '0;");
                                v.line("if (ow == W_OUT - 1) begin");
                                v.block(|v| {
                                    v.line("ow <= '0;");
                                    v.line("oh <= oh + 1;");
                                });
                                v.line("end else ow <= ow + 1;");
                            });
                            v.line("end else oc_block <= oc_block + 1;");
                        });
                        v.line("end else req_lane <= req_lane + 1;");
                    });
                    v.line("end");
                });
                v.line("end");
            });
            v.blank();

            v.always_comb(|v| {
                v.line("next   = state;");
                v.line("x_addr = in_idx;");
                v.line("b_addr = oc_block;");
                if hints.shared_requant.is_none() {
                    v.line("r_addr = oc_block;");
                }
                v.line("y_addr = (oh * W_OUT + ow) * C_OUT + oc_block * P + req_lane;");
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
                        v.line("if (req_lane == P - 1) begin");
                        v.block(|v| {
                            v.line("if (oh == H_OUT - 1 && ow == W_OUT - 1 && oc_block == OC_BLOCKS - 1)");
                            v.line("    next = S_DONE;");
                            v.line("else");
                            v.line("    next = S_LOAD_BIAS;");
                        });
                        v.line("end else next = S_WRITE;");
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
