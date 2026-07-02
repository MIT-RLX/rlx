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

//! Q0.31 requantize epilogue, combinational. Bit-exact mirror of
//! `quant::requantize_q31` — same `srdhm` (truncate-toward-zero division
//! by 2^31), same rounding-divide-by-power-of-two, same saturating cast
//! to i8.

use crate::verilog::V;

/// Q0.15 variant: half-width epilogue. M0 is i16, the multiplier
/// fits in a single DSP slice on every modern part. Bit-exact match
/// for `quant::requantize_q15`.
pub fn emit_requant_q15() -> String {
    let mut v = V::new();
    v.banner("requant_q15 — half-width Q0.15 epilogue (≤1 ulp vs Q0.31)");
    v.comment("Inputs:");
    v.comment("  acc     i32   accumulator");
    v.comment("  m0      i16   Q0.15 significand, in [2^14, 2^15)");
    v.comment("  shift   u4    right-shift count, in [0, 15]");
    v.comment("  out_zp  i32   output zero point");
    v.comment("Output:");
    v.comment("  q       i8");
    v.blank();

    v.module(
        "requant_q15",
        &[],
        &[
            "input  logic signed [31:0] acc".into(),
            "input  logic signed [15:0] m0".into(),
            "input  logic        [3:0]  shift".into(),
            "input  logic signed [31:0] out_zp".into(),
            "output logic signed [7:0]  q".into(),
        ],
        |v| {
            v.line("logic signed [47:0] ab;");
            v.line("logic signed [47:0] nudge;");
            v.line("logic signed [47:0] sum;");
            v.line("logic signed [31:0] srdhm_out;");
            v.blank();
            v.always_comb(|v| {
                v.line("ab    = $signed(acc) * $signed(m0);          // i32 × i16 → i48");
                v.line("nudge = ab >= 0 ? 48'sd16384 : -48'sd16383;    // +2^14 / -(2^14 - 1)");
                v.line("sum   = ab + nudge;");
                v.comment("Truncate-toward-zero division by 2^15");
                v.line("if (sum >= 0) srdhm_out = sum[46:15];");
                v.line("else          srdhm_out = -((-sum) >>> 15);");
            });
            v.blank();
            v.line("logic signed [31:0] mask_v, rem_v, thresh_v, rdpot_out;");
            v.always_comb(|v| {
                v.line("if (shift == 4'd0) begin");
                v.block(|v| {
                    v.line("mask_v    = 32'sd0;");
                    v.line("rem_v     = 32'sd0;");
                    v.line("thresh_v  = 32'sd0;");
                    v.line("rdpot_out = srdhm_out;");
                });
                v.line("end else begin");
                v.block(|v| {
                    v.line("mask_v    = (32'sd1 <<< shift) - 32'sd1;");
                    v.line("rem_v     = srdhm_out & mask_v;");
                    v.line("thresh_v  = (mask_v >>> 1) + (srdhm_out < 0 ? 32'sd1 : 32'sd0);");
                    v.line("rdpot_out = (srdhm_out >>> shift) + ((rem_v > thresh_v) ? 32'sd1 : 32'sd0);");
                });
                v.line("end");
            });
            v.blank();
            v.line("logic signed [31:0] zp_added;");
            v.always_comb(|v| {
                v.line("zp_added = rdpot_out + out_zp;");
                v.line("if      (zp_added >  32'sd127)   q = 8'sh7F;");
                v.line("else if (zp_added < -32'sd128)   q = 8'sh80;");
                v.line("else                              q = zp_added[7:0];");
            });
        },
    );
    v.into_string()
}

pub fn emit_requant_q31() -> String {
    let mut v = V::new();
    v.banner("requant_q31 — Q0.31 requantize: srdhm + rounding-shift + sat_i8");
    v.comment("Inputs:");
    v.comment("  acc     i32   GEMM/conv accumulator");
    v.comment("  m0      i32   Q0.31 significand, in [2^30, 2^31)");
    v.comment("  shift   u5    right-shift count, in [0, 31]");
    v.comment("  out_zp  i32   output zero point");
    v.comment("Output:");
    v.comment("  q       i8    saturated, requantized code");
    v.comment("");
    v.comment("Mirrors gemmlowp / TFLite-Micro / CMSIS-NN. The lone");
    v.comment("saturation case for SRDHM is (i32::MIN, i32::MIN) →");
    v.comment("i32::MAX; we don't expect to hit it, but the guard is");
    v.comment("free and matches the Rust reference.");
    v.blank();

    v.module(
        "requant_q31",
        &[],
        &[
            "input  logic signed [31:0] acc".into(),
            "input  logic signed [31:0] m0".into(),
            "input  logic        [4:0]  shift".into(),
            "input  logic signed [31:0] out_zp".into(),
            "output logic signed [7:0]  q".into(),
        ],
        |v| {
            // SRDHM
            v.comment("─── SRDHM (saturating rounding doubling high mul) ───");
            v.line("logic signed [63:0] ab;");
            v.line("logic signed [63:0] nudge;");
            v.line("logic signed [63:0] sum;");
            v.line("logic signed [31:0] srdhm_out;");
            v.line("logic               srdhm_overflow;");
            v.blank();
            v.always_comb(|v| {
                v.line("ab    = acc * m0;                              // 64-bit signed");
                v.line("nudge = ab >= 0 ? 64'sd1073741824               // +2^30");
                v.line("                 : -64'sd1073741823;             // 1 - 2^30");
                v.line("sum   = ab + nudge;");
                v.comment("Truncate-toward-zero division by 2^31 (NOT >>>, which");
                v.comment("rounds toward -inf). For sum >= 0, identical to >> 31;");
                v.comment("for sum < 0, we negate, shift, then negate.");
                v.line("if (sum >= 0) begin");
                v.block(|v| v.line("srdhm_out = sum[62:31];"));
                v.line("end else begin");
                v.block(|v| v.line("srdhm_out = -((-sum) >>> 31);"));
                v.line("end");
                v.comment("Saturating overflow: only (MIN, MIN) wraps.");
                v.line("srdhm_overflow = (acc == 32'sh80000000) && (m0 == 32'sh80000000);");
                v.line("if (srdhm_overflow) srdhm_out = 32'sh7FFFFFFF;");
            });
            v.blank();

            // RDPOT
            v.comment("─── RDPOT (rounding divide by power of two) ───");
            v.comment("mask     = (1 << shift) - 1");
            v.comment("rem      = srdhm_out & mask");
            v.comment("threshold= (mask >> 1) + (srdhm_out < 0)");
            v.comment("rdpot_out= (srdhm_out >>> shift) + (rem > threshold)");
            v.line("logic signed [31:0] mask_v;");
            v.line("logic signed [31:0] rem_v;");
            v.line("logic signed [31:0] thresh_v;");
            v.line("logic signed [31:0] rdpot_out;");
            v.blank();
            v.always_comb(|v| {
                v.line("if (shift == 5'd0) begin");
                v.block(|v| {
                    v.line("mask_v    = 32'sd0;");
                    v.line("rem_v     = 32'sd0;");
                    v.line("thresh_v  = 32'sd0;");
                    v.line("rdpot_out = srdhm_out;");
                });
                v.line("end else begin");
                v.block(|v| {
                    v.line("mask_v   = (32'sd1 <<< shift) - 32'sd1;");
                    v.line("rem_v    = srdhm_out & mask_v;");
                    v.line("thresh_v = (mask_v >>> 1) + (srdhm_out < 0 ? 32'sd1 : 32'sd0);");
                    v.line("rdpot_out = (srdhm_out >>> shift)");
                    v.line("          + ((rem_v > thresh_v) ? 32'sd1 : 32'sd0);");
                });
                v.line("end");
            });
            v.blank();

            // Add zp + saturate
            v.comment("─── + out_zp, then saturate to i8 ───");
            v.line("logic signed [31:0] zp_added;");
            v.always_comb(|v| {
                v.line("zp_added = rdpot_out + out_zp;");
                v.line("if      (zp_added >  32'sd127)   q = 8'sh7F;");
                v.line("else if (zp_added < -32'sd128)   q = 8'sh80;");
                v.line("else                              q = zp_added[7:0];");
            });
        },
    );
    v.into_string()
}
