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
// ─────────────────────────────────────────────────────────────────
// requant_q31 — Q0.31 requantize: srdhm + rounding-shift + sat_i8
// ─────────────────────────────────────────────────────────────────
// Inputs:
//   acc     i32   GEMM/conv accumulator
//   m0      i32   Q0.31 significand, in [2^30, 2^31)
//   shift   u5    right-shift count, in [0, 31]
//   out_zp  i32   output zero point
// Output:
//   q       i8    saturated, requantized code
// Mirrors gemmlowp / TFLite-Micro / CMSIS-NN. The lone
// saturation case for SRDHM is (i32::MIN, i32::MIN) →
// i32::MAX; we don't expect to hit it, but the guard is
// free and matches the Rust reference.

module requant_q31 (
    input  logic signed [31:0] acc,
    input  logic signed [31:0] m0,
    input  logic        [4:0]  shift,
    input  logic signed [31:0] out_zp,
    output logic signed [7:0]  q
);
    // ─── SRDHM (saturating rounding doubling high mul) ───
    logic signed [63:0] ab;
    logic signed [63:0] nudge;
    logic signed [63:0] sum;
    logic signed [31:0] srdhm_out;
    logic               srdhm_overflow;

    always_comb begin
        ab    = acc * m0;                              // 64-bit signed
        nudge = ab >= 0 ? 64'sd1073741824               // +2^30
                         : -64'sd1073741823;             // 1 - 2^30
        sum   = ab + nudge;
        // Truncate-toward-zero division by 2^31 (NOT >>>, which
        // rounds toward -inf). For sum >= 0, identical to >> 31;
        // for sum < 0, we negate, shift, then negate.
        if (sum >= 0) begin
            srdhm_out = sum[62:31];
        end else begin
            srdhm_out = -((-sum) >>> 31);
        end
        // Saturating overflow: only (MIN, MIN) wraps.
        srdhm_overflow = (acc == 32'sh80000000) && (m0 == 32'sh80000000);
        if (srdhm_overflow) srdhm_out = 32'sh7FFFFFFF;
    end

    // ─── RDPOT (rounding divide by power of two) ───
    // mask     = (1 << shift) - 1
    // rem      = srdhm_out & mask
    // threshold= (mask >> 1) + (srdhm_out < 0)
    // rdpot_out= (srdhm_out >>> shift) + (rem > threshold)
    logic signed [31:0] mask_v;
    logic signed [31:0] rem_v;
    logic signed [31:0] thresh_v;
    logic signed [31:0] rdpot_out;

    always_comb begin
        if (shift == 5'd0) begin
            mask_v    = 32'sd0;
            rem_v     = 32'sd0;
            thresh_v  = 32'sd0;
            rdpot_out = srdhm_out;
        end else begin
            mask_v   = (32'sd1 <<< shift) - 32'sd1;
            rem_v    = srdhm_out & mask_v;
            thresh_v = (mask_v >>> 1) + (srdhm_out < 0 ? 32'sd1 : 32'sd0);
            rdpot_out = (srdhm_out >>> shift)
                      + ((rem_v > thresh_v) ? 32'sd1 : 32'sd0);
        end
    end

    // ─── + out_zp, then saturate to i8 ───
    logic signed [31:0] zp_added;
    always_comb begin
        zp_added = rdpot_out + out_zp;
        if      (zp_added >  32'sd127)   q = 8'sh7F;
        else if (zp_added < -32'sd128)   q = 8'sh80;
        else                              q = zp_added[7:0];
    end
endmodule  // requant_q31

