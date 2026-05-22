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
// ───────────────────────────────────────────────────────────
// requant_q15 — half-width Q0.15 epilogue (≤1 ulp vs Q0.31)
// ───────────────────────────────────────────────────────────
// Inputs:
//   acc     i32   accumulator
//   m0      i16   Q0.15 significand, in [2^14, 2^15)
//   shift   u4    right-shift count, in [0, 15]
//   out_zp  i32   output zero point
// Output:
//   q       i8

module requant_q15 (
    input  logic signed [31:0] acc,
    input  logic signed [15:0] m0,
    input  logic        [3:0]  shift,
    input  logic signed [31:0] out_zp,
    output logic signed [7:0]  q
);
    logic signed [47:0] ab;
    logic signed [47:0] nudge;
    logic signed [47:0] sum;
    logic signed [31:0] srdhm_out;

    always_comb begin
        ab    = $signed(acc) * $signed(m0);          // i32 × i16 → i48
        nudge = ab >= 0 ? 48'sd16384 : -48'sd16383;    // +2^14 / -(2^14 - 1)
        sum   = ab + nudge;
        // Truncate-toward-zero division by 2^15
        if (sum >= 0) srdhm_out = sum[46:15];
        else          srdhm_out = -((-sum) >>> 15);
    end

    logic signed [31:0] mask_v, rem_v, thresh_v, rdpot_out;
    always_comb begin
        if (shift == 4'd0) begin
            mask_v    = 32'sd0;
            rem_v     = 32'sd0;
            thresh_v  = 32'sd0;
            rdpot_out = srdhm_out;
        end else begin
            mask_v    = (32'sd1 <<< shift) - 32'sd1;
            rem_v     = srdhm_out & mask_v;
            thresh_v  = (mask_v >>> 1) + (srdhm_out < 0 ? 32'sd1 : 32'sd0);
            rdpot_out = (srdhm_out >>> shift) + ((rem_v > thresh_v) ? 32'sd1 : 32'sd0);
        end
    end

    logic signed [31:0] zp_added;
    always_comb begin
        zp_added = rdpot_out + out_zp;
        if      (zp_added >  32'sd127)   q = 8'sh7F;
        else if (zp_added < -32'sd128)   q = 8'sh80;
        else                              q = zp_added[7:0];
    end
endmodule  // requant_q15

