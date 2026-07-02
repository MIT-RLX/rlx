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
// ───────────────────────────────────────────────────────────────
// weight_unpack — extract one logical weight from a packed byte
// ───────────────────────────────────────────────────────────────
// BITS ∈ {2, 4, 8}.  `lane` selects the sub-byte slot:
//   BITS=2  →  lane ∈ {0,1,2,3}  (4 crumbs per byte, LSB first)
//   BITS=4  →  lane ∈ {0,1}      (low nibble, then high nibble)
//   BITS=8  →  lane is ignored
// Output is sign-extended to 32 bits to feed the i32 MAC path.

module weight_unpack #(
    parameter int BITS = 8
) (
    input  logic        [7:0]  byte_in,
    input  logic        [1:0]  lane,
    output logic signed [31:0] w_out
);
    generate
        if (BITS == 8) begin : g_b8
            assign w_out = $signed({{24{byte_in[7]}}, byte_in});
        end else if (BITS == 4) begin : g_b4
            logic [3:0] nib;
            always_comb begin
                if (lane[0]) nib = byte_in[7:4];
                else         nib = byte_in[3:0];
            end
            assign w_out = $signed({{28{nib[3]}}, nib});
        end else if (BITS == 2) begin : g_b2
            logic [1:0] crumb;
            always_comb begin
                unique case (lane)
                    2'd0: crumb = byte_in[1:0];
                    2'd1: crumb = byte_in[3:2];
                    2'd2: crumb = byte_in[5:4];
                    2'd3: crumb = byte_in[7:6];
                endcase
            end
            assign w_out = $signed({{30{crumb[1]}}, crumb});
        end else begin : g_bad
            // $error must be inside an initial; for now compile-time fallback to 0
            assign w_out = 32'sd0;
        end
    endgenerate
endmodule  // weight_unpack

