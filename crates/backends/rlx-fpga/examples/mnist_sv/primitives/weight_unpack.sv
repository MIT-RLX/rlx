// ───────────────────────────────────────────────────────────────
// weight_unpack — extract one logical weight from a packed byte
// ───────────────────────────────────────────────────────────────
// BITS ∈ {2, 4, 8}.  ENCODING: 0=signed-int, 1=FP4-E2M1 (BITS=4).
//   BITS=2  →  lane ∈ {0,1,2,3}  (4 crumbs per byte, LSB first)
//   BITS=4  →  lane ∈ {0,1}      (low nibble, then high nibble)
//   BITS=8  →  lane is ignored

module weight_unpack #(
    parameter int BITS = 8,
    parameter int ENCODING = 0
) (
    input  logic        [7:0]  byte_in,
    input  logic        [1:0]  lane,
    output logic signed [31:0] w_out
);
    generate
        if (ENCODING == 1 && BITS == 4) begin : g_fp4
            logic [3:0] nib;
            always_comb begin
                if (lane[0]) nib = byte_in[7:4];
                else         nib = byte_in[3:0];
            end
            // F4E2M1 decode × 2 → signed int for the MAC
            always_comb begin
                unique case (nib)
                    4'h0: w_out = 32'sd0;
                    4'h1: w_out = 32'sd1;
                    4'h2: w_out = 32'sd2;
                    4'h3: w_out = 32'sd3;
                    4'h4: w_out = 32'sd4;
                    4'h5: w_out = 32'sd6;
                    4'h6: w_out = 32'sd8;
                    4'h7: w_out = 32'sd12;
                    4'h8: w_out = 32'sd0;
                    4'h9: w_out = 32'sd-1;
                    4'hA: w_out = 32'sd-2;
                    4'hB: w_out = 32'sd-3;
                    4'hC: w_out = 32'sd-4;
                    4'hD: w_out = 32'sd-6;
                    4'hE: w_out = 32'sd-8;
                    4'hF: w_out = 32'sd-12;
                    default: w_out = 32'sd0;
                endcase
            end
        end else if (BITS == 8) begin : g_b8
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
            assign w_out = 32'sd0;
        end
    endgenerate
endmodule  // weight_unpack

