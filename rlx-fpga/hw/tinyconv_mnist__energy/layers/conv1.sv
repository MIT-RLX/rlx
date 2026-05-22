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
// ─────────────────────────────────────────────────────────────────────────
// conv1_kernel — INT8 conv2d 3x3 stride 1x1 on [28x28x1] → [26x26x8] (w8)
// ─────────────────────────────────────────────────────────────────────────
// x_zp=0 w_zp=0 out_zp=0; weights=72 bytes (72 logical, 8-bit); requant per-OC (8 entries).
// hints: fast_mac + Q0_15 + fuses_relu

module conv1_kernel (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       start,
    output logic                       done,
    output logic [9:0]              x_addr,
    input  logic signed [7:0]          x_dout,
    output logic [12:0]              y_addr,
    output logic                       y_we,
    output logic signed [7:0]          y_din
);
    localparam int H_IN=28, W_IN=28, C_IN=1;
    localparam int H_OUT=26, W_OUT=26, C_OUT=8;
    localparam int KH=3, KW=3, SH=1, SW=1;
    localparam int X_ZP=0, W_ZP=0, OUT_ZP=0;
    localparam int W_BITS=8;
    localparam int W_LOG_LEN=72;
    localparam int W_BYTE_LEN=72;

    // Weight ROM (byte-addressed; logical idx → byte idx via >>0)
    logic [6:0] w_addr;
    logic        [7:0] w_byte;
    block_rom #(.WIDTH(8), .DEPTH(W_BYTE_LEN),
      .INIT_FILE("weights/conv1_w.mem"))
    u_w_rom (.clk(clk), .addr(w_addr), .dout(w_byte));

    logic [1:0] w_lane_q;
    // Combinational weight unpack
    logic signed [31:0] w_val;
    weight_unpack #(.BITS(W_BITS)) u_w_unpack (
        .byte_in(w_byte),
        .lane(w_lane_q),
        .w_out(w_val)
    );

    // Bias ROM (i32)
    logic [2:0] b_addr;
    logic signed [31:0] b_dout;
    block_rom #(.WIDTH(32), .DEPTH(8),
      .INIT_FILE("weights/conv1_b.mem"))
    u_b_rom (.clk(clk), .addr(b_addr), .dout(b_dout));

    // Requant ROMs (per-OC)
    logic [2:0] r_addr;
    logic signed [15:0] m0_q;
    logic        [7:0]  sh_q;
    block_rom #(.WIDTH(16), .DEPTH(8),
      .INIT_FILE("weights/conv1_m0.mem"))
    u_m0_rom (.clk(clk), .addr(r_addr), .dout(m0_q));
    block_rom #(.WIDTH(8),  .DEPTH(8),
      .INIT_FILE("weights/conv1_sh.mem"))
    u_sh_rom (.clk(clk), .addr(r_addr), .dout(sh_q));
    always_comb r_addr = oc;

    // Requantize epilogue (combinational)
    logic signed [7:0] q_raw;
    logic signed [7:0] q_out;
    requant_q15 u_requant (
        .acc(acc),
        .m0(m0_q[15:0]),
        .shift(sh_q[3:0]),
        .out_zp(OUT_ZP),
        .q(q_raw)
    );
    // fuses_relu: clamp at OUT_ZP (= relu zero_point)
    assign q_out = (q_raw < OUT_ZP[7:0]) ? OUT_ZP[7:0] : q_raw;

    // Counters
    logic [4:0] oh, ow, oc;
    logic [1:0] kh_i, kw_i;
    logic [0:0] ic;
    logic signed [31:0] acc;

    // Address derivation
    logic [9:0] in_idx;
    logic [12:0] out_idx;
    logic [6:0] w_idx_logical;
    always_comb begin
        in_idx        = ((oh*SH + kh_i) * W_IN + (ow*SW + kw_i)) * C_IN + ic;
        out_idx       = (oh * W_OUT + ow) * C_OUT + oc;
        w_idx_logical = ((oc * KH + kh_i) * KW + kw_i) * C_IN + ic;
        w_addr = w_idx_logical;
        w_lane_q = 2'd0;
    end

    // Per-cycle MAC delta
    logic signed [31:0] xv_corrected;
    assign xv_corrected = $signed({{24{x_dout[7]}}, x_dout});
    logic signed [31:0] mac_delta;
    assign mac_delta = xv_corrected * w_val;

    // Pipelined FSM — 1 MAC/cycle in S_PIPE
    //   prev_valid    : last cycle issued a real read (data is valid this cycle)
    //   done_issuing  : all addresses driven; remaining cycles drain the pipeline
    typedef enum logic [2:0] {
        S_IDLE, S_LOAD_BIAS, S_BIAS_WAIT, S_INIT_ACC, S_PIPE, S_WRITE, S_DONE
    } state_t;
    state_t state, next;
    logic prev_valid;
    logic done_issuing;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            oh<='0; ow<='0; oc<='0; kh_i<='0; kw_i<='0; ic<='0;
            acc<='0;
            prev_valid <= 1'b0;
            done_issuing <= 1'b0;
        end else begin
            state <= next;
            if (state == S_IDLE && start) begin
                oh<='0; ow<='0; oc<='0;
            end
            if (state == S_INIT_ACC) begin
                acc <= b_dout;
                kh_i <= '0; kw_i <= '0; ic <= '0;
                prev_valid <= 1'b0;
                done_issuing <= 1'b0;
            end
            if (state == S_PIPE) begin
                // (1) Accumulate the previous cycle's MAC if its read was real.
                if (prev_valid) acc <= acc + mac_delta;
                // (2) If still issuing, advance counters and mark issued.
                if (!done_issuing) begin
                    if (ic == C_IN - 1) begin
                        ic <= '0;
                        if (kw_i == KW - 1) begin
                            kw_i <= '0;
                            if (kh_i == KH - 1) begin
                                kh_i <= '0;
                                done_issuing <= 1'b1;  // last address driven
                            end else kh_i <= kh_i + 1;
                        end else kw_i <= kw_i + 1;
                    end else ic <= ic + 1;
                    prev_valid <= 1'b1;
                end else begin
                    prev_valid <= 1'b0;
                end
            end
            if (state == S_WRITE) begin
                if (oc == C_OUT - 1) begin
                    oc <= '0;
                    if (ow == W_OUT - 1) begin
                        ow <= '0;
                        oh <= oh + 1;
                    end else ow <= ow + 1;
                end else oc <= oc + 1;
            end
        end
    end

    always_comb begin
        next    = state;
        x_addr  = in_idx;
        b_addr  = oc;
        y_addr  = out_idx;
        y_we    = 1'b0;
        y_din   = q_out;
        done    = 1'b0;
        unique case (state)
            S_IDLE      : if (start) next = S_LOAD_BIAS;
            S_LOAD_BIAS : next = S_BIAS_WAIT;
            S_BIAS_WAIT : next = S_INIT_ACC;
            S_INIT_ACC  : next = S_PIPE;
            // S_PIPE exits when the last MAC has been accumulated:
            //   done_issuing means all reads have been issued; prev_valid means
            //   the data from the very last read is being consumed THIS cycle.
            S_PIPE      : next = (done_issuing && prev_valid) ? S_WRITE : S_PIPE;
            S_WRITE     : begin
                y_we = 1'b1;
                next = (oh == H_OUT - 1 && ow == W_OUT - 1 && oc == C_OUT - 1)
                       ? S_DONE : S_LOAD_BIAS;
            end
            S_DONE      : begin done = 1'b1; if (!start) next = S_IDLE; end
        endcase
    end
endmodule  // conv1_kernel

