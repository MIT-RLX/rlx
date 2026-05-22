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
// ──────────────────────────────────────────────────────────
// top — tinyconv_mnist (arena: 2x BRAM @ SCRATCH_LEN=5408)
// ──────────────────────────────────────────────────────────
// Tune: Tune { fold_zp=true ternary_fast=true shared_requant=true bram_en=false requant=Q0_15 P=1 P_ic=1 }
// Pipeline (post-fusion):
//   L00  Conv2d    out_len=5408  slot 0→1
//   L01  MaxPool   out_len=1352  slot 1→0
//   L02  Conv2d    out_len=1936  slot 0→1
//   L03  MaxPool   out_len=400  slot 1→0
//   L04  Dense     out_len=10  slot 0→1
//   L05  Argmax    out_len=1  slot 1→0

module top (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       start,
    output logic                       done,
    input  logic [9:0]              in_addr,
    input  logic                       in_we,
    input  logic signed [7:0]          in_din,
    output logic signed [7:0]          pred
);
    localparam int SCRATCH_LEN = 5408;
    localparam int SCRATCH_AB  = 13;

    // ─── 2 ping-pong arena BRAMs ───
    logic [SCRATCH_AB-1:0] ar0_addr;
    logic                  ar0_we;
    logic signed [7:0]     ar0_din;
    logic signed [7:0]     ar0_dout;
    block_ram #(.WIDTH(8), .DEPTH(SCRATCH_LEN)) u_ar0 (
        .clk(clk), .we(ar0_we), .addr(ar0_addr),
        .din(ar0_din), .dout(ar0_dout)
    );
    logic [SCRATCH_AB-1:0] ar1_addr;
    logic                  ar1_we;
    logic signed [7:0]     ar1_din;
    logic signed [7:0]     ar1_dout;
    block_ram #(.WIDTH(8), .DEPTH(SCRATCH_LEN)) u_ar1 (
        .clk(clk), .we(ar1_we), .addr(ar1_addr),
        .din(ar1_din), .dout(ar1_dout)
    );

    // ─── per-layer kernel instances ───
    logic l0_start, l0_done;
    logic [9:0] l0_x_addr;
    logic [12:0] l0_y_addr;
    logic        l0_y_we;
    logic signed [7:0] l0_y_din;
    logic signed [7:0] l0_x_dout;
    conv1_kernel u_conv1 (
        .clk(clk), .rst(rst),
        .start(l0_start), .done(l0_done),
        .x_addr(l0_x_addr), .x_dout(l0_x_dout),
        .y_addr(l0_y_addr), .y_we(l0_y_we), .y_din(l0_y_din)
    );

    logic l1_start, l1_done;
    logic [12:0] l1_x_addr;
    logic [10:0] l1_y_addr;
    logic        l1_y_we;
    logic signed [7:0] l1_y_din;
    logic signed [7:0] l1_x_dout;
    pool1_kernel u_pool1 (
        .clk(clk), .rst(rst),
        .start(l1_start), .done(l1_done),
        .x_addr(l1_x_addr), .x_dout(l1_x_dout),
        .y_addr(l1_y_addr), .y_we(l1_y_we), .y_din(l1_y_din)
    );

    logic l2_start, l2_done;
    logic [10:0] l2_x_addr;
    logic [10:0] l2_y_addr;
    logic        l2_y_we;
    logic signed [7:0] l2_y_din;
    logic signed [7:0] l2_x_dout;
    conv2_kernel u_conv2 (
        .clk(clk), .rst(rst),
        .start(l2_start), .done(l2_done),
        .x_addr(l2_x_addr), .x_dout(l2_x_dout),
        .y_addr(l2_y_addr), .y_we(l2_y_we), .y_din(l2_y_din)
    );

    logic l3_start, l3_done;
    logic [10:0] l3_x_addr;
    logic [8:0] l3_y_addr;
    logic        l3_y_we;
    logic signed [7:0] l3_y_din;
    logic signed [7:0] l3_x_dout;
    pool2_kernel u_pool2 (
        .clk(clk), .rst(rst),
        .start(l3_start), .done(l3_done),
        .x_addr(l3_x_addr), .x_dout(l3_x_dout),
        .y_addr(l3_y_addr), .y_we(l3_y_we), .y_din(l3_y_din)
    );

    logic l4_start, l4_done;
    logic [8:0] l4_x_addr;
    logic [3:0] l4_y_addr;
    logic        l4_y_we;
    logic signed [7:0] l4_y_din;
    logic signed [7:0] l4_x_dout;
    fc_kernel u_fc (
        .clk(clk), .rst(rst),
        .start(l4_start), .done(l4_done),
        .x_addr(l4_x_addr), .x_dout(l4_x_dout),
        .y_addr(l4_y_addr), .y_we(l4_y_we), .y_din(l4_y_din)
    );

    logic l5_start, l5_done;
    logic [3:0] l5_x_addr;
    logic [0:0] l5_y_addr;
    logic        l5_y_we;
    logic signed [7:0] l5_y_din;
    logic signed [7:0] l5_x_dout;
    argmax_kernel u_argmax (
        .clk(clk), .rst(rst),
        .start(l5_start), .done(l5_done),
        .x_addr(l5_x_addr), .x_dout(l5_x_dout),
        .y_addr(l5_y_addr), .y_we(l5_y_we), .y_din(l5_y_din)
    );

    // ─── arena port routing — when stage == i, layer i drives slots ───
    always_comb begin
        ar0_addr = '0;
        ar0_we   = 1'b0;
        ar0_din  = 8'sd0;
        ar1_addr = '0;
        ar1_we   = 1'b0;
        ar1_din  = 8'sd0;
        if (!start && cstate == C_IDLE) begin
            ar0_addr = SCRATCH_AB'(in_addr);
            ar0_we   = in_we;
            ar0_din  = in_din;
        end else begin
            unique case (stage)
                0: begin
                    ar0_addr = SCRATCH_AB'(l0_x_addr);
                    ar1_addr = SCRATCH_AB'(l0_y_addr);
                    ar1_we   = l0_y_we;
                    ar1_din  = l0_y_din;
                end
                1: begin
                    ar1_addr = SCRATCH_AB'(l1_x_addr);
                    ar0_addr = SCRATCH_AB'(l1_y_addr);
                    ar0_we   = l1_y_we;
                    ar0_din  = l1_y_din;
                end
                2: begin
                    ar0_addr = SCRATCH_AB'(l2_x_addr);
                    ar1_addr = SCRATCH_AB'(l2_y_addr);
                    ar1_we   = l2_y_we;
                    ar1_din  = l2_y_din;
                end
                3: begin
                    ar1_addr = SCRATCH_AB'(l3_x_addr);
                    ar0_addr = SCRATCH_AB'(l3_y_addr);
                    ar0_we   = l3_y_we;
                    ar0_din  = l3_y_din;
                end
                4: begin
                    ar0_addr = SCRATCH_AB'(l4_x_addr);
                    ar1_addr = SCRATCH_AB'(l4_y_addr);
                    ar1_we   = l4_y_we;
                    ar1_din  = l4_y_din;
                end
                5: begin
                    ar1_addr = SCRATCH_AB'(l5_x_addr);
                    ar0_addr = SCRATCH_AB'(l5_y_addr);
                    ar0_we   = l5_y_we;
                    ar0_din  = l5_y_din;
                end
                default: ;
            endcase
        end
    end

    // ─── per-layer x_dout: route from the layer's input slot ───
    assign l0_x_dout = ar0_dout;
    assign l1_x_dout = ar1_dout;
    assign l2_x_dout = ar0_dout;
    assign l3_x_dout = ar1_dout;
    assign l4_x_dout = ar0_dout;
    assign l5_x_dout = ar1_dout;

    // Expose slot 0 as `pred` (last layer's output).
    assign pred = ar0_dout;

    // ───────────────────────────────────────────────────────────
    // controller — assert each layer's `start`, wait for `done`
    // ───────────────────────────────────────────────────────────
    logic [2:0] stage;
    typedef enum logic [1:0] {
        C_IDLE, C_RUN, C_STEP, C_DONE
    } ctrl_t;
    ctrl_t cstate, cnext;

    always_comb begin
        l0_start = (cstate == C_RUN) && (stage == 0);
        l1_start = (cstate == C_RUN) && (stage == 1);
        l2_start = (cstate == C_RUN) && (stage == 2);
        l3_start = (cstate == C_RUN) && (stage == 3);
        l4_start = (cstate == C_RUN) && (stage == 4);
        l5_start = (cstate == C_RUN) && (stage == 5);
        done = (cstate == C_DONE);
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            cstate <= C_IDLE;
            stage  <= '0;
        end else begin
            cstate <= cnext;
            if (cstate == C_IDLE && start) stage <= '0;
            if (cstate == C_STEP) stage <= stage + 1;
        end
    end

    always_comb begin
        cnext = cstate;
        unique case (cstate)
            C_IDLE : if (start) cnext = C_RUN;
            C_RUN  : begin
                if    (stage == 0 && l0_done) cnext = C_STEP;
                else if (stage == 1 && l1_done) cnext = C_STEP;
                else if (stage == 2 && l2_done) cnext = C_STEP;
                else if (stage == 3 && l3_done) cnext = C_STEP;
                else if (stage == 4 && l4_done) cnext = C_STEP;
                else if (stage == 5 && l5_done) cnext = C_STEP;
            end
            C_STEP : cnext = (stage == 5) ? C_DONE : C_RUN;
            C_DONE : if (!start) cnext = C_IDLE;
        endcase
    end
endmodule  // top

