// ──────────────────────────────────────────────────────────────────────────
// conv2_kernel — INT8 conv2d 3x3 stride 1x1 on [13x13x8] → [11x11x16] (w8)
// ──────────────────────────────────────────────────────────────────────────
// x_zp=0 w_zp=0 out_zp=0; weights=1152 bytes (1152 logical, 8-bit); requant per-OC (16 entries).

module conv2_kernel (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       start,
    output logic                       done,
    output logic [10:0]              x_addr,
    input  logic signed [7:0]          x_dout,
    output logic [10:0]              y_addr,
    output logic                       y_we,
    output logic signed [7:0]          y_din
);
    localparam int H_IN=13, W_IN=13, C_IN=8;
    localparam int H_OUT=11, W_OUT=11, C_OUT=16;
    localparam int KH=3, KW=3, SH=1, SW=1;
    localparam int X_ZP=0, W_ZP=0, OUT_ZP=0;
    localparam int W_BITS=8;
    localparam int W_LOG_LEN=1152;
    localparam int W_BYTE_LEN=1152;

    // Weight ROM (byte-addressed; logical idx → byte idx via >>0)
    logic [10:0] w_addr;
    logic        [7:0] w_byte;
    block_rom #(.WIDTH(8), .DEPTH(W_BYTE_LEN),
      .INIT_FILE("weights/conv2_w.mem"))
    u_w_rom (.clk(clk), .addr(w_addr), .dout(w_byte));

    // Combinational weight unpack
    logic [1:0]              w_lane_q;
    logic signed [31:0]      w_val;
    weight_unpack #(.BITS(W_BITS)) u_w_unpack (
        .byte_in(w_byte),
        .lane(w_lane_q),
        .w_out(w_val)
    );

    // Bias ROM (i32)
    logic [3:0] b_addr;
    logic signed [31:0] b_dout;
    block_rom #(.WIDTH(32), .DEPTH(16),
      .INIT_FILE("weights/conv2_b.mem"))
    u_b_rom (.clk(clk), .addr(b_addr), .dout(b_dout));

    // Requant ROMs (M0 i32, shift u8 [low 5 bits used])
    logic [3:0] r_addr;
    logic signed [31:0] m0_dout;
    logic        [7:0]  sh_dout;
    block_rom #(.WIDTH(32), .DEPTH(16),
      .INIT_FILE("weights/conv2_m0.mem"))
    u_m0_rom (.clk(clk), .addr(r_addr), .dout(m0_dout));
    block_rom #(.WIDTH(8),  .DEPTH(16),
      .INIT_FILE("weights/conv2_sh.mem"))
    u_sh_rom (.clk(clk), .addr(r_addr), .dout(sh_dout));

    // Requantize epilogue (combinational)
    logic signed [7:0] q_out;
    requant_q31 u_requant (
        .acc(acc),
        .m0(m0_dout),
        .shift(sh_dout[4:0]),
        .out_zp(OUT_ZP),
        .q(q_out)
    );

    // Counters
    logic [3:0] oh, ow, oc;
    logic [1:0] kh_i, kw_i;
    logic [2:0] ic;
    logic signed [31:0] acc;

    // Address derivation
    logic [10:0] in_idx;
    logic [10:0] out_idx;
    logic [10:0] w_idx_logical;
    always_comb begin
        in_idx        = ((oh*SH + kh_i) * W_IN + (ow*SW + kw_i)) * C_IN + ic;
        out_idx       = (oh * W_OUT + ow) * C_OUT + oc;
        w_idx_logical = ((oc * KH + kh_i) * KW + kw_i) * C_IN + ic;
        w_addr = w_idx_logical;
        w_lane_q = 2'd0;
    end

    // FSM
    typedef enum logic [3:0] {
        S_IDLE, S_LOAD_BIAS, S_BIAS_WAIT, S_INIT_ACC, S_READ, S_WAIT, S_MAC, S_REQ_ADDR, S_REQ_WAIT, S_REQ_DO, S_WRITE, S_DONE
    } state_t;
    state_t state, next;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            oh<='0; ow<='0; oc<='0; kh_i<='0; kw_i<='0; ic<='0;
            acc<='0;
        end else begin
            state <= next;
            if (state == S_IDLE && start) begin
                oh<='0; ow<='0; oc<='0;
                kh_i<='0; kw_i<='0; ic<='0;
            end
            if (state == S_INIT_ACC) begin
                acc <= b_dout;
            end
            if (state == S_MAC) begin
                acc <= acc + ($signed({{24{x_dout[7]}}, x_dout}) - X_ZP)
                           * (w_val - W_ZP);
                // advance ic, kw_i, kh_i
                if (ic == C_IN - 1) begin
                    ic <= '0;
                    if (kw_i == KW - 1) begin
                        kw_i <= '0;
                        if (kh_i == KH - 1) begin
                            kh_i <= '0;  // window done
                        end else kh_i <= kh_i + 1;
                    end else kw_i <= kw_i + 1;
                end else ic <= ic + 1;
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
        r_addr  = oc;
        y_addr  = out_idx;
        y_we    = 1'b0;
        y_din   = q_out;
        done    = 1'b0;
        unique case (state)
            S_IDLE      : if (start) next = S_LOAD_BIAS;
            S_LOAD_BIAS : next = S_BIAS_WAIT;
            S_BIAS_WAIT : next = S_INIT_ACC;
            S_INIT_ACC  : next = S_READ;
            S_READ      : next = S_WAIT;
            S_WAIT      : next = S_MAC;
            S_MAC       : next = (kh_i == KH - 1 && kw_i == KW - 1 && ic == C_IN - 1)
                                 ? S_REQ_ADDR : S_READ;
            S_REQ_ADDR  : next = S_REQ_WAIT;
            S_REQ_WAIT  : next = S_REQ_DO;
            S_REQ_DO    : next = S_WRITE;
            S_WRITE     : begin
                y_we = 1'b1;
                next = (oh == H_OUT - 1 && ow == W_OUT - 1 && oc == C_OUT - 1)
                       ? S_DONE : S_LOAD_BIAS;
            end
            S_DONE      : begin done = 1'b1; if (!start) next = S_IDLE; end
        endcase
    end
endmodule  // conv2_kernel

